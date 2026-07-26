import ufl
import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from typing import Tuple, Callable, Union, Optional
from pathlib import Path
from dolfinx import default_scalar_type, fem
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector

from mesh_loader import Mesh3DLoader
from material_library import ThermalMaterialLibrary


# A boundary/source value may be a constant float or a callable f(t) -> float
# that returns the (spatially uniform) value at time t.
TimeValue = Union[float, Callable[[float], float]]


def eval_at(value: TimeValue, t: float) -> float:
    """Evaluate a possibly time-dependent value at time t.

    Returns ``value(t)`` if ``value`` is callable, otherwise ``value`` itself.
    A constant is its own theta-average, so constants and callables share one
    code path everywhere downstream.
    """
    return value(t) if callable(value) else value


class TransientThermalSimulator:
    """A Finite Element simulator for *transient* heat conduction using dolfinx.

    Solves the time-dependent heat equation

        rho * cp * dT/dt = div(k * grad(T)) + Q

    subject to Dirichlet (temperature), Neumann (heat flux) and Robin
    (convection) boundary conditions, using the generalized trapezoidal
    (theta) time-stepping scheme:

        theta = 0.5  -> Crank-Nicolson / trapezoidal rule (2nd order, A-stable)
        theta = 1.0  -> backward (implicit) Euler          (1st order, A-stable)
        theta = 0.0  -> forward (explicit) Euler           (conditionally stable)

    This mirrors ``ThermalSimulator`` (steady state) as closely as possible:
    same mesh loading, same BC tuple formats, same material library, same
    plotting stack. The Dirichlet temperature, heat flux and internal heat
    generation values may each be given as a constant *or* as a callable
    ``f(t)`` so they can vary in time (while staying uniform over their region).
    Because only right-hand-side quantities are allowed to vary in time (the
    convection coefficient h stays constant), the system matrix is assembled
    and LU-factorized once and reused for every step.
    """

    def __init__(
        self,
        geometry_path: Path,
        materials: Tuple[Tuple[int, str], ...],
        temperature_bcs: Tuple[Tuple[int, TimeValue], ...],
        dt: float,
        t_end: float,
        T_initial: float,
        convection_bcs: Tuple[Tuple[int, float], ...] = None,
        flux_bcs: Tuple[Tuple[int, TimeValue], ...] = None,
        elements_order: int = 2,
        T_amb: float = 25,
        internal_heat_generation: Tuple[Tuple[int, TimeValue], ...] = None,
        theta: float = 0.5,
        save_every: int = 1,
        output_path: Optional[Path] = None,
    ):
        """Initializes the simulator with physical/time parameters and loads the mesh.

        Parameters
        ----------
        geometry_path : Path
            Path to the .msh file (the sibling _volume/_surface/... XDMF files
            must already exist, exactly like ``ThermalSimulator``).
        materials : Tuple[Tuple[int, str], ...]
            Tuples of (volume_tag, material_name).
        temperature_bcs : Tuple[Tuple[int, float | Callable], ...]
            Tuples of (face_tag, temperature) for Dirichlet conditions. The
            temperature may be a constant or a callable f(t).
        dt : float
            Time step [s].
        t_end : float
            Total simulated time [s]. Number of steps = round(t_end / dt).
        T_initial : float
            Uniform initial temperature of the whole domain [same unit as BCs].
        convection_bcs : Tuple[Tuple[int, float], ...], optional
            Tuples of (face_tag, h_coefficient) for convection. Constant in time.
        flux_bcs : Tuple[Tuple[int, float | Callable], ...], optional
            Tuples of (face_tag, total_flux) for Neumann conditions. May be
            constant or a callable f(t).
        elements_order : int, optional
            Polynomial order for Lagrange elements, by default 2.
        T_amb : float, optional
            Ambient temperature for convection surfaces, by default 25. Constant.
        internal_heat_generation : Tuple[Tuple[int, float | Callable], ...], optional
            Tuples of (volume_tag, heat_generation). May be constant or f(t).
        theta : float, optional
            Time-stepping parameter (0.5 trapezoidal, 1.0 backward Euler),
            by default 0.5.
        save_every : int, optional
            Store/write every Nth step (plus the initial state), by default 1.
        output_path : Path, optional
            If given, an XDMF time series is written here for ParaView.
        """
        self.geometry_path = geometry_path
        self.materials = materials
        self.elements_order = elements_order
        self.bcs = []
        self.dirichlet_bcs = temperature_bcs
        self.robin_bcs = convection_bcs
        self.neumann_bcs = flux_bcs
        self.T_amb = T_amb
        self.internal_heat_generation = internal_heat_generation

        self.dt = dt
        self.t_end = t_end
        self.T_initial = T_initial
        self.theta = theta
        self.save_every = save_every
        self.output_path = output_path

        self.conduction_coeff = None
        self.rhocp_coeff = None

        # Populated during run().
        self._dirichlet_updates = []   # list of (fem.Constant, TimeValue)
        self._load_updates = []        # list of (fem.Constant, TimeValue, kind, area)
        self.T_n = None                # previous-step temperature (fem.Function)
        self.times = []                # recorded time levels
        self.history = []              # recorded dof arrays (one per saved step)

        self.load_mesh()
        # Automatically detects boundary facets: dimension 2 (surfaces) in 3D
        # meshes, or dimension 1 (edges) in 2D meshes.
        facet_dim = self.mesh.topology.dim - 1
        facet_tags = None
        for tags_obj in self.tags.values():
            if tags_obj is not None and tags_obj.dim == facet_dim:
                facet_tags = tags_obj
                break

        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tags)
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.tags['cell'])

        self._function_space = None
        self._u = ufl.TrialFunction(self.function_space)
        self._v = ufl.TestFunction(self.function_space)
        self.a = 0
        self.L = 0

    def load_mesh(self) -> None:
        """Loads the mesh and extracts physical tags for volumes and surfaces."""
        mesh_loader = Mesh3DLoader(self.geometry_path.with_name(self.geometry_path.stem))
        self.mesh, self.tags = mesh_loader.load_mesh()

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """The Function Space of the elements for the thermal problem."""
        if self._function_space is None:
            self._function_space = fem.functionspace(self.mesh, ("Lagrange", self.elements_order))
        return self._function_space

    def run(self) -> dolfinx.fem.Function:
        """Executes the complete transient simulation workflow.

        Returns
        -------
        dolfinx.fem.Function
            The final temperature field (after the last time step). The full
            time history is available on ``self.history`` / ``self.times``.
        """
        self.conduction_coeff, self.rhocp_coeff = self.apply_materials()
        self.apply_dirichlet_bc()
        self.build_forms(self.conduction_coeff, self.rhocp_coeff)
        print("Solving the transient problem...")
        return self.solve_time_dependent()

    def apply_dirichlet_bc(self) -> None:
        """Processes Dirichlet boundary conditions (fixed temperature).

        Locates degrees of freedom on the specified tags and populates the
        ``self.bcs`` list with dolfinx.fem.dirichletbc objects. Each BC is
        backed by a ``fem.Constant`` whose value is refreshed at every time
        step (constant values simply never change). Scans across all loaded
        tag dimensions (cell/surface/edge/point) to find whichever entity the
        tag lives on.
        """
        for tag, value in self.dirichlet_bcs:
            found = False

            for entity_type, tags_obj in self.tags.items():
                if tags_obj is None:
                    continue

                entities = tags_obj.find(tag)
                if len(entities) > 0:
                    ent_dim = tags_obj.dim
                    dofs = fem.locate_dofs_topological(self.function_space, ent_dim, entities)
                    bc_const = fem.Constant(self.mesh, default_scalar_type(eval_at(value, 0.0)))
                    self.bcs.append(fem.dirichletbc(bc_const, dofs, self.function_space))
                    self._dirichlet_updates.append((bc_const, value))
                    found = True
                    print(f"[Linked Dirichlet BC] Applied tag {tag} to '{entity_type}' region (Dimension: {ent_dim})")
                    break

            if not found:
                raise KeyError(
                    f"Dirichlet tag {tag} was not found in any loaded mesh tags "
                    f"(cells, surfaces, edges, or points). Verify your Gmsh physical group IDs."
                )

    def apply_materials(self) -> Tuple[dolfinx.fem.Function, dolfinx.fem.Function]:
        """Maps material properties from the library onto the mesh subdomains.

        Creates two Discontinuous Galerkin (DG0, cell-wise constant) fields:
        the thermal conductivity ``k`` and the volumetric heat capacity
        ``rho * cp`` (needed for the transient mass term).

        Returns
        -------
        (conduction_coeff, rhocp_coeff) : Tuple[dolfinx.fem.Function, dolfinx.fem.Function]
            Spatially varying conductivity and rho*cp fields.
        """
        materials_library = ThermalMaterialLibrary()
        materials = [
            (tag, materials_library.get(material)) for tag, material in self.materials
        ]

        # constant value across each element
        disconnected_galerkin_space = fem.functionspace(self.mesh, ("DG", 0))
        global_conduction_coeff = fem.Function(disconnected_galerkin_space)
        global_rhocp_coeff = fem.Function(disconnected_galerkin_space)
        k_values = global_conduction_coeff.x.array
        rhocp_values = global_rhocp_coeff.x.array

        for tag, material in materials:
            cells = self.tags['cell'].find(tag)
            k_values[cells] = material.k
            rhocp_values[cells] = material.rho * material.cp
        return global_conduction_coeff, global_rhocp_coeff

    def build_forms(self, conduction_coeff, rhocp_coeff) -> None:
        """Assembles the trapezoidal bilinear (a) and linear (L) forms.

        With the mass form ``m(u,v) = rho*cp * u * v dx`` and the steady
        operator ``K(u,v) = k grad(u).grad(v) dx + sum_robin h u v ds``::

            a(u,v) = (1/dt) m(u,v) + theta K(u,v)
            L(v)   = (1/dt) m(T_n,v) - (1-theta) K(T_n,v)
                     + sum_robin h*T_amb*v*ds                 (constant load)
                     + sum_src Q_const*v*dx(tag)              (theta-averaged)
                     + sum_neu q_const*v*ds(tag)              (theta-averaged)

        The time-varying source/flux magnitudes live inside ``fem.Constant``
        objects (``self._load_updates``) refreshed each step; a constant value
        is its own theta-average so both paths are identical here.
        """
        u, v = self._u, self._v
        theta = self.theta
        inv_dt = 1.0 / self.dt

        self.T_n = fem.Function(self.function_space)
        self.T_n.x.array[:] = self.T_initial

        # Mass and stiffness (conduction) operators.
        mass = rhocp_coeff * u * v * ufl.dx
        stiffness = conduction_coeff * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

        # LHS: (1/dt) m + theta K(conduction).
        self.a = inv_dt * mass + theta * stiffness

        # RHS: (1/dt) m(T_n) - (1-theta) K(T_n)(conduction).
        self.L = (
            inv_dt * rhocp_coeff * self.T_n * v * ufl.dx
            - (1.0 - theta) * conduction_coeff * ufl.dot(ufl.grad(self.T_n), ufl.grad(v)) * ufl.dx
        )

        # Robin / convection: h*u*v on the LHS operator, its T_n counterpart on
        # the RHS, and the constant ambient load h*T_amb*v (never time-varying).
        if self.robin_bcs is not None:
            for tag, h in self.robin_bcs:
                h_const = fem.Constant(self.mesh, default_scalar_type(h))
                self.a += theta * h_const * u * v * self.ds(tag)
                self.L += -(1.0 - theta) * h_const * self.T_n * v * self.ds(tag)
                self.L += h_const * self.T_amb * v * self.ds(tag)

        # Internal heat generation (volumetric source). Time-varying magnitude
        # captured by a Constant refreshed each step.
        if self.internal_heat_generation is not None:
            for tag, q_gen in self.internal_heat_generation:
                src_const = fem.Constant(self.mesh, default_scalar_type(eval_at(q_gen, 0.0)))
                self.L += src_const * v * self.dx(tag)
                self._load_updates.append((src_const, q_gen, "source", None))

        # Neumann / heat flux (total flux over the tagged surface, divided by
        # the MPI-reduced surface area to get a flux density).
        if self.neumann_bcs is not None:
            for tag, q_flux in self.neumann_bcs:
                area_form = fem.form(fem.Constant(self.mesh, 1.0) * self.ds(tag))
                area_local = fem.assemble_scalar(area_form)
                area_total = self.mesh.comm.allreduce(area_local, op=MPI.SUM)

                flux_const = fem.Constant(
                    self.mesh, default_scalar_type(eval_at(q_flux, 0.0) / area_total)
                )
                self.L += flux_const * v * self.ds(tag)
                self._load_updates.append((flux_const, q_flux, "neumann", area_total))

    def _update_time_dependent_data(self, t_old: float, t_new: float) -> None:
        """Refreshes all time-varying Constants for the step t_old -> t_new.

        RHS loads use the theta-averaged value ``theta*f(t_new) + (1-theta)*f(t_old)``;
        Dirichlet values are imposed at the new time level t_new.
        """
        theta = self.theta
        for const, value, kind, area in self._load_updates:
            averaged = theta * eval_at(value, t_new) + (1.0 - theta) * eval_at(value, t_old)
            if kind == "neumann":
                const.value = averaged / area
            else:
                const.value = averaged

        for const, value in self._dirichlet_updates:
            const.value = eval_at(value, t_new)

    def solve_time_dependent(self) -> dolfinx.fem.Function:
        """Runs the time-stepping loop, reusing one LU factorization throughout.

        Assembles the (constant) system matrix once, then at each step rebuilds
        only the RHS vector, applies the boundary conditions and solves. Records
        the temperature history and optionally writes an XDMF time series.
        """
        a_form = fem.form(self.a)
        L_form = fem.form(self.L)

        # Constant system matrix -> assemble + factorize once.
        A = assemble_matrix(a_form, bcs=self.bcs)
        A.assemble()
        b = create_vector(self.function_space)

        solver = PETSc.KSP().create(self.mesh.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        T = fem.Function(self.function_space, name="Temperature")
        T.x.array[:] = self.T_initial

        n_steps = int(round(self.t_end / self.dt))

        # Optional XDMF time-series output (interpolate to P1 for writing).
        xdmf = None
        T_out = None
        if self.output_path is not None:
            xdmf = XDMFFile(self.mesh.comm, str(self.output_path), "w")
            xdmf.write_mesh(self.mesh)
            V_out = fem.functionspace(self.mesh, ("Lagrange", 1))
            T_out = fem.Function(V_out, name="Temperature")

        def record(t: float, u: dolfinx.fem.Function) -> None:
            self.times.append(t)
            self.history.append(u.x.array.copy())
            if xdmf is not None:
                T_out.interpolate(u)
                xdmf.write_function(T_out, t)

        # Record the initial state as the first frame.
        record(0.0, self.T_n)

        for step in range(1, n_steps + 1):
            t_old = (step - 1) * self.dt
            t_new = step * self.dt
            self._update_time_dependent_data(t_old, t_new)

            # Rebuild only the RHS.
            with b.localForm() as loc_b:
                loc_b.set(0.0)
            assemble_vector(b, L_form)
            apply_lifting(b, [a_form], bcs=[self.bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, self.bcs)

            solver.solve(b, T.x.petsc_vec)
            T.x.scatter_forward()

            # Roll forward: T_n <- T.
            self.T_n.x.array[:] = T.x.array

            if step % self.save_every == 0 or step == n_steps:
                record(t_new, T)
            print(f"  step {step}/{n_steps}  t = {t_new:.4g} s")

        if xdmf is not None:
            xdmf.close()
            print(f"Wrote XDMF time series to {self.output_path}")

        solver.destroy()
        A.destroy()
        b.destroy()
        print("Success!")
        return T


if __name__ == "__main__":
    from thermal_plotter import VolumeViewer, TimeSeriesViewer

    ##########################################################################
    # ### important note! ###
    # Before running this simulation, you have to convert the .msh file to
    # .xdmf and .h5 files, using the Msh2Xdmf class.
    ##########################################################################

    path = Path('/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/test/Assem1.msh')
    materials = ((51, 'Aluminium-6061'), (52, 'Copper'))

    # Time-varying Dirichlet: the surface ramps from 25 C to 120 C over 60 s,
    # then holds. Any callable f(t) -> float works here.
    ramp = lambda t: 25.0 + 95.0 * min(t / 60.0, 1.0)
    dirichlet_bc = ((49, ramp), )

    simulation = TransientThermalSimulator(
        path,
        materials,
        dirichlet_bc,
        dt=20.0,
        t_end=5000.0,   # this part's thermal time constant is ~1000 s
        T_initial=25.0,
        convection_bcs=((50, 5), ),
        internal_heat_generation=((52, 1.0e5), (51, 3e5)),
        T_amb=25,
        theta=0.5,       # 0.5 = trapezoidal (Crank-Nicolson); 1.0 = backward Euler
        save_every=1,
        # Note: avoid the "<meshstem>_*.xdmf" naming — Mesh3DLoader globs that
        # pattern for mesh components, so a hyphen keeps the output out of its way.
        output_path=path.with_name("Assem1-transient.xdmf"),
    )
    u = simulation.run()

    # Final field in the existing viewer.
    volume_viewer = VolumeViewer(u, simulation.function_space, to_probe=True)
    volume_viewer.plotter.show()

    # Scrub to any t_n and play a movie of all frames.
    time_viewer = TimeSeriesViewer(
        simulation.function_space, simulation.history, simulation.times, to_probe=True
    )
    time_viewer.plotter.show()
