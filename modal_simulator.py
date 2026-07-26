import time

import numpy as np
import ufl
import dolfinx
from mpi4py import MPI
from typing import List, Tuple
from pathlib import Path
from slepc4py import SLEPc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix

from mesh_loader import Mesh3DLoader
from material_library import ElasticMaterialLibrary


class ModalSimulator:
    """A Finite Element simulator for modal (natural frequency) analysis using dolfinx.

    This class handles the setup and solution of the undamped free-vibration
    eigenvalue problem of linear elasticity:

        K u = lambda * M u,   with   lambda = omega^2

    where K is the stiffness matrix, M the mass matrix, and the natural
    frequencies follow as f = sqrt(lambda) / (2 * pi).
    """

    def __init__(
        self,
        geometry_path: Path,
        materials: Tuple[Tuple[int, str], ...],
        displacement_bcs: Tuple[Tuple[int, Tuple[float, float, float]], ...] = None,
        num_modes: int = 6,
        elements_order: int = 2,
    ):
        """Initializes the modal simulator and loads the mesh.

        Parameters
        ----------
        geometry_path : Path
            Path to the .msh file.
        materials : Tuple[Tuple[int, str], ...]
            Collection of tuples: (volume_tag, material_name).
        displacement_bcs : Tuple[Tuple[int, Tuple[float, float, float]], ...], optional
            Tuples of (tag, (u_x, u_y, u_z)) marking fixed supports. In a modal
            analysis the constraint is always homogeneous — only the tag
            location matters, the values are applied as-is and should be zero.
            Omit entirely for a free-free analysis (rigid body modes are
            filtered out of the results).
        num_modes : int, optional
            Number of natural frequencies/mode shapes to compute, by default 6.
        elements_order : int, optional
            Polynomial order for Lagrange elements, by default 2.
        """
        self.geometry_path = geometry_path
        self.materials = materials
        self.elements_order = elements_order
        self.bcs = []
        self.dirichlet_bcs = displacement_bcs
        self.num_modes = num_modes
        self.lmbda = None
        self.mu = None
        self.rho = None

        self.load_mesh()
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.tags['cell'])

        self._function_space = None
        self._u = ufl.TrialFunction(self.function_space)
        self._v = ufl.TestFunction(self.function_space)
        self.k_form = None
        self.m_form = None

    def load_mesh(self) -> None:
        """Loads the mesh and extracts physical tags for volumes and surfaces."""
        mesh_loader = Mesh3DLoader(self.geometry_path.with_name(self.geometry_path.stem))
        self.mesh, self.tags = mesh_loader.load_mesh()

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """The Function Space of the elements for the modal problem"""
        if self._function_space is None:
            # For structural problems, we require a Vector Element space
            g_dim = self.mesh.geometry.dim
            self._function_space = fem.functionspace(self.mesh, ("Lagrange", self.elements_order, (g_dim,)))
        return self._function_space

    def epsilon(self, u):
        """Returns the symmetric strain tensor: 0.5 * (grad(u) + grad(u)^T)"""
        return ufl.sym(ufl.grad(u))

    def sigma(self, u):
        """Returns the Cauchy stress tensor using Hooke's Law
        (constitutive relation between stress tensor and strain tensor for
        linear elastic isotropic material)."""
        g_dim = self.mesh.geometry.dim
        return self.lmbda * ufl.tr(self.epsilon(u)) * ufl.Identity(g_dim) + 2.0 * self.mu * self.epsilon(u)

    def run(self) -> List[Tuple[float, dolfinx.fem.Function]]:
        """Executes the complete simulation workflow.

        This orchestrates material assignment, boundary condition application,
        variational form assembly, and the final eigensolver execution.

        Returns
        -------
        List[Tuple[float, dolfinx.fem.Function]]
            The computed modes as (natural_frequency_hz, mode_shape) pairs,
            sorted by ascending frequency. Each mode shape is normalized so
            its maximum displacement magnitude is 1.
        """
        self.lmbda, self.mu, self.rho = self.apply_materials()
        self.apply_dirichlet_bc()
        self.create_bilinear_functions()
        print("Solving the generalized eigenvalue problem...")
        return self.solve()

    def apply_dirichlet_bc(self) -> None:
        """Processes and stores Dirichlet boundary conditions (fixed supports).

        Locates degrees of freedom on the specified tags and populates the
        `self.bcs` list with dolfinx.fem.dirichletbc objects. Scans across all
        loaded tag dimensions (cell/surface/edge/point) to find whichever
        entity the tag lives on.
        """
        if self.dirichlet_bcs is None:
            print("[Modal] No supports supplied — running a free-free analysis.")
            return
        for tag, value in self.dirichlet_bcs:
            found = False

            for entity_type, tags_obj in self.tags.items():
                if tags_obj is None:
                    continue

                entities = tags_obj.find(tag)
                if len(entities) > 0:
                    ent_dim = tags_obj.dim

                    bc_vector = fem.Constant(self.mesh, dolfinx.default_scalar_type(value))
                    dofs = fem.locate_dofs_topological(self.function_space, ent_dim, entities)

                    self.bcs.append(fem.dirichletbc(bc_vector, dofs, self.function_space))
                    found = True
                    print(f"[Linked Dirichlet BC] Applied tag {tag} to '{entity_type}' region (Dimension: {ent_dim})")
                    break

            if not found:
                raise KeyError(
                    f"Dirichlet tag {tag} was not found in any loaded mesh tags "
                    f"(cells, surfaces, edges, or points). Verify your Gmsh physical group IDs."
                )

    def apply_materials(self) -> Tuple[dolfinx.fem.Function, dolfinx.fem.Function, dolfinx.fem.Function]:
        """Computes Lamé constants (lambda, mu) and density (rho) across volumes.

        Creates Discontinuous Galerkin (DG) functions to represent the
        material properties across different volume tags. Unlike the static
        simulator, the modal problem also needs the density field for the
        mass matrix.

        Returns
        -------
        Tuple[dolfinx.fem.Function, dolfinx.fem.Function, dolfinx.fem.Function]
            Spatially varying fields (lambda, mu, rho).
        """
        materials_library = ElasticMaterialLibrary()
        materials = [
            (tag, materials_library.get(material)) for tag, material in self.materials
        ]

        # constant value across each element
        disconnected_galerkin_space = fem.functionspace(self.mesh, ("DG", 0))
        global_lmbda_field = fem.Function(disconnected_galerkin_space)
        global_mu_field = fem.Function(disconnected_galerkin_space)
        global_rho_field = fem.Function(disconnected_galerkin_space)
        global_lmbda_field_values = global_lmbda_field.x.array
        global_mu_field_values = global_mu_field.x.array
        global_rho_field_values = global_rho_field.x.array

        for tag, material in materials:
            element_indices = self.tags['cell'].find(tag)
            global_lmbda_field_values[element_indices] = material.lmbda
            global_mu_field_values[element_indices] = material.mu
            global_rho_field_values[element_indices] = material.rho
        return global_lmbda_field, global_mu_field, global_rho_field

    def create_bilinear_functions(self) -> None:
        """Defines the stiffness and mass bilinear forms.

        Unlike the static/thermal simulators there is no linear form (RHS):
        the eigenvalue problem is fully described by the two bilinear forms
        k(u, v) = inner(sigma(u), epsilon(v)) and m(u, v) = rho * dot(u, v).
        """
        self.k_form = fem.form(ufl.inner(self.sigma(self._u), self.epsilon(self._v)) * self.dx)
        self.m_form = fem.form(self.rho * ufl.dot(self._u, self._v) * self.dx)

    def solve(self) -> List[Tuple[float, dolfinx.fem.Function]]:
        """Assembles K and M and solves K u = lambda * M u with SLEPc.

        Fixed-support DOFs are deflated by writing a huge value on the K
        diagonal and a tiny one on the M diagonal, which pushes their
        artificial eigenvalues far above the physical spectrum (standard
        trick — keeps M non-singular for the shift-and-invert transform).

        Returns
        -------
        List[Tuple[float, dolfinx.fem.Function]]
            (natural_frequency_hz, normalized_mode_shape) pairs, sorted by
            ascending frequency.
        """
        n_dofs = self.function_space.dofmap.index_map.size_global * self.function_space.dofmap.index_map_bs
        print(f"[Modal] Assembling stiffness and mass matrices ({n_dofs} DOFs)...", flush=True)
        # Artificial BC eigenvalue = 1e8 / 1e-8 = 1e16 -> f ~ 1.6e7 Hz, far above any physical mode
        K = assemble_matrix(self.k_form, bcs=self.bcs, diag=1e8)
        K.assemble()
        M = assemble_matrix(self.m_form, bcs=self.bcs, diag=1e-8)
        M.assemble()

        eps = SLEPc.EPS().create(self.mesh.comm)
        eps.setOperators(K, M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setDimensions(nev=self.num_modes)
        # Shift-and-invert around a low frequency to converge on the smallest
        # physical eigenvalues (and skip any rigid-body/BC artifacts).
        target_frequency_hz = 1.0
        eps.setTarget((2.0 * np.pi * target_frequency_hz) ** 2)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        ksp = st.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        # Symmetric Cholesky via MUMPS is the fastest factorization available
        # here — PETSc's built-in LU takes 2-3x longer on 3D problems. Even
        # so, expect roughly a minute per 100k DOFs (elements_order=2 on a
        # fine mesh); use elements_order=1 for quick runs.
        pc.setType("cholesky")
        pc.setFactorSolverType("mumps")
        solve_start = time.perf_counter()
        print("[Modal] Factorizing and solving the eigenvalue problem "
              "(this is the slow step — no output until it finishes)...", flush=True)
        eps.solve()
        print(f"[Modal] Eigensolver finished in {time.perf_counter() - solve_start:.1f} s.")

        n_converged = eps.getConverged()
        print(f"[Modal] Eigensolver converged {n_converged} eigenpairs.")

        # Rigid body modes (free-free analysis) show up as lambda ~ 0; skip them.
        rigid_body_tol = 1e-2

        modes = []
        eigenvector = K.createVecRight()
        for i in range(n_converged):
            eigenvalue = eps.getEigenpair(i, eigenvector)
            if eigenvalue.real < rigid_body_tol:
                continue
            frequency = np.sqrt(eigenvalue.real) / (2.0 * np.pi)

            mode_shape = fem.Function(self.function_space, name=f"mode_{len(modes) + 1}")
            n_local = eigenvector.getLocalSize()
            mode_shape.x.array[:n_local] = eigenvector.array_r
            mode_shape.x.scatter_forward()
            self._normalize_mode(mode_shape)

            modes.append((frequency, mode_shape))

        modes.sort(key=lambda pair: pair[0])
        modes = modes[:self.num_modes]
        for i, (frequency, _) in enumerate(modes, start=1):
            print(f"  Mode {i}: f = {frequency:.2f} Hz")
        print("Success!")
        return modes

    def _normalize_mode(self, mode_shape: dolfinx.fem.Function) -> None:
        """Rescales a mode shape in-place so its max displacement magnitude is 1.

        Eigenvector amplitudes are arbitrary; a unit maximum makes the shapes
        comparable across modes and convenient to scale in the plotter.
        """
        g_dim = self.mesh.geometry.dim
        magnitudes = np.linalg.norm(mode_shape.x.array.reshape((-1, g_dim)), axis=1)
        local_max = magnitudes.max() if magnitudes.size > 0 else 0.0
        global_max = self.mesh.comm.allreduce(local_max, op=MPI.MAX)
        if global_max > 0.0:
            mode_shape.x.array[:] /= global_max


def analytical_cantilever_frequencies(
    E: float, rho: float, nu: float, length: float, width: float, thickness: float, n_modes: int = 3
) -> List[Tuple[float, float, str]]:
    """Analytical cantilever bending frequencies: Euler-Bernoulli + Timoshenko.

    Euler-Bernoulli (thin-beam) frequency:

        f_EB = (beta_n * L)^2 / (2 * pi * L^2) * sqrt(E * I / (rho * A))

    Euler-Bernoulli assumes plane sections stay plane and perpendicular to the
    neutral axis, i.e. it ignores transverse shear deformation and rotary
    inertia. Both of those *soften* the beam, so f_EB over-estimates the true
    frequency. The error is not set by the overall slenderness L/t but by each
    mode's own bending wavelength: the shear term scales as (beta_n*L)^2*(r/L)^2,
    so it grows ~6x from mode 1 to mode 2 and is larger for the stiff (wide)
    plane (bigger radius of gyration r = sqrt(I/A)). That is why a converged 3D
    FEM sits well below f_EB for higher / wide-plane modes while still
    approaching the *true* solution from above. The first-order Timoshenko
    correction adds the softening back and is the fair yardstick for the FEM:

        f_T = f_EB / sqrt(1 + (beta_n*L)^2 * (r/L)^2 * (1 + E / (kappa * G)))

    with G = E / (2 (1 + nu)) and kappa = 5/6 (shear coefficient, rectangle).

    Both bending planes are evaluated (about the weak axis, I = w*t^3/12, and
    about the strong axis, I = t*w^3/12). Torsional and axial modes are not
    covered — expect the FEM results to contain extra modes between these.

    Returns
    -------
    List[Tuple[float, float, str]]
        (f_euler_bernoulli_hz, f_timoshenko_hz, description) sorted by ascending
        Timoshenko frequency.
    """
    beta_l = [1.87510407, 4.69409113, 7.85475744, 10.99554073][:n_modes]
    area = width * thickness
    shear_modulus = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0  # Timoshenko shear coefficient for a rectangular section

    frequencies = []
    bending_planes = (
        (width * thickness ** 3 / 12.0, "bending, thin direction"),
        (thickness * width ** 3 / 12.0, "bending, wide direction"),
    )
    for inertia, label in bending_planes:
        radius_gyration_sq = inertia / area  # r^2 = I / A
        for n, bl in enumerate(beta_l, start=1):
            f_eb = (bl ** 2 / (2.0 * np.pi * length ** 2)) * np.sqrt(E * inertia / (rho * area))
            shear_rotary = bl ** 2 * (radius_gyration_sq / length ** 2) * (1.0 + E / (kappa * shear_modulus))
            f_timoshenko = f_eb / np.sqrt(1.0 + shear_rotary)
            frequencies.append((f_eb, f_timoshenko, f"{label} #{n}"))

    frequencies.sort(key=lambda triple: triple[1])
    return frequencies


if __name__ == "__main__":
    from modal_plotter import ModeShapeViewer, ModeAnimationViewer, ModalStressViewer

    ##########################################################################
    # ### important note! ###
    # Before running this simulation, you have to convert the .msh file to
    # .xdmf and .h5 files, using the Msh2Xdmf class.
    ##########################################################################

    # Benchmark: cantilever beam (standard_fin3), clamped at the base surface.
    path = Path('/mnt/c/Users/saharl/Documents/simulations_api/simulations/test_dxf_exporter/standard_fin3.msh')
    materials = ((15, 'Aluminum-6061'), )
    fixed_supports = ((13, (0.0, 0.0, 0.0)), )

    # elements_order=1 runs in seconds and lands within ~2% of the order-2
    # frequencies on this mesh; order 2 has ~120k DOFs and takes ~1 min to solve.
    simulation = ModalSimulator(path, materials, fixed_supports, num_modes=6, elements_order=2)
    modes = simulation.run()

    # --- Analytical cantilever benchmark ------------------------------------
    # Compare the FEM against BOTH the Euler-Bernoulli and the Timoshenko
    # (shear + rotary-inertia corrected) frequencies. This bar is stubby
    # (L/t ~ 7.5), so E-B over-predicts the higher / wide-plane modes; the
    # Timoshenko column is the one the converged 3D FEM should track closely.
    coordinates = simulation.mesh.geometry.x
    extents = np.sort(coordinates.max(axis=0) - coordinates.min(axis=0))[::-1]
    length, width, thickness = extents  # mesh is already rescaled to meters
    material = ElasticMaterialLibrary.get('Aluminum-6061')
    analytical = analytical_cantilever_frequencies(
        material.E, material.rho, material.nu, length, width, thickness
    )

    print(f"\nBeam dimensions (m): L={length:.4f}, w={width:.4f}, t={thickness:.4f}")
    print("Analytical cantilever bending frequencies (Euler-Bernoulli vs Timoshenko):")
    for f_eb, f_timoshenko, label in analytical:
        print(f"  E-B {f_eb:10.2f} Hz | Timoshenko {f_timoshenko:10.2f} Hz  ({label})")
    print("FEM natural frequencies:")
    for i, (frequency, _) in enumerate(modes, start=1):
        print(f"  {frequency:10.2f} Hz  (mode {i})")

    # Continuous animation ("small video") of the fundamental mode. Saves a
    # looping GIF when headless, otherwise plays continuously in a window.
    # animation = ModeAnimationViewer(modes, simulation.function_space, mode_index=0)
    # animation.show(save_path="mode1_animation.gif")

    # Static (relative) modal stress field for the fundamental mode — shows
    # where that mode concentrates stress.
    stress = ModalStressViewer(modes, simulation.function_space, material.E, material.nu, mode_index=0)
    stress.show()

    viewer = ModeShapeViewer(modes, simulation.function_space)
    viewer.plotter.show()
