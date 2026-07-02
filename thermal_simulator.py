import ufl
import dolfinx
from mpi4py import MPI
from typing import Tuple
from pathlib import Path
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type, fem

from mesh_loader import Mesh3DLoader
from material_library import ThermalMaterialLibrary
from thermal_fins_motor import plot_3d


class ThermalSimulator:
    """A Finite Element simulator for steady-state heat conduction using dolfinx.
    
    This class handles the setup and solution of the heat equation:
    -k * ∇²T = Q
    where k is thermal conductivity and Q is internal heat generation, subject to 
    Dirichlet (temperature), Neumann (heat flux), and Robin (convection) boundary conditions.
    """
    
    def __init__(
        self,
        geometry_path: Path,
        materials: Tuple[Tuple[int, str], ...],
        temperature_bcs: Tuple[Tuple[int, float], ...],
        convection_bcs: Tuple[Tuple[int, float], ...] = None,
        flux_bcs: Tuple[Tuple[int, float], ...] = None,
        elements_order: int = 2,
        T_amb: float = 25,
        internal_heat_generation: Tuple[Tuple[int, float], ...] = None,
    ):
        """Initializes the simulator with physical parameters and loads the mesh.

        Parameters
        ----------
        geometry_path : Path
            Path to the .msh file.
        materials : Tuple[Tuple[int, str], ...]
            A collection of tuples where each contains a (volume_tag, material_name).
        temperature_bcs : Tuple[Tuple[int, float], ...]
            Tuples of (face_tag, temperature_value) for Dirichlet conditions.
        convection_bcs : Tuple[Tuple[int, float], ...], optional
            Tuples of (face_tag, h_coefficient) for convection, by default None.
        flux_bcs : Tuple[Tuple[int, float], ...], optional
            Tuples of (face_tag, total_flux) for Neumann conditions, by default None.
        elements_order : int, optional
            Polynomial order for Lagrange elements, by default 2.
        T_amb : float, optional
            Environmental temperature for rubin surfaces, by default 25.
        internal_heat_generation : Tuple[Tuple[int, float], ...], optional
            Tuples of (volume_tag, heat_generation_value), by default None.
        """
        self.geometry_path = geometry_path
        self.materials = materials
        self.elements_order = elements_order
        self.bcs = []
        self.dirichlet_bcs = temperature_bcs
        self.rubin_bcs = convection_bcs
        self.neuman_bcs = flux_bcs
        self.T_amb = T_amb
        self.internal_heat_generation = internal_heat_generation
        
        self.load_mesh()
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.face_tags)
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.volume_tag)
        
        self._function_space = None
        self._u = ufl.TrialFunction(self.function_space)
        self._v = ufl.TestFunction(self.function_space)
        self.a = 0
        self.L = 0

    def load_mesh(self) -> None:
        """Loads the mesh and extracts physical tags for volumes and surfaces."""
        mesh_loader = Mesh3DLoader(self.geometry_path.with_name(self.geometry_path.stem))
        self.mesh, (self.volume_tag, self.face_tags) = mesh_loader.load_mesh()
    
    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """The Function Space of the elements for the thermal problem"""
        if self._function_space is None:
            # Define a standard Lagrange element for Heat Transfer
            self._function_space = fem.functionspace(self.mesh, ("Lagrange", self.elements_order))
        return self._function_space

    def run(self) -> dolfinx.fem.Function:
        """Executes the complete simulation workflow.

        This orchestrates material assignment, boundary condition application,
        variational form assembly, and the final solver execution.

        Returns
        -------
        dolfinx.fem.Function
            The resulting temperature field solution.
        """
        conduction_coeff = self.apply_materials()
        self.apply_dirichlet_bc()
        self.create_bilinear_function(conduction_coeff)
        self.apply_robin()
        self.apply_neuman()
        return self.solve()

    def apply_dirichlet_bc(self) -> None:
        """Processes and stores Dirichlet boundary conditions (fixed temperature).

        Locates degrees of freedom on the specified surface tags and 
        populates the `self.bcs` list with dolfinx.fem.dirichletbc objects.
        """
        for tag, value in self.dirichlet_bcs:
            degrees_on_boundary = self.face_tags.find(tag)
            face_dimension = self.mesh.topology.dim - 1
            self.bcs.append(
                fem.dirichletbc(
                    default_scalar_type(value),
                    fem.locate_dofs_topological(
                        self.function_space, face_dimension, degrees_on_boundary
                    ),
                    self.function_space,
                )
            )

    def apply_materials(self) -> dolfinx.fem.Function:
        """Maps material properties from the library to the mesh subdomains.

        Creates a Discontinuous Galerkin (DG) function to represent the 
        thermal conductivity (k) across different volume tags.

        Returns
        -------
        dolfinx.fem.Function
            A spatially varying field containing the thermal conductivity values.
        """
        materials_library = ThermalMaterialLibrary()
        materials = [
            (tag, materials_library.get(material)) for tag, material in self.materials
        ]

        # constant value across each element
        disconnected_galerkin_space = fem.functionspace(self.mesh, ("DG", 0))
        global_conduction_coeff = fem.Function(disconnected_galerkin_space)
        global_conduction_coeff_values = global_conduction_coeff.x.array

        for tag, material in materials:
            global_conduction_coeff_values[self.volume_tag.find(tag)] = material.k
        return global_conduction_coeff

    def create_bilinear_function(self, conduction_coeff) -> None:
        """Defines the governing variational forms (LHS and RHS).

        Sets up the bilinear form `a` for heat conduction and begins 
        constructing the linear form `L`, including internal heat 
        generation terms if provided.

        Parameters
        ----------
        conduction_coeff : dolfinx.fem.Function
            The conductivity field defined over the mesh.
        """
        if self.internal_heat_generation is not None:
            self.a = conduction_coeff * ufl.dot(ufl.grad(self._u), ufl.grad(self._v)) * ufl.dx
            for tag, heat_generator in self.internal_heat_generation:
                heat_generator = fem.Constant(self.mesh, default_scalar_type(heat_generator))
                self.L += heat_generator * self._v * self.dx(tag)
        else:
            self.a = conduction_coeff * ufl.dot(ufl.grad(self._u), ufl.grad(self._v)) * ufl.dx
            self.L = 0

    def apply_robin(self) -> None:
        """Applies Robin (convection) boundary conditions to the variational forms.

        Modifies both the bilinear form `a` and the linear form `L` to 
        account for heat exchange with the ambient environment.
        """
        if self.rubin_bcs is None: return
        for tag, h in self.rubin_bcs:
            h = fem.Constant(self.mesh, default_scalar_type(h))
            self.a += h * self._u * self._v * self.ds(tag)
            self.L += h * self.T_amb * self._v * self.ds(tag)

    def apply_neuman(self) -> None:
        """Applies Neumann (heat flux) boundary conditions.

        Calculates the flux density per unit area based on the total 
        input flux and the integrated surface area of the tagged boundary.
        """
        if self.neuman_bcs is None: return
        for tag, q in self.neuman_bcs:
            area_form = fem.form(fem.Constant(self.mesh, 1.0) * self.ds(tag))
            area_local = fem.assemble_scalar(area_form)
            area_total = self.mesh.comm.allreduce(area_local, op=MPI.SUM)

            q_per_area = q / area_total
            q_bc = fem.Constant(self.mesh, default_scalar_type(q_per_area))

            self.L += q_bc * self._v * self.ds(tag)

    def solve(self) -> dolfinx.fem.Function:
        """Solves the assembled linear system using PETSc.

        Configures a direct solver (LU decomposition) to compute the 
        temperature distribution.

        Returns
        -------
        dolfinx.fem.Function
            The solved temperature field.
        """
        problem = LinearProblem(
            self.a,
            self.L,
            bcs=self.bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="fin_heat_",
        )
        u_t = problem.solve()
        print("Success!")
        return u_t


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from thermal_plotter import plot_3d_solution, plot_3d_solution_slice
    
    
    ##########################################################################
    # ### importnat note! ###
    # Before running this simulation, you have to convert the .msh file to
    # .xdmf and .h5 files, using the Msh2Xdmf class.
    ##########################################################################
    
    path = Path('/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/test/Assem1.msh')
    materials = ((51, 'Aluminium-6061'), (52, 'Copper'))
    dirichlet_bc = ((49, 120), )
    
    simulation = ThermalSimulator(path, materials, dirichlet_bc, convection_bcs=((50, 5), ), internal_heat_generation=((52, 1.0e5), (51, 3e5)), T_amb=25)
    u = simulation.run()
    # plot_3d(u, simulation.function_space)
    plotter1 = plot_3d_solution(u, simulation.function_space, to_probe=True)
    plotter1 = plot_3d_solution_slice(u, simulation.function_space, to_probe=True)
    plotter1.show()
