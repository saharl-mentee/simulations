import ufl
import dolfinx
from mpi4py import MPI
from typing import Tuple
from pathlib import Path
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type, fem

from mesh_loader import Mesh3DLoader
from material_library import ElasticMaterialLibrary
from thermal_fins_motor import plot_3d


class StaticStructuralSimulator:
    """A Finite Element simulator for linear static structural analysis using dolfinx.
    
    This class handles the setup and solution of the linear elasticity problem:
    div(sigma) + f = 0
    
    where sigma is the stress tensor and f is the body force vector.
    """
    
    def __init__(
        self,
        geometry_path: Path,
        materials: Tuple[Tuple[int, str], ...],
        displacement_bcs: Tuple[Tuple[int, Tuple[float, float, float]], ...],
        traction_bcs: Tuple[Tuple[int, Tuple[float, float, float]], ...] = None,
        elements_order: int = 2,
        T_amb: float = 25,
        body_forces: Tuple[Tuple[int, Tuple[float, float, float]], ...] = None,
    ):
        """Initializes the structural simulator and loads the mesh.

        Parameters
        ----------
        geometry_path : Path
            Path to the .msh file.
        materials : Tuple[Tuple[int, str], ...]
            Collection of tuples: (volume_tag, material_name).
        displacement_bcs : Tuple[Tuple[int, Tuple[float, float, float]], ...]
            Tuples of (face_tag, (u_x, u_y, u_z)) for Dirichlet conditions (Fixed/Prescribed displacement).
        traction_bcs : Tuple[Tuple[int, Tuple[float, float, float]], ...], optional
            Tuples of (face_tag, (F_x, F_y, F_z)) representing total force applied on that surface.
        elements_order : int, optional
            Polynomial order for Lagrange elements, by default 2.
        body_forces : Tuple[Tuple[int, Tuple[float, float, float]], ...], optional
            Tuples of (volume_tag, (b_x, b_y, b_z)) for gravity or centrifugal loads.
        """
        self.geometry_path = geometry_path
        self.materials = materials
        self.elements_order = elements_order
        self.bcs = []
        self.dirichlet_bcs = displacement_bcs
        self.neuman_bcs = traction_bcs
        self.T_amb = T_amb
        self.internal_forces = body_forces
        self.lmbda = None
        self.mu = None

        self.load_mesh()
        # 1. DYNAMIC FACET MEASURE SETUP
        # Automatically detects boundary facets: dimension 2 (surfaces) in 3D meshes, 
        # or dimension 1 (edges) in 2D meshes.
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
        """The Function Space of the elements for the thermal problem"""
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
    
    def run(self) -> dolfinx.fem.Function:
        """Executes the complete simulation workflow.

        This orchestrates material assignment, boundary condition application,
        variational form assembly, and the final solver execution.

        Returns
        -------
        dolfinx.fem.Function
            The resulting temperature field solution.
        """
        self.lmbda, self.mu = self.apply_materials()
        self.apply_dirichlet_bc()
        self.create_bilinear_function()
        self.apply_neuman()
        print("Solving the linear system...")
        return self.solve()

    def apply_dirichlet_bc(self) -> None:
        """Processes and stores Dirichlet boundary conditions (fixed temperature).

        Locates degrees of freedom on the specified surface tags and 
        populates the `self.bcs` list with dolfinx.fem.dirichletbc objects.
        """
        for tag, value in self.dirichlet_bcs:
            found = False
            
            # Scan through all topological categories populated by the loader
            for entity_type, tags_obj in self.tags.items():
                if tags_obj is None:
                    continue
                
                # Check if this specific tag exists within the current entity category
                entities = tags_obj.find(tag)
                if len(entities) > 0:
                    # Read the exact topological dimension (3=volume, 2=surface, 1=edge, 0=point)
                    ent_dim = tags_obj.dim
                    
                    # FIXED: Everything is now inside the block where the entities actually exist
                    bc_vector = fem.Constant(self.mesh, default_scalar_type(value))
                    dofs = fem.locate_dofs_topological(self.function_space, ent_dim, entities)
                    
                    self.bcs.append(fem.dirichletbc(bc_vector, dofs, self.function_space))
                    found = True
                    print(f"[Linked Dirichlet BC] Applied tag {tag} to '{entity_type}' region (Dimension: {ent_dim})")
                    
                    # FIXED: Break out of the inner loop immediately so later dimensions don't overwrite this data
                    break
            
            if not found:
                raise KeyError(
                    f"Dirichlet tag {tag} was not found in any loaded mesh tags "
                    f"(cells, surfaces, edges, or points). Verify your Gmsh physical group IDs."
                )

    def apply_materials(self) -> Tuple[dolfinx.fem.Function, dolfinx.fem.Function]:
        """Computes Lamé constants (lambda and mu) from E and nu across volumes.
        Note: Implement/import a ElasticMaterialLibrary yielding .E and .nu attributes
        
        Creates a Discontinuous Galerkin (DG) function to represent the 
        materials properties (lambda and mu) across different volume tags.

        Returns
        -------
        dolfinx.fem.Function
            A spatially varying field containing the Lamé constants.
        """
        materials_library = ElasticMaterialLibrary()
        materials = [
            (tag, materials_library.get(material)) for tag, material in self.materials
        ]

        # constant value across each element
        disconnected_galerkin_space = fem.functionspace(self.mesh, ("DG", 0))
        global_lmbda_field = fem.Function(disconnected_galerkin_space)
        global_mu_field = fem.Function(disconnected_galerkin_space)
        global_lmbda_field_values = global_lmbda_field.x.array
        global_mu_field_values = global_mu_field.x.array

        for tag, material in materials:

            element_indices = self.tags['cell'].find(tag)
            global_lmbda_field_values[element_indices] = material.lmbda
            global_mu_field_values[element_indices] = material.mu
        return global_lmbda_field, global_mu_field

    def create_bilinear_function(self) -> None:
        """Defines the internal virtual work (LHS) and body forces (RHS).

        Sets up the bilinear form `a` for heat conduction and begins 
        constructing the linear form `L`, including internal heat 
        generation terms if provided.

        Parameters
        ----------
        conduction_coeff : dolfinx.fem.Function
            The conductivity field defined over the mesh.
        """
        self.a = ufl.inner(self.sigma(self._u), self.epsilon(self._v)) * ufl.dx
        g_dim = self.mesh.geometry.dim
        zero_force = fem.Constant(self.mesh, default_scalar_type((0.0,) * g_dim))
        self.L = ufl.dot(zero_force, self._v) * ufl.dx
        
        if self.internal_forces is not None:
            for tag, f_vector in self.internal_forces:
                f_bc = fem.Constant(self.mesh, default_scalar_type(f_vector))
                self.L += ufl.dot(f_bc, self._v) * self.dx(tag)

    def apply_neuman(self) -> None:
        """Applies Neumann (force/traction) boundary conditions.

        Calculates the flux density per unit area based on the total 
        input flux and the integrated surface area of the tagged boundary.
        """
        if self.neuman_bcs is None: return
        for tag, q in self.neuman_bcs:
            area_form = fem.form(fem.Constant(self.mesh, 1.0) * self.ds(tag))
            area_local = fem.assemble_scalar(area_form)
            area_total = self.mesh.comm.allreduce(area_local, op=MPI.SUM)

            q_per_area = [q_i / area_total for q_i in q]  # Convert total flux to flux density
            q_bc = fem.Constant(self.mesh, default_scalar_type(q_per_area))

            self.L += ufl.dot(q_bc, self._v) * self.ds(tag)

    def solve(self) -> dolfinx.fem.Function:
        """Solves the assembled linear system using PETSc.

        Configures a direct solver (LU decomposition) to compute the 
        temperature distribution.

        Returns
        -------
        dolfinx.fem.Function
            The solved temperature field.
        """
        # Optimized PETSc settings for 3D Elasticity
        structural_options = {
            "ksp_type": "cg",          # Conjugate Gradient iterative solver
            "pc_type": "gamg",         # Algebraic Multigrid preconditioner
            "ksp_rtol": 1e-6,          # Relative tolerance
            "ksp_atol": 1e-10,         # Absolute tolerance
            "mat_block_size": self.mesh.geometry.dim,  # Tells GAMG it's a 3D vector problem
        }
        
        problem = LinearProblem(
            self.a,
            self.L,
            bcs=self.bcs,
            petsc_options=structural_options,
            petsc_options_prefix="struct_sim_",
        )
        u_t = problem.solve()
        print("Success!")
        return u_t


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from static_structural_plotter import StressVolumeViewer, DeflectionViewer, VonMisesViewer, compute_von_mises_from_simulator
    
    
    ##########################################################################
    # ### importnat note! ###
    # Before running this simulation, you have to convert the .msh file to
    # .xdmf and .h5 files, using the Msh2Xdmf class.
    ##########################################################################
    
    path = Path('/mnt/c/Users/saharl/Documents/simulations_api/simulations/test_dxf_exporter/standard_fin3.msh')
    materials = ((15, 'Aluminum-6061'), )
    dirichlet_bc = ((13, (0, 0, 0)), (18, (0, 1e-3, 0)))
    # dirichlet_bc = ((13, (0, 0, 0)), (16, (0, 1e-3, 0)))
    
    simulation = StaticStructuralSimulator(path, materials, dirichlet_bc, elements_order=1)
    u = simulation.run()
    # plot_3d(u, simulation.function_space)
    # viewer = StressVolumeViewer(u, simulation.function_space, to_probe=True)
    deflection_viewer = DeflectionViewer(u, simulation.function_space, scaling_factor=10)
    deflection_viewer.plotter.show()
    stress_field = compute_von_mises_from_simulator(u, simulation.lmbda, simulation.mu)
    von_mises_viewer = VonMisesViewer(u, stress_field, scaling_factor=1)
    von_mises_viewer.show()
    # viewer = SliceViewer(u, simulation.function_space, to_probe=True)    
    # viewer = FluxViewer(u, simulation.function_space, simulation.conduction_coeff, to_probe=True)
    # viewer.plotter.show()
