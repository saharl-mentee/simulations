import ufl
import dolfinx
from mpi4py import MPI
from typing import Tuple
from pathlib import Path
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type, fem
import ufl.finiteelement

from mesh_loader import Mesh3DLoader
from material_library import ThermalMaterialLibrary


class Thermal_simulator:
    def __init__(
        self,
        geometry_path: Path,
        materials: Tuple[Tuple[int, str], ...],
        temperature_bcs: Tuple[Tuple[int, float], ...],
        convection_bcs: Tuple[Tuple[int, float], ...] = None,
        neuman_bcs: Tuple[Tuple[int, float], ...] = None,
        elements_order: int = 2,
        T_amb: float = 25,
        internal_heat_generation: float = 0,
    ):
        self.geometry_path = geometry_path
        self.materials = materials
        self.elements_order = elements_order
        self.bcs = []
        self.dirichlet_bcs = temperature_bcs
        self.convection_bcs = convection_bcs
        self.neuman_bcs = neuman_bcs
        self.T_amb = T_amb

        mesh_loader = Mesh3DLoader(self.geometry_path)
        self.mesh, (self.volume_tag, self.face_tags) = mesh_loader.load_mesh()
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.face_tags)
        
        self._function_space = None
        self._u = ufl.TrialFunction(self.function_space)
        self._v = ufl.TestFunction(self.function_space)
        self.a = 0
        self.L = 0
        
        self.internal_heat_generation = fem.Constant(self.mesh, default_scalar_type(internal_heat_generation))

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """The FunctionSpace for the thermal problem.
        Created on demand (Lazy Initialization)."""
        if self._function_space is None:
            # Define a standard Lagrange element for Heat Transfer
            self._function_space = fem.functionspace(self.mesh, ("Lagrange", self.elements_order))
        return self._function_space
    
    # def u(self):
    #     if self._u is None:
    #         self._u = ufl.TrialFunction(self._function_space)
    #     return self._u
    
    # def v(self):
    #     if self._v is None:
    #         self._v = ufl.TestFunction(self._function_space)
    #     return self._v

    def run(self):
        conduction_coeff = self.apply_materials()
        self.create_dirichlet_bc()
        self.create_bilinear_function(conduction_coeff)
        self.apply_convection()
        self.apply_neuman()
        return self.solve()

    def create_dirichlet_bc(self) -> None:
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

    def apply_materials(self):
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

    def create_bilinear_function(self, conduction_coeff):
        self.a = conduction_coeff * ufl.dot(ufl.grad(self._u), ufl.grad(self._v)) * ufl.dx
        self.L = self.internal_heat_generation * self._v * ufl.dx
        self.apply_convection()
        self.apply_neuman()

    def apply_convection(self) -> None:
        if self.convection_bcs is None: return
        for tag, h in self.convection_bcs:
            h = fem.Constant(self.mesh, default_scalar_type(h))
            self.a += h * self._u * self._v * self.ds(tag)
            self.L += h * self.T_amb * self._v * self.ds(tag)

    def apply_neuman(self):
        if self.neuman_bcs is None: return
        for tag, q in self.neuman_bcs:
            area_form = fem.form(fem.Constant(self.mesh, 1.0) * self.ds(tag))
            area_local = fem.assemble_scalar(area_form)
            area_total = self.mesh.comm.allreduce(area_local, op=MPI.SUM)

            q_per_area = q / area_total
            q_bc = fem.Constant(self.mesh, default_scalar_type(q_per_area))

            self.L += q_bc * self._v * self.ds(tag)

    def solve(self):
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
    path = Path('/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/solid_test_2/standard_fin')
    materials = ((742, 'Aluminium-6061'), (743, 'Copper'))
    dirichlet_bc = ((746, 120), )
    
    simulation = Thermal_simulator(path, materials, dirichlet_bc)
    simulation.run()
