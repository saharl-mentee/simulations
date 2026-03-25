import ufl
import dolfinx
from mpi4py import MPI
from typing import Tuple
from pathlib import Path
from dolfinx.fem.petsc import LinearProblem
from dolfinx import default_scalar_type, fem

from mesh_loader import Mesh3DLoader
from material_library import ThermalMaterialLibrary


class Thermal_simulator:
    def __init__(
        self,
        geometry_path: Path,
        materials: Tuple[(int, float), ...],
        temperature_bcs: Tuple[(int, float), ...],
        convection_bcs: Tuple[(int, float), ...] = None,
        neuman_bcs: Tuple[(int, float), ...] = None,
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
        self.internal_heat_generation = internal_heat_generation
        
        self.a = 0
        self.L = 0

        mesh_loader = Mesh3DLoader(self.geometry_path)
        self.mesh, self.volume_tag, self.face_tags = mesh_loader.load_mesh()
        self._function_space = fem.functionspace(
            self.mesh, ("Lagrange", self.elements_order)
        )

    @property
    def v(self) -> dolfinx.fem.FunctionSpace:
        """The FunctionSpace for the thermal problem.
        Created on demand (Lazy Initialization)."""
        if self._function_space is None:
            # Define a standard Lagrange element for Heat Transfer
            el = ufl.element(
                "Lagrange", self.mesh.topology.cell_name(), self.elements_order
            )
            self._function_space = dolfinx.fem.functionspace(self.mesh, el)
        return self._function_space

    def create_dirichlet_bc(self) -> None:
        for tag, value in self.dirichlet_bcs:
            degrees_on_boundary = self.facet_tags.find(tag)
            facet_dimension = self.mesh.topology.dim - 1
            self.bcs.append(
                fem.dirichletbc(
                    default_scalar_type(value),
                    fem.locate_dofs_topological(
                        self.v, facet_dimension, degrees_on_boundary
                    ),
                    self.v,
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
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.face_tags)

        self.a = conduction_coeff * ufl.dot(ufl.grad(self.v), ufl.grad(self.v)) * ufl.dx
        self.L = self.internal_heat_generation * self.v * ufl.dx
        self.apply_convection(ds)
        self.apply_neuman(ds)

    def apply_convection(self,ds) -> None:
        for tag, h in self.convection_bcs:
            self.a += h * self.v * self.v * ds(tag)
            self.L += h * self.T_amb * self.v * ds(tag)

    def apply_neuman(self, ds):
        for tag, q in self.neuman_bcs:
            area_form = fem.form(fem.Constant(self.mesh, 1.0) * ds(tag))
            area_local = fem.assemble_scalar(area_form)
            area_total = self.mesh.comm.allreduce(area_local, op=MPI.SUM)
            
            q_per_area = q / area_total
            q_bc = fem.Constant(self.mesh, default_scalar_type(q_per_area))
            
            self.L += q_bc * self.v * ds(tag)
    
    def solve(self):
        problem = LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="fin_heat_",
        )  # <--- This is the missing piece
        u_t = problem.solve()
        print("Success!")


if __name__ == "__main__":
    simulation = Thermal_simulator()
