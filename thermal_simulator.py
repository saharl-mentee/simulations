from typing import Tuple

import dolfinx
from pathlib import Path
from dolfinx import default_scalar_type, fem
from mesh_loader import Mesh3DLoader
from ufl import element
from material_library import ThermalMaterialLibrary


class Thermal_simulator:
    def __init__(
        self,
        geometry_path: Path,
        dirichlet_bcs: Tuple[(int, float), ...],
        elements_order: int = 2,
    ):
        self.geometry_path = geometry_path
        self.elements_order = elements_order
        self.bcs = []
        self.dirichlet_bcs = dirichlet_bcs

        mesh_loader = Mesh3DLoader(self.geometry_path)
        self.mesh, self.volume_tag, self.face_tags = mesh_loader.load_mesh()
        self._function_space = fem.functionspace(
            self.mesh, ("Lagrange", self.elements_order)
        )

    @property
    def V(self) -> dolfinx.fem.FunctionSpace:
        """The FunctionSpace for the thermal problem.
        Created on demand (Lazy Initialization)."""
        if self._function_space is None:
            # Define a standard Lagrange element for Heat Transfer
            el = element(
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
                        self.V, facet_dimension, degrees_on_boundary
                    ),
                    self.V,
                )
            )
    
    def create_neuman_bc(self) -> None:
        for tag, value in self.dirichlet_bcs:
            degrees_on_boundary = self.facet_tags.find(tag)
            facet_dimension = self.mesh.topology.dim - 1
            self.bcs.append(
                fem.dirichletbc(
                    default_scalar_type(value),
                    fem.locate_dofs_topological(
                        self.V, facet_dimension, degrees_on_boundary
                    ),
                    self.V,
                )
            )


if __name__ == "__main__":
    simulation = Thermal_simulator()
