import dolfinx
from typing import Tuple
from mpi4py import MPI
from pathlib import Path
from dolfinx.io import XDMFFile


class Mesh3DLoader:
    """A utility class to handle loading, scaling, and topology initialization
    for FEniCSx meshes from XDMF files.
    """

    def __init__(self, path: Path, in_mm: bool = True):
        """
        Parameters
        ----------
        path : Path
            The base path/prefix of the XDMF files without suffix(e.g., 'data/mesh')
        in_mm : bool, optional
            If True, scales the mesh coordinates by 1e-3 to convert
            millimeters to meters. Defaults to True.
        """
        self.comm = MPI.COMM_WORLD
        self.path = path
        self.in_mm = in_mm

    def load_mesh(
        self,
    ) -> Tuple[dolfinx.mesh.Mesh, Tuple[dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]]:
        """High-level method to perform the full loading sequence: volume,
        connectivity, and facets.

        Returns
        -------
        mesh : dolfinx.mesh.Mesh
            The distributed volume mesh.
        cell_tags : dolfinx.mesh.MeshTags
            Markers for subdomains (e.g., different materials).
        face_tags : dolfinx.mesh.MeshTags
            Markers for faces for different BC (e.g., Dirichlet or Neumann).
        """
        mesh, cell_tags = self.load_volume_mesh()
        self.ensure_connectivity(mesh)
        face_tags = self.load_face_mesh(mesh)
        if self.in_mm:
            mesh.geometry.x[:, :] *= 0.001
        return mesh, (cell_tags, face_tags)

    def load_volume_mesh(self) -> Tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags]:
        """Reads the tetrahedral volume mesh and associated cell markers.

        Returns
        -------
        mesh : dolfinx.mesh.Mesh
            The distributed volume mesh.
        cell_tags : dolfinx.mesh.MeshTags
            Markers for subdomains (e.g., different materials).
        """
        path_vol = self.path.with_name(f"{self.path.name}_volume.xdmf")
        with XDMFFile(self.comm, path_vol, "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
            cell_tags = xdmf.read_meshtags(mesh, name="Grid")
        return mesh, cell_tags

    def load_face_mesh(self, mesh: dolfinx.mesh.Mesh) -> dolfinx.mesh.MeshTags:
        """Reads the triangular surface markers and applies coordinate scaling.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The distributed volume mesh.

        Returns
        -------
        face_tags : dolfinx.mesh.MeshTags
            Markers for faces for different BC (e.g., Dirichlet or Neumann).
        """
        path_surface = self.path.with_name(f"{self.path.name}_surface.xdmf")
        with XDMFFile(self.comm, path_surface, "r") as xdmf:
            facet_tags = xdmf.read_meshtags(mesh, name="Grid")
        return facet_tags

    def ensure_connectivity(self, mesh: dolfinx.mesh.Mesh) -> None:
        """Builds the internal mesh topology relationships required to map 
        lower-dimensional facets to the 3D volume.

        In FEniCSx, meshes are loaded as a collection of cells (3D) and 
        vertices by default. Intermediate entities like facets (2D faces) 
        must be explicitly created and linked to the cells to apply 
        boundary conditions or read surface markers.

        This method performs three steps:
        1. Identifies the facet dimension (e.g., 2 for a 3D mesh).
        2. Generates the facet entities in the mesh topology.
        3. Computes the '2 -> 3' connectivity, mapping each facet to 
           its parent cell(s).

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The distributed volume mesh.
        """
        facet_dim = mesh.topology.dim - 1
        mesh.topology.create_entities(facet_dim)
        mesh.topology.create_connectivity(facet_dim, mesh.topology.dim)
