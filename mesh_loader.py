import glob
import dolfinx
from typing import List, Optional, Tuple
from mpi4py import MPI
from pathlib import Path
from dolfinx.io import XDMFFile
import numpy as np

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
        mesh_files = glob.glob(f'{self.path}_*.xdmf')
        file_type = [str(file).split('_')[-1].split('.')[0] for file in mesh_files]
        if "volume" in file_type: 
            # primary_path, is_3d = self.path.with_name(f"{self.path.name}_volume.xdmf"), True
            primary_path= self.path.with_name(f"{self.path.name}_volume.xdmf")
            file_type.remove("volume")
        elif "surface" in file_type: 
            # primary_path, is_3d = self.path.with_name(f"{self.path.name}_surface.xdmf"), False
            primary_path = self.path.with_name(f"{self.path.name}_surface.xdmf")
            file_type.remove("surface")
        else:
            raise FileNotFoundError(
                f"No valid XDMF files found for volume or surface mesh at {self.path}."
            )
        
        mesh, cell_tags = self.load_primary_mesh(primary_path)
        self.ensure_connectivity(mesh)
        tags = self.populate_tags(mesh, cell_tags, file_type)
        
        if self.in_mm:
            mesh.geometry.x[:, :] *= 0.001
        return mesh, tags
    
    def load_primary_mesh(self, path) -> Tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags]:
        """Reads the tetrahedral volume mesh and associated cell markers.

        Returns
        -------
        mesh : dolfinx.mesh.Mesh
            The distributed volume mesh.
        cell_tags : dolfinx.mesh.MeshTags
            Markers for subdomains (e.g., different materials).
        """

        with XDMFFile(self.comm, path, "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
            try:
                cell_tags = xdmf.read_meshtags(mesh, name="Grid")
            except RuntimeError:
                # Fallback if the main domain has no material groups
                num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
                cell_tags = dolfinx.mesh.meshtags(
                    mesh, mesh.topology.dim, np.arange(num_cells, dtype=np.int32), np.zeros(num_cells, dtype=np.int32)
                )
        return mesh, cell_tags
    
    def populate_tags(self, mesh: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, file_type: List[str]) -> Tuple[dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
        # 4. Initialize our clean tags database
        tags_db = {
            "cell": cell_tags,
            "surface": None,
            "edge": None,
            "point": None
        }
        
        # 5. Populate tags dimension by dimension based on what files exist
        for file in file_type:
            tags_db[file] = self._read_tags_safely(mesh, self.path.with_name(f"{self.path.name}_{file}.xdmf"))
        return tags_db
            
    def _read_tags_safely(self, mesh: dolfinx.mesh.Mesh, target_path: Path) -> Optional[dolfinx.mesh.MeshTags]:
        """Helper to read tags from a file without throwing exceptions if empty."""
        if not target_path.exists():
            return None
        with XDMFFile(self.comm, target_path, "r") as xdmf:
            try:
                tags = xdmf.read_meshtags(mesh, name="Grid")
                print(f"[Linked] Successfully mapped: {target_path.name}")
                return tags
            except RuntimeError:
                print(f"[Warning] Found file {target_path.name} but extraction failed.")
                return None

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
        # 3. Dynamic Connectivity Generation for ALL intermediate dimensions
        bulk_dim = mesh.topology.dim
        for d in range(1, bulk_dim):
            mesh.topology.create_entities(d)
            mesh.topology.create_connectivity(d, bulk_dim)
        mesh.topology.create_connectivity(0, bulk_dim) # Always connect vertices (0) to cells

if __name__ == '__main__':
    geometry_path = Path('/mnt/c/Users/saharl/Documents/simulations_api/simulations/test_dxf_exporter/standard_fin2.msh')
    mesh_loader = Mesh3DLoader(geometry_path.with_name(geometry_path.stem))
    output = mesh_loader.load_mesh()
    print('success')
    