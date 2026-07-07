import glob
import dolfinx
import meshio
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
        primary_path, file_type = self.get_primary_grid_path()
        mesh, cell_tags = self.load_primary_mesh(primary_path)
        self.ensure_connectivity(mesh)
        tags = self.populate_tags(mesh, cell_tags, file_type)
        
        if self.in_mm:
            mesh.geometry.x[:, :] *= 0.001
        return mesh, tags
    
    def get_primary_grid_path(self) -> Path:
        mesh_files = glob.glob(f'{self.path}_*.xdmf')
        file_type = [str(file).split('_')[-1].split('.')[0] for file in mesh_files]
        if "volume" in file_type: 
            primary_path= self.path.with_name(f"{self.path.name}_volume.xdmf")
            file_type.remove("volume")
        elif "surface" in file_type: 
            primary_path = self.path.with_name(f"{self.path.name}_surface.xdmf")
            file_type.remove("surface")
        else:
            raise FileNotFoundError(
                f"No valid XDMF files found for volume or surface mesh at {self.path}."
            )
        return primary_path, file_type
    
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
            if file == "point":
                tags_db[file] = self._read_points_via_meshio(mesh, self.path.with_name(f"{self.path.name}_{file}.xdmf"))
            else:
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
            
    def _read_points_via_meshio(self, mesh: dolfinx.mesh.Mesh, target_path: Path) -> Optional[dolfinx.mesh.MeshTags]:
        """Safely extracts 0D points using meshio and maps them geometrically 
        to the master mesh vertices to create a native DOLFINx MeshTags object.
        """
        if not target_path.exists():
            print(f"⚠️ [Loader Warning] Point file does not exist at: {target_path}")
            return None
        try:
            import meshio
            pt_mesh = meshio.read(target_path)
            
            if "vertex" in pt_mesh.cells_dict:
                # Extract the tracking point coordinates
                vertex_indices = pt_mesh.cells_dict["vertex"].flatten()
                actual_coords = pt_mesh.points[vertex_indices]
                
                # 🛠️ FIXED: Safely extract tags from the "Grid" attribute without triggering KeyErrors
                grid_data = pt_mesh.cell_data_dict.get("Grid", {})
                if isinstance(grid_data, dict):
                    # Look for the explicit "vertex" key, or fall back to the first available array
                    tags = grid_data.get("vertex", list(grid_data.values())[0])
                else:
                    tags = grid_data[0]
                
                matched_vertices = []
                matched_tags = []
                
                # 3. Reference the actual master 3D mesh node coordinate grid
                # (Both matrices are currently in millimeters at this stage)
                mesh_coords = mesh.geometry.x
                
                for p_coord, tag in zip(actual_coords, tags):
                    distances = np.linalg.norm(mesh_coords - p_coord, axis=1)
                    closest_node_idx = np.argmin(distances)
                    min_distance = distances[closest_node_idx]
                    
                    # A tolerance of 0.1mm easily catches the misaligned transfinite node
                    if min_distance < 1e-1:
                        matched_vertices.append(closest_node_idx)
                        matched_tags.append(tag)
                        print(f"🎯 [Proximity Match] Point Tag {tag} successfully mapped to Mesh Node ID {closest_node_idx} (Offset: {min_distance:.8f} mm)")
                    else:
                        print(f"❌ [Geometric Error] Point Tag {tag} at {p_coord.tolist()} has no nearby mesh vertices.")
                        print(f"   Closest mesh node found is {min_distance:.2f} mm away.")
                
                if matched_vertices:
                    # 4. Construct and return the native DOLFINx 0D MeshTags object
                    return dolfinx.mesh.meshtags(
                        mesh, 0, 
                        np.array(matched_vertices, dtype=np.int32), 
                        np.array(matched_tags, dtype=np.int32)
                    )
                else:
                    print("⚠️ [Loader Warning] Point file processed, but zero nodes fell within tolerance parameters.")
            else:
                print("⚠️ [Loader Warning] No tracking vertices found in the point mesh layout structure.")
        except Exception as e:
            print(f"❌ [Loader Error] Proximity mapping loop encountered an error: {e}")
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
        
        # 🛠️ FIXED: Force DOLFINx to create topological entity indexes for 0D vertices
        mesh.topology.create_entities(0)
        
        for d in range(1, bulk_dim):
            mesh.topology.create_entities(d)
            mesh.topology.create_connectivity(d, bulk_dim)
        mesh.topology.create_connectivity(0, bulk_dim) # Always connect vertices (0) to cells
        
def verify_exported_points(xdmf_path: Path):
    import meshio
    """Reads an exported XDMF point file and prints a structural sanity check."""
    if not xdmf_path.exists():
        print(f"❌ Verification Failed: File does not exist at {xdmf_path}")
        return

    # Read the exported file back in using meshio
    mesh = meshio.read(xdmf_path)
    
    print("\n==============================")
    print(f"📊 VERIFYING: {xdmf_path.name}")
    print("==============================")
    
    # 1. Check if vertices exist
    num_points = len(mesh.points)
    print(f"✅ Total Points Exported: {num_points}")
    
    # 2. Print the exact spatial coordinates
    print("\n📍 Exported Coordinates (X, Y, Z):")
    for i, coords in enumerate(mesh.points):
        # print(f"  Point [{i}]: {coords}")
        pass
        
    # 3. FIXED: Look for "Grid" instead of "gmsh:physical"
    if "Grid" in mesh.cell_data_dict:
        tags = mesh.cell_data_dict["Grid"]
        
        # Unpack the array if meshio returned it inside a block list
        if isinstance(tags, list) and len(tags) > 0:
            tags = tags[0]
            
        print("\n🏷️  Associated Physical Tags:")
        for i, tag in enumerate(tags):
            print(f"  Point [{i}] Tag ID: {tag}")
    else:
        print("❌ Error: No physical tags found under 'Grid' in this XDMF file.")
        
    print("==============================\n")

if __name__ == '__main__':
    geometry_path = Path('/mnt/c/Users/saharl/Documents/simulations_api/simulations/test_dxf_exporter/standard_fin3.msh')
    mesh_loader = Mesh3DLoader(geometry_path.with_name(geometry_path.stem))
    output = mesh_loader.load_mesh()
    print(output)
    verify_exported_points(geometry_path.with_name(f"{geometry_path.stem}_point.xdmf"))
    print('success')
    