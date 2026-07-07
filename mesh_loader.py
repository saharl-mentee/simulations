import glob
from basix import topology
from basix import topology
import dolfinx
import meshio
from typing import List, Optional, Tuple, Dict
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

    def load_mesh(self,) -> Tuple[dolfinx.mesh.Mesh, Dict[str, Optional[dolfinx.mesh.MeshTags]]]:
        """High-level method to perform the full loading sequence: volume,
        connectivity, and facets.

        Returns
        -------
        mesh : dolfinx.mesh.Mesh
            The distributed volume mesh.
        tags : Dict[str, Optional[dolfinx.mesh.MeshTags]]
            A dictionary mapping geometric components ('cell', 'surface', 'edge', 'point')
            to their respective MeshTags objects.
        """
        primary_path, file_type = self.get_primary_grid_path()
        mesh, cell_tags = self.load_primary_mesh(primary_path)
        self.ensure_connectivity(mesh)
        tags = self.populate_tags(mesh, cell_tags, file_type)
        
        if self.in_mm:
            self._scale_mesh_coordinates(mesh, scale_factor=0.001)
        return mesh, tags
    
    def get_primary_grid_path(self) -> Tuple[Path, List[str]]:
        """
        Locate the primary XDMF mesh file and identify remaining mesh components.

        This method scans for XDMF files matching the object's base path. It prioritizes 
        the volume mesh if present, falls back to the surface mesh if not, and identifies 
        any remaining mesh types (e.g., boundaries or physical regions).

        Returns
        -------
        tuple[Path, list[str]]
            A tuple containing:
            - Path: The filepath of the prioritized primary mesh (volume or surface).
            - list[str]: The suffixes of the other discovered mesh files after 
            removing the primary type.

        Raises
        ------
        FileNotFoundError
            If neither a volume nor a surface mesh file is discovered matching the 
            expected naming convention.
        """
        matching_files = glob.glob(f'{self.path}_*.xdmf')
        discovered_types = [Path(file).stem.split('_')[-1] for file in matching_files]
        if "volume" in discovered_types: 
            primary_path= self.path.with_name(f"{self.path.name}_volume.xdmf")
            discovered_types.remove("volume")
        elif "surface" in discovered_types: 
            primary_path = self.path.with_name(f"{self.path.name}_surface.xdmf")
            discovered_types.remove("surface")
        else:
            raise FileNotFoundError(
                f"No valid XDMF files found for volume or surface mesh at {self.path}."
            )
        return primary_path, discovered_types
    
    def load_primary_mesh(self, mesh_path: str | Path) -> Tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags]:
        """
        Reads the mesh and associated cell markers from an XDMF file.

        Parameters
        ----------
        mesh_path : str or Path
            The filepath to the XDMF file containing the mesh data.

        Returns
        -------
        mesh : dolfinx.mesh.Mesh
            The distributed volume mesh.
        cell_tags : dolfinx.mesh.MeshTags
            Markers for subdomains (e.g., different materials). If no markers are 
            found in the file, a fallback mesh tag initialized to zero is returned.
        """
        with XDMFFile(self.comm, mesh_path, "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
            try:
                cell_tags = xdmf.read_meshtags(mesh, name="Grid")
            except RuntimeError:
                # Fallback if the main domain has no material groups
                topology = mesh.topology
                cell_dim = topology.dim
                num_cells = topology.index_map(cell_dim).size_global
                
                cell_indices = np.arange(num_cells, dtype=np.int32)
                default_marker_values = np.zeros(num_cells, dtype=np.int32)
                
                cell_tags = dolfinx.mesh.meshtags(mesh, cell_dim, cell_indices, default_marker_values)
        return mesh, cell_tags
    
    def populate_tags(self, mesh: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, mesh_components: List[str]) -> Dict[str, Optional[dolfinx.mesh.MeshTags]]:
        """
        Populate a database of mesh tags for various geometric dimensions.

        This method reads supplementary mesh tag files (such as surfaces, edges, or points) 
        associated with the primary mesh base path and maps them to their respective 
        geometric features.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The distributed primary mesh topology.
        cell_tags : dolfinx.mesh.MeshTags
            The pre-loaded mesh tags corresponding to the cells (volume elements).
        mesh_components : List[str]
            A list of geometric component suffixes (e.g., ['surface', 'edge', 'point']) 
            indicating which supplementary XDMF files exist and should be processed.

        Returns
        -------
        Dict[str, Optional[dolfinx.mesh.MeshTags]]
            A dictionary mapping geometric feature names ('cell', 'surface', 'edge', 'point') 
            to their corresponding MeshTags objects, or None if the component was not loaded.
        """
        # Initialize our clean tags database
        tags_db = {
            "cell": cell_tags,
            "surface": None,
            "edge": None,
            "point": None
        }
        
        # Populate tags dimension by dimension based on what files exist
        for component in mesh_components:
            component_path = self.path.with_name(f"{self.path.name}_{component}.xdmf")
            if component == "point":
                tags_db[component] = self._read_points_via_meshio(mesh, component_path)
            else:
                tags_db[component] = self._read_tags_safely(mesh, component_path)
        return tags_db
            
    def _read_tags_safely(self, mesh: dolfinx.mesh.Mesh, target_path: Path) -> Optional[dolfinx.mesh.MeshTags]:
        """
        Safely read mesh tags from an XDMF file if it exists, handling extraction errors.

        This helper method checks for the file's existence before attempting to parse it.
        If the file is missing, it returns None. If the file exists but the mesh tags 
        cannot be read (e.g., due to an incompatible format or missing "Grid" dataset), 
        the error is caught, logged as a warning, and None is returned.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The distributed mesh topology to which the tags will be mapped.
        target_path : Path
            The filepath of the target XDMF file containing the geometric tags.

        Returns
        -------
        Optional[dolfinx.mesh.MeshTags]
            The extracted mesh tags object if successful, or None if the file is 
            missing or extraction fails.
        """
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
        """
        Read 0D point markers via meshio and map them onto the native DOLFINx mesh nodes.

        This method parses an external mesh file containing point/vertex data, extracts 
        its coordinates and tags, and uses a proximity matching algorithm to link those 
        points to the master DOLFINx mesh geometry indices.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The master distributed coordinate mesh topology.
        target_path : Path
            The filepath to the meshio-compatible file containing the point markers.

        Returns
        -------
        Optional[dolfinx.mesh.MeshTags]
            A 0-dimensional DOLFINx MeshTags object matching vertices to tags if 
            successful; returns None if the file is missing, invalid, or no nodes 
            fall within proximity tolerances.
        """
        if not target_path.exists():
            print(f"⚠️ [Loader Warning] Point file does not exist at: {target_path}")
            return None
        try:
            point_mesh = meshio.read(target_path)
            
            if "vertex" not in point_mesh.cells_dict:
                print("⚠️ [Loader Warning] No tracking vertices found in the point mesh layout structure.")
                return None
                
            point_tags, point_coords = self._extract_vertex_tags_and_coords(point_mesh)
            
            matched_node_indices, matched_tag_values = self._match_physical_group_to_mesh_nodes(point_tags, point_coords, mesh)
            
            if not matched_node_indices:
                print("⚠️ [Loader Warning] Point file processed, but zero nodes fell within tolerance parameters.")
                return None
            return dolfinx.mesh.meshtags(
                mesh, 
                0,  # Dimension 0 denotes point/vertex entities
                np.array(matched_node_indices, dtype=np.int32), 
                np.array(matched_tag_values, dtype=np.int32)
            )
            
        except Exception as e:
            print(f"❌ [Loader Error] Proximity mapping loop encountered an error: {e}")
            return None
    
    def _extract_vertex_tags_and_coords(self, point_mesh: meshio.Mesh) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract vertex tags and their corresponding spatial coordinates from a meshio Mesh.

        This helper parses meshio's underlying cell dictionaries to locate 0D vertex elements,
        isolates their exact spatial coordinate vectors, and safely extracts any associated 
        physical tags or entity markers.

        Parameters
        ----------
        point_mesh : meshio.Mesh
            The input meshio data structure containing vertex cells and cell data attributes.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - vertex_tags (np.ndarray): Numerical tags/markers assigned to the vertices.
            - vertex_coords (np.ndarray): Spatial coordinates (X, Y, Z) for each tagged vertex.
        """
        # Extract the tracking point coordinates
        vertex_indices = point_mesh.cells_dict["vertex"].flatten()
        vertex_coords = point_mesh.points[vertex_indices]
        
        # 🛠️ FIXED: Safely extract tags from the "Grid" attribute without triggering KeyErrors
        grid_data = point_mesh.cell_data_dict.get("Grid", {})
        vertex_tags = np.array([], dtype=np.int32)
        if isinstance(grid_data, dict) and grid_data:
            # Check for explicit "vertex" markers; fallback cleanly to the first available dictionary array
            vertex_tags = grid_data.get("vertex")
            if vertex_tags is None:
                vertex_tags = next(iter(grid_data.values()))
                
        elif grid_data is not None and len(grid_data) > 0:
            vertex_tags = grid_data[0]
        
        return np.asarray(vertex_tags), vertex_coords

    def _match_physical_group_to_mesh_nodes(self, point_tags:np.ndarray, point_coords:np.ndarray, mesh: dolfinx.mesh.Mesh, match_tolerance:float = 1e-1) -> Tuple[List[int], List[int]]:
        """
        Map a point cloud of physical tags onto the closest vertex indices of a master mesh.

        This helper calculates the Euclidean distances between a collection of external 
        tracking points and the nodal coordinate array of a distributed DOLFINx mesh. Points 
        falling within a specified proximity threshold are mapped to their closest node ID.

        Parameters
        ----------
        point_tags : np.ndarray
            An array of numerical tags or identifiers assigned to each coordinate entry.
        point_coords : np.ndarray
            An Nx3 array of spatial coordinates (X, Y, Z) representing the tracking point cloud.
        mesh : dolfinx.mesh.Mesh
            The master distributed mesh geometry containing the target coordinate grid.

        Returns
        -------
        Tuple[List[int], List[int]]
            A tuple containing two lists:
            - matched_node_indices: The coordinate indices of the matched master mesh nodes.
            - matched_tag_values: The corresponding physical tags linked to those nodes.
        """
        # 3. Reference the actual master 3D mesh node coordinate grid
        # (Both matrices are currently in millimeters at this stage)
        master_coords = mesh.geometry.x
        matched_node_indices: List[int] = []
        matched_tag_values: List[int] = []

        for point_coord, tag in zip(point_coords, point_tags):
            distances = np.linalg.norm(master_coords - point_coord, axis=1)
            closest_node_idx = np.argmin(distances)
            min_distance = distances[closest_node_idx]
            
            # A tolerance of 0.1mm easily catches the misaligned transfinite node
            if min_distance < match_tolerance:
                matched_node_indices.append(closest_node_idx)
                matched_tag_values.append(int(tag))
                print(
                f"🎯 [Proximity Match] Point Tag {tag} successfully mapped to Mesh Node ID {closest_node_idx} (Offset: {min_distance:.8f} mm)"
            )
            else:
                print(f"❌ [Geometric Error] Point Tag {tag} at {point_coord.tolist()} has no nearby mesh vertices.")
                print(f"   Closest mesh node found is {min_distance:.2f} mm away.")
        return matched_node_indices, matched_tag_values

    def ensure_connectivity(self, mesh: dolfinx.mesh.Mesh) -> None:
        """
        Build the internal mesh topology and connectivity for all lower-dimensional entities.

        By default, DOLFINx only instantiates the highest-dimensional cell entities 
        (e.g., 3D tetrahedra) and 0D vertices. Intermediate geometric entities like 
        facets (2D faces) and edges (1D segments) must be explicitly generated and 
        linked to the primary cells before you can apply boundary conditions, read 
        surface markers, or evaluate interface integrals.

        This method dynamically queries the bulk dimension of the mesh and constructs:
        1. The missing topological entity definitions for all sub-dimensions.
        2. The upward connectivity mapping each sub-entity back to its parent cell(s).

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The distributed volume or surface mesh whose topology needs synchronization.
        """
        topology = mesh.topology
        cell_dim = topology.dim
        
        # Force DOLFINx to generate topological entity indices for 0D vertices
        topology.create_entities(0)
        
        # Dynamically build entities and upward connectivity for all intermediate dimensions
        # (e.g., loops through 1D edges and 2D facets if it's a 3D mesh)
        for entity_dim in range(1, cell_dim):
            topology.create_entities(entity_dim)
            topology.create_connectivity(entity_dim, cell_dim)
            
        # Always map the 0D vertices upward to the primary cells
        topology.create_connectivity(0, cell_dim)
        
    def _scale_mesh_coordinates(self, mesh: dolfinx.mesh.Mesh, scale_factor: float) -> None:
        """
        Scale the raw geometric coordinates of the mesh in-place.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            The target distributed mesh layout.
        scale_factor : float
            Multiplier applied across all geometric nodal coordinates.
        """
        mesh.geometry.x[:, :] *= scale_factor
        
        
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
    