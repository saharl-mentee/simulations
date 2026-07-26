import sys
import h5py
import meshio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class MeshSanityChecker:
    """
    An evaluation utility class to handle geometric and topological audits
    for finite element mesh pipelines (XDMF/H5).
    """

    def __init__(self):
        pass

    def comprehensive_mesh_audit_xdmf(self, base_path: Path) -> bool:
        """
        Perform a comprehensive spatial alignment audit across all dimensional entities.
        """
        print("==================================================")
        print(f"🔍 COMPREHENSIVE MESH AUDIT FOR: {base_path.name}")
        print("==================================================")
        
        vol_path = base_path.with_name(f"{base_path.name}_volume.xdmf")
        surf_path = base_path.with_name(f"{base_path.name}_surface.xdmf")
        edge_path = base_path.with_name(f"{base_path.name}_edge.xdmf")
        h5_pt_path = base_path.with_name(f"{base_path.name}_point.h5")
        
        if not vol_path.exists():
            print(f"❌ CRITICAL ERROR: Volume file does not exist at {vol_path}")
            return False
            
        vol = meshio.read(vol_path)
        master_points = vol.points
        print(f"📦 [Master baseline] Volume points shape: {master_points.shape}")
        
        vol_type = next(iter(vol.cells_dict.keys()))
        vol_grid_data = vol.cell_data_dict.get("Grid", {})
        vol_tags = np.unique(next(iter(vol_grid_data.values()))) if vol_grid_data else np.array([])
        print(f"     • {len(vol.cells_dict[vol_type])} {vol_type} elements tagged with IDs: {vol_tags.tolist()}")
        
        # Audit 2D Surface domains
        if surf_path.exists():
            surf = meshio.read(surf_path)
            print(f"\n📐 [Auditing Surface] Points shape: {surf.points.shape}")
            if surf.points.shape != master_points.shape or not np.allclose(surf.points, master_points, atol=1e-6):
                print("  ❌ COMPATIBILITY ERROR: Surface point layout is NOT synchronized with the Volume!")
                return False
            surf_type = next(iter(surf.cells_dict.keys()))
            surf_grid_data = surf.cell_data_dict.get("Grid", {})
            surf_tags = np.unique(next(iter(surf_grid_data.values()))) if surf_grid_data else np.array([])
            print(f"  ✅ Synchronized! {len(surf.cells_dict[surf_type])} {surf_type} elements tagged with IDs: {surf_tags.tolist()}")

        # Audit 1D Edge domains
        if edge_path.exists():
            edge = meshio.read(edge_path)
            print(f"\n🌐 [Auditing Edge] Points shape: {edge.points.shape}")
            if edge.points.shape != master_points.shape or not np.allclose(edge.points, master_points, atol=1e-6):
                print("  ❌ COMPATIBILITY ERROR: Edge point layout is NOT synchronized with the Volume!")
                return False
            edge_type = next(iter(edge.cells_dict.keys()))
            edge_grid_data = edge.cell_data_dict.get("Grid", {})
            edge_tags = np.unique(next(iter(edge_grid_data.values()))) if edge_grid_data else np.array([])
            print(f"  ✅ Synchronized! {len(edge.cells_dict[edge_type])} {edge_type} elements tagged with IDs: {edge_tags.tolist()}")

        # Audit 0D Point domain
        if h5_pt_path.exists():
            print(f"\n📍 [Auditing H5 Points] File: {h5_pt_path.name}")
            
            with h5py.File(h5_pt_path, "r") as f:
                # NOTE: [CHANGE] Adjusted unpacking signature to receive the newly discovered topology path key
                coord_key, topo_key, tag_key = self._discover_core_h5_datasets(f)
                
                if not coord_key or not tag_key or not topo_key:
                    print("  ❌ SCHEMA ERROR: Could not map coordinates, topology, or tags inside H5 structure.")
                    return False
                    
                global_coords = np.array(f[coord_key])
                vertex_indices = np.array(f[topo_key]).flatten()
                pt_tags = np.array(f[tag_key]).flatten()
                
                # NOTE: [CHANGE] Crucial Bug Fix: Map the global geometry coordinates using the vertex cell index 
                # tracking array to isolate the TRUE positions of the tracking points.
                actual_coords = global_coords[vertex_indices]
                
            print(f"  • Found {len(actual_coords)} actual tracking point(s) mapped out of {len(global_coords)} global entries.")
            
            for pt_coord, pt_tag in zip(actual_coords, pt_tags):
                distances = np.linalg.norm(master_points - pt_coord, axis=1)
                closest_node_idx = np.argmin(distances)
                
                if distances[closest_node_idx] < 1e-5:
                    print(f"  ✅ Tag {pt_tag}: Coordinate {pt_coord.tolist()} perfectly maps to Volume Node ID {closest_node_idx}")
                else:
                    print(f"  ❌ GEOMETRIC ERROR: Tag {pt_tag} at {pt_coord.tolist()} does not land on any valid mesh vertex!")
                    return False
        else:
            print("\nℹ️  No tracking points H5 file found (Skipping).")

        print("\n==================================================")
        print("🎉 AUDIT COMPLETE: Your entire XDMF mesh pipeline is perfectly compatible!")
        print("==================================================\n")
        return True

    def blindly_inspect_h5(self, h5_path: Path) -> None:
        """
        Dynamically probe the internal dataset matrix architectures contained inside an H5 file.
        """
        if not h5_path.exists():
            print(f"❌ Error: File not found at {h5_path}")
            return

        print("==================================================")
        print(f"📂 DYNAMICALLY PROBING BINARY H5 FILE: {h5_path.name}")
        print("==================================================")

        with h5py.File(h5_path, "r") as f:
            datasets: List[Tuple[str, h5py.Dataset]] = []
            f.visititems(lambda name, obj: datasets.append((name, obj)) if isinstance(obj, h5py.Dataset) else None)
            
            if not datasets:
                print("❌ Error: The H5 file appears to be completely empty.")
                return

            print(f"🔍 Discovered {len(datasets)} raw datasets inside the H5 architecture.\n")
            print("-" * 50)

            for name, dset in datasets:
                shape = dset.shape
                dtype = dset.dtype
                data_array = np.array(dset)
                
                print(f"📄 Real Internal Path: /{name}")
                print(f"   Array Shape: {shape} | Data Type: {dtype}")
                
                # Pass the 'name' string as a parameter to aid identification
                role, description = self._identify_h5_dataset_role(shape, dtype, data_array, name=name)
                if role != "unknown":
                    print(f"   🎯 IDENTIFIED: {description}")
                    
                if role == "coordinates":
                    print("   📋 Printing first 5 node coordinates (Node ID ──> X, Y, Z):")
                    for idx, coords in enumerate(data_array[:5]):
                        print(f"      Node [{idx:4d}] : {coords.tolist()}")
                    if len(data_array) > 5:
                        print(f"      ... and {len(data_array) - 5} more vertices.")
                        
                elif role in ["hex_elements", "triangle_elements"]:
                    print("   📋 Printing first 3 element connections:")
                    for idx, cell in enumerate(data_array[:3]):
                        print(f"      Element [{idx:4d}] : {cell.tolist()}")
                        
                elif role == "tags":
                    print(f"   🏷️  Unique Subdomain/Physical Group IDs stored here: {np.unique(data_array).tolist()}")
                    
                print("-" * 50)
                
        print("==================================================\n")

    def inspect_point_h5_file(self, h5_path: Path) -> None:
        """
        Specialized validation inspection for 0D marker tracking H5 files.

        Parameters
        ----------
        h5_path : Path
            The file system path pointing to the tracking point H5 binary container.
        """
        if not h5_path.exists():
            print(f"❌ Error: Point H5 file not found at {h5_path}")
            return

        print("==================================================")
        print(f"📍 PROBING POINT H5 DATASET: {h5_path.name}")
        print("==================================================")

        with h5py.File(h5_path, "r") as f:
            datasets: List[Tuple[str, h5py.Dataset]] = []
            f.visititems(lambda name, obj: datasets.append((name, obj)) if isinstance(obj, h5py.Dataset) else None)
            
            print(f"🔍 Discovered {len(datasets)} raw datasets inside the point file structure.\n")
            print("-" * 50)

            for name, dset in datasets:
                shape = dset.shape
                dtype = dset.dtype
                data_array = np.array(dset)
                
                print(f"📄 Dataset Path: /{name}")
                print(f"   Array Shape: {shape} | Data Type: {dtype}")
                
                role, description = self._identify_h5_dataset_role(shape, dtype, data_array)
                print(f"   🎯 WHAT THIS IS: {description}")
                
                if np.issubdtype(dtype, np.integer):
                    print(f"   📋 Sample Data (Showing up to 5 rows):")
                    for idx, val in enumerate(data_array.flatten()[:5]):
                        print(f"      Row [{idx}] value ──> {val}")
                        
                print("-" * 50)
                
        print("==================================================\n")

    def verify_exported_points(self, xdmf_path: Path) -> None:
        """
        Read an exported XDMF point definition layout file and execute a structural check.

        Parameters
        ----------
        xdmf_path : Path
            The file system path to the point mesh file.
        """
        if not xdmf_path.exists():
            print(f"❌ Verification Failed: File does not exist at {xdmf_path}")
            return

        mesh = meshio.read(xdmf_path)
        
        print("\n==============================")
        print(f"📊 VERIFYING: {xdmf_path.name}")
        print("==============================")
        
        print(f"✅ Total Points Exported: {len(mesh.points)}")
        
        if len(mesh.points) > 0:
            print(f"   Sample Coordinate Bounds: Min {np.min(mesh.points, axis=0)} -> Max {np.max(mesh.points, axis=0)}")
        
        if "Grid" in mesh.cell_data_dict:
            tags = mesh.cell_data_dict["Grid"]
            if isinstance(tags, list) and len(tags) > 0:
                tags = tags[0]
                
            print("\n🏷️  Associated Physical Tags:")
            for i, tag in enumerate(tags[:10]):  # Cap text print limits to avoiding flooding out screen
                print(f"   Point [{i}] Tag ID: {tag}")
            if len(tags) > 10:
                print(f"   ... and {len(tags) - 10} more point tags.")
        else:
            print("❌ Error: No physical tags found under 'Grid' in this XDMF file.")
            
        print("==============================\n")

    def _identify_h5_dataset_role(self, shape: Tuple[int, ...], dtype: np.dtype, data_array: np.ndarray) -> Tuple[str, str]:
        """
        Analyze an H5 array structure to classify its role in Finite Element topologies.
        """
        if len(shape) == 2 and shape[1] == 3 and np.issubdtype(dtype, np.floating):
            return "coordinates", "Vertex Coordinates (N x 3 Spatial Matrix)"
        if len(shape) == 2 and shape[1] == 8 and np.issubdtype(dtype, np.integer):
            return "hex_elements", "3D Element Topology (Hexahedrons Conn Map)"
        if len(shape) == 2 and shape[1] == 3 and np.issubdtype(dtype, np.integer):
            return "triangle_elements", "2D Boundary Face Topology (Triangles Conn Map)"
        if len(shape) == 2 and shape[1] == 2 and np.issubdtype(dtype, np.integer):
            return "edge_elements", "1D Edge Topology (Lines Conn Map)"
        if (len(shape) == 1) or (len(shape) == 2 and shape[1] == 1):
            unique_vals = len(np.unique(data_array))
            if unique_vals <= 5 or (len(data_array) > 0 and np.max(data_array) < 1000):
                return "tags", f"Physical Tags Array (Unique Identifiers: {np.unique(data_array).tolist()})"
            else:
                return "cell_topology", "CELL TOPOLOGY (Sequential Master Node ID Indirection Map)"
        return "unknown", "Unrecognized general matrix architecture dataset"

    # NOTE: [CHANGE] Expanded return signature tuple to track and return the topology map key identifier
    def _discover_core_h5_datasets(self, h5_file: h5py.File) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Scans an open H5 container dynamically to locate coordinate paths, topology indices, and tag keys.
        """
        coord_path, topo_path, tag_path = None, None, None
        candidate_integers = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype
                # 1. Isolate the master coordinates matrix (N x 3 floats)
                if len(shape) == 2 and shape[1] == 3 and np.issubdtype(dtype, np.floating):
                    nonlocal coord_path
                    coord_path = name
                # 2. Collect flat or single-column integer vectors for evaluation
                elif np.issubdtype(dtype, np.integer) and (len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)):
                    candidate_integers.append((name, obj))

        h5_file.visititems(visitor)

        # Contextual tie-breaker routing for micro-datasets (e.g., 2 points)
        if len(candidate_integers) == 2:
            name1, _ = candidate_integers[0]
            name2, _ = candidate_integers[1]
            
            # Use explicit meshio markers ('data1' vs 'data2') or standard geometric keywords
            if "data1" in name1 or "cell" in name1.lower() or "topo" in name1.lower():
                topo_path = name1
                tag_path = name2
            elif "data2" in name1 or "tag" in name1.lower() or "grid" in name1.lower():
                tag_path = name1
                topo_path = name2
            elif "data1" in name2 or "cell" in name2.lower() or "topo" in name2.lower():
                topo_path = name2
                tag_path = name1
            elif "data2" in name2 or "tag" in name2.lower() or "grid" in name2.lower():
                tag_path = name2
                topo_path = name1
            else:
                # Default safety fallback: alphabetical/structural file ordering order
                sorted_candidates = sorted(candidate_integers, key=lambda x: x[0])
                topo_path = sorted_candidates[0][0]
                tag_path = sorted_candidates[1][0]
                
        elif len(candidate_integers) == 1:
            name = candidate_integers[0][0]
            if "tag" in name.lower() or "grid" in name.lower() or "data2" in name:
                tag_path = name
            else:
                topo_path = name
                
        return coord_path, topo_path, tag_path
    
    def _identify_h5_dataset_role(self, shape: Tuple[int, ...], dtype: np.dtype, data_array: np.ndarray, name: str = "") -> Tuple[str, str]:
        """
        Analyze an H5 array structure to classify its role in Finite Element topologies.
        """
        if len(shape) == 2 and shape[1] == 3 and np.issubdtype(dtype, np.floating):
            return "coordinates", "Vertex Coordinates (N x 3 Spatial Matrix)"
        if len(shape) == 2 and shape[1] == 8 and np.issubdtype(dtype, np.integer):
            return "hex_elements", "3D Element Topology (Hexahedrons Conn Map)"
        if len(shape) == 2 and shape[1] == 3 and np.issubdtype(dtype, np.integer):
            return "triangle_elements", "2D Boundary Face Topology (Triangles Conn Map)"
        if len(shape) == 2 and shape[1] == 2 and np.issubdtype(dtype, np.integer):
            return "edge_elements", "1D Edge Topology (Lines Conn Map)"
            
        if (len(shape) == 1) or (len(shape) == 2 and shape[1] == 1):
            name_lower = name.lower()
            # Intercept array classifications early if explicit keyword traces are found
            if "data1" in name or "cell" in name_lower or "topo" in name_lower:
                return "cell_topology", "CELL TOPOLOGY (Sequential Master Node ID Indirection Map)"
            if "data2" in name or "tag" in name_lower or "grid" in name_lower:
                return "tags", f"Physical Tags Array (Unique Identifiers: {np.unique(data_array).tolist()})"
            
            # Numerical fallback route if string parameters are blank
            unique_vals = len(np.unique(data_array))
            if unique_vals <= 5 or (len(data_array) > 0 and np.max(data_array) < 1000):
                return "tags", f"Physical Tags Array (Unique Identifiers: {np.unique(data_array).tolist()})"
            else:
                return "cell_topology", "CELL TOPOLOGY (Sequential Master Node ID Indirection Map)"
                
        return "unknown", "Unrecognized general matrix architecture dataset"


def run_single_mesh_prefix_audit(base_path_prefix: str | Path) -> None:
    """
    Audit a specific mesh family using its base path prefix.
    """
    base_path = Path(base_path_prefix)
    
    # Define the expected master volume file to verify it exists
    expected_volume_file = base_path.with_name(f"{base_path.name}_volume.xdmf")
    
    print("==================================================")
    print(f"🚀 INITIALIZING SINGLE MESH PIPELINE AUDIT")
    print("==================================================")
    print(f"Target Base Prefix: {base_path.resolve()}")
    print(f"Expected Master Vol: {expected_volume_file.name}\n")
    
    if not expected_volume_file.exists():
        print(f"❌ Error: Cannot find the master volume file at: {expected_volume_file}")
        print("   Ensure the path prefix matches your file naming convention.")
        sys.exit(1)
        
    # Initialize the sanity checker class
    checker = MeshSanityChecker()
    
    # Run the comprehensive audit
    audit_passed = checker.comprehensive_mesh_audit_xdmf(base_path)
    
    print("==================================================")
    print("📊 MESH AUDIT RESULT")
    print("==================================================")
    if audit_passed:
        print("🚀 Success! All associated mesh files match perfectly and are synchronized.")
    else:
        print("🚨 Audit Failed: Geometric or structural mismatches found.")
        sys.exit(1)


if __name__ == "__main__":
    # 🛠️ SET THIS: Pass the exact prefix of your mesh files (no extension or suffix)
    # Example: If your files are /mnt/c/data/my_mesh_volume.xdmf and my_mesh_surface.xdmf,
    # use: "/mnt/c/data/my_mesh"
    MY_MESH_PREFIX = "/mnt/c/Users/saharl/Documents/simulations_api/simulations/test_dxf_exporter/standard_fin3"
    
    # Alternative: Uncomment to pass via terminal: python script.py /path/to/base_name
    # if len(sys.argv) > 1:
    #     MY_MESH_PREFIX = sys.argv[1]
        
    run_single_mesh_prefix_audit(MY_MESH_PREFIX)