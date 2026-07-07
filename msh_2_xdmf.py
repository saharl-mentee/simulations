import gc
import re
import meshio
import pandas as pd
from typing import Tuple
from pathlib import Path
from collections import defaultdict


ELEMENT_TYPE_MAPPING = {
    "tetra": "volume",
    "hexahedron": "volume",
    "triangle": "surface",
    "quad": "surface",
    "line": "edge", 
    "vertex": "point"
}

class Msh2Xdmf:
    """
    Convert Gmsh .msh files into XDMF and H5 formats compatible with FEniCSx.

    This class extracts mesh topologies, checks for physical group integrity 
    within corresponding Gmsh geometry files, and segregates different element 
    dimensions into individual sub-meshes for seamless finite element ingestion.
    """

    def __init__(self, path: Path, save_path: Path = None):
        """Initialize the MeshConverter and load the mesh data.

        Parameters
        ----------
        path : Path
            the path to the .msh file to be converted.
        to_save : Path, optional
            new place to save the files to. defaults to the same folder.
            if not None, should have full path, with the desired file name, with no sufix.
            the function will add sufix and type to the files respectively.
            default to None.
            
        Attributes
        ----------
        path : Path
            Stored path to the source mesh.
        to_save : Path or None
            Destination path for output files.
        mesh : meshio.Mesh
            The mesh object loaded from the input file using meshio.
        """
        self.path = path
        self.save_path = save_path if save_path is not None else path
        self.mesh = meshio.read(path)

    def convert(self) -> None:
        """
        Execute the pipeline to convert the loaded mesh into XDMF/H5 formats.

        This method validates the integrity of the mesh mapping using the matching 
        `.geo` file before extracting and writing every individual element cell 
        topology present in the mesh file to disk.

        Raises
        ------
        ValueError
            If geometry entities are found to be cross-assigned to multiple 
            overlapping physical groups.
        FileNotFoundError
            If the corresponding `.geo` file cannot be located at the expected path.
        """
        self.check_geo_duplicate_physical_groups(self.path.with_suffix(".geo"))
         
        cells = list(self.mesh.cells_dict.keys())
        [self.save_elements(el_type) for el_type in cells]


    def check_geo_duplicate_physical_groups(self, path) -> None:
        """
        Verify that geometry entities are uniquely assigned to one physical group.

        Parses the corresponding `.geo` file to ensure that no single entity 
        (surface or volume) is mistakenly shared across multiple discrete 
        Gmsh physical groups, which could otherwise corrupt boundary or domain 
        markers in subsequent simulations.

        Parameters
        ----------
        path : Path
            Path to the `.geo` file that generated the source `.msh` file.

        Raises
        ------
        ValueError
            If any geometry entity IDs overlap across multiple physical groups.
        FileNotFoundError
            If the target `.geo` file does not exist at the provided location.
        """
        pattern = r'Physical (Surface|Volume)\("([^"]+)",\s*(\d+)?\)\s*=\s*\{([^}]+)\};'

        with open(path, "r") as f:
            content = f.read()
            matches = re.findall(pattern, content)

            entity_to_groups = self.regroup_geometry_entities(matches)

            duplicates = [
                [s_id, groups]
                for s_id, groups in entity_to_groups.items()
                if len(groups) > 1
            ]
            if duplicates:
                headlines = ["Surface No", "Common Physical Group"]
                df = pd.DataFrame(duplicates, columns=headlines)
                raise ValueError(
                    f"Geometry entities are assigned to MULTIPLE groups:\n {df.to_string(index=False)}"
                )

    def save_elements(self, element_type:str) -> None:
        """
        Extract and isolate a specific element topology type to a separate XDMF file.

        Creates a mesh subset isolating only the requested element type, binds 
        its associated physical group data markers, writes it to disk, and forces 
        immediate garbage collection to optimize memory overhead.

        Parameters
        ----------
        element_type : str
            The specific topology identifier to extract from the mesh data 
            (e.g., 'tetra', 'hexahedron', 'triangle', 'quad', 'line').

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the requested `element_type` does not exist inside the global 
            `ELEMENT_TYPE_MAPPING` registry.
        """
        
        
        mesh_subset = meshio.Mesh(
            points=self.mesh.points,
            cells={element_type: self.mesh.cells_dict[element_type]},
            cell_data={"Grid": [self.mesh.cell_data_dict["gmsh:physical"][element_type]]},
        )
        save_path = self.save_path.with_stem(self.save_path.stem + f"_{ELEMENT_TYPE_MAPPING[element_type]}").with_suffix(".xdmf")
        meshio.write(save_path, mesh_subset)
        del mesh_subset
        gc.collect()
        
    @staticmethod
    def regroup_geometry_entities(matches) -> defaultdict[list]:
        """
        Regroup raw regex geometry string matches into a structured index map.

        Processes raw token configurations scraped out of `.geo` scripts, tokenizes 
        comma-separated lists of sub-entity numbers, and indexes them directly 
        against their parent physical group metadata tags.

        Parameters
        ----------
        matches : list of tuple of str
            A list containing regex group extractions where each tuple represents:
            (geometry_type, group_name, group_tag, entity_ids_str).

        Returns
        -------
        defaultdict of list
            A dictionary tracking assignments where keys are strings formatted 
            as "{entity_id} ({geometry_type})" and values are lists of component 
            lists containing `[group_name, group_tag]`.

        Raises
        ------
        ValueError
            If the `matches` collection is empty, indicating a total lack of 
            definable physical entities within the checked source.
        """
        entity_to_groups = defaultdict(list)
        if not matches:
            raise ValueError("No physical group found in the .msh file")
        for geometry_type, group_name, group_tag, entity_id in matches:
            entity_ids = [int(e_id.strip()) for e_id in entity_id.split(",")]
            for entity_id in entity_ids:
                key = f"{entity_id} ({geometry_type})"
                entity_to_groups[key].append([group_name, group_tag])
        return entity_to_groups
    

def comprehensive_mesh_audit_xdmf(base_path: Path):
    print("==================================================")
    print(f"🔍 COMPREHENSIVE MESH AUDIT FOR: {base_path.name}")
    print("==================================================")
    
    # 1. Define paths for all expected XDMF outputs
    vol_path = base_path.with_name(f"{base_path.name}_volume.xdmf")
    surf_path = base_path.with_name(f"{base_path.name}_surface.xdmf")
    edge_path = base_path.with_name(f"{base_path.name}_edge.xdmf")
    pt_path = base_path.with_name(f"{base_path.name}_point.xdmf")
    h5_pt_path = base_path.with_name(f"{base_path.name}_point.h5")
    
    # 2. Load primary volume mesh (The Master Baseline)
    if not vol_path.exists():
        print(f"❌ CRITICAL ERROR: Volume file does not exist at {vol_path}")
        return False
        
    vol = meshio.read(vol_path)
    master_points = vol.points
    print(f"📦 [Master baseline] Volume points shape: {master_points.shape}")
    vol_type = list(vol.cells_dict.keys())[0]
    vol_tags = np.unique(list(vol.cell_data_dict["Grid"].values())[0])
    print(f"     • {len(vol.cells_dict[vol_type])} {vol_type} elements tagged with IDs: {vol_tags}")
    
    # 3. Audit 2D Surface domains (If they exist)
    if surf_path.exists():
        surf = meshio.read(surf_path)
        print(f"\n📐 [Auditing Surface] Points shape: {surf.points.shape}")
        
        if surf.points.shape != master_points.shape or not np.allclose(surf.points, master_points, atol=1e-6):
            print("  ❌ COMPATIBILITY ERROR: Surface point layout is NOT synchronized with the Volume!")
            return False
            
        surf_type = list(surf.cells_dict.keys())[0]
        surf_tags = np.unique(list(surf.cell_data_dict["Grid"].values())[0])
        print(f"  ✅ Synchronized! {len(surf.cells_dict[surf_type])} {surf_type} elements tagged with IDs: {surf_tags}")
    else:
        print("\nℹ️  No surface file found (Skipping).")

    # 4. Audit 1D Edge domains (If they exist)
    if edge_path.exists():
        edge = meshio.read(edge_path)
        print(f"\n線 [Auditing Edge] Points shape: {edge.points.shape}")
        
        if edge.points.shape != master_points.shape or not np.allclose(edge.points, master_points, atol=1e-6):
            print("  ❌ COMPATIBILITY ERROR: Edge point layout is NOT synchronized with the Volume!")
            return False
            
        edge_type = list(edge.cells_dict.keys())[0]
        edge_tags = np.unique(list(edge.cell_data_dict["Grid"].values())[0])
        print(f"  ✅ Synchronized! {len(edge.cells_dict[edge_type])} {edge_type} elements tagged with IDs: {edge_tags}")
    else:
        print("\nℹ️  No edge file found (Skipping).")

    # 5. 🛠️ FIXED: Audit 0D Point domain via Direct H5 reading to bypass meshio's 'Vertex' KeyError
    if h5_pt_path.exists():
        print(f"\n📍 [Auditing H5 Points] File: {h5_pt_path.name}")
        
        with h5py.File(h5_pt_path, "r") as f:
            # meshio maps coordinates to data0, cell definitions to data1, and tags to data2
            actual_coords = np.array(f["data0"])
            pt_tags = np.array(f["data2"]).flatten()
            
        print(f"  • Found {len(actual_coords)} tracking point(s) in H5 binary data.")
        
        for pt_coord, pt_tag in zip(actual_coords, pt_tags):
            # Calculate the distance from this tracking point to every node in the 3D master volume
            distances = np.linalg.norm(master_points - pt_coord, axis=1)
            closest_node_idx = np.argmin(distances)
            
            # If the closest 3D mesh node is practically at the same coordinate, it's a match!
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


def blindly_inspect_h5(h5_path: Path):
    if not h5_path.exists():
        print(f"❌ Error: File not found at {h5_path}")
        return

    print("==================================================")
    print(f"📂 DYNAMICALLY PROBING BINARY H5 FILE: {h5_path.name}")
    print("==================================================")

    with h5py.File(h5_path, "r") as f:
        datasets = []
        
        # Internal visitor function to gather all raw dataset paths
        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, obj))
        
        f.visititems(collect_datasets)
        
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
            
            # 1. Look for Coordinate Matrix: Float data with an (N, 3) matrix layout
            if len(shape) == 2 and shape[1] == 3 and np.issubdtype(dtype, np.floating):
                print("   🎯 IDENTIFIED: This dataset holds your Vertex Coordinates!")
                print("   📋 Printing first 5 node coordinates (Node ID ──> X, Y, Z):")
                for idx, coords in enumerate(data_array[:5]):
                    print(f"      Node [{idx:4d}] : {coords.tolist()}")
                if len(data_array) > 5:
                    print(f"      ... and {len(data_array) - 5} more vertices.")
                    
            # 2. Look for 3D Hexahedron Elements: Integer data with an (N, 8) cell connection map
            elif len(shape) == 2 and shape[1] == 8 and np.issubdtype(dtype, np.integer):
                print("   🎯 IDENTIFIED: This dataset holds your 3D Element Topology (Hexahedrons)!")
                print("   📋 Printing first 3 elements (Element ID ──> Master Vertex IDs):")
                for idx, cell in enumerate(data_array[:3]):
                    print(f"      Element [{idx:4d}] : {cell.tolist()}")
                    
            # 3. Look for 2D Boundary Facets (Triangles): Integer data with an (N, 3) cell connection map
            elif len(shape) == 2 and shape[1] == 3 and np.issubdtype(dtype, np.integer):
                print("   🎯 IDENTIFIED: This dataset holds your 2D Boundary Face Topology (Triangles)!")
                print("   📋 Printing first 3 surface elements:")
                for idx, cell in enumerate(data_array[:3]):
                    print(f"      Element [{idx:4d}] : {cell.tolist()}")
                    
            # 4. Look for 1D Edges (Lines): Integer data with an (N, 2) connectivity shape
            elif len(shape) == 2 and shape[1] == 2 and np.issubdtype(dtype, np.integer):
                print("   🎯 IDENTIFIED: This dataset holds your 1D Edge Topology (Lines)!")
                
            # 5. Look for Physical Material / Boundary Tags: Column Vector or Flat array
            elif (len(shape) == 1) or (len(shape) == 2 and shape[1] == 1):
                unique_tags = np.unique(data_array)
                print(f"   🎯 IDENTIFIED: This dataset holds your Physical Tags Array!")
                print(f"   🏷️  Unique Subdomain/Physical Group IDs stored here: {unique_tags.tolist()}")
                
            print("-" * 50)
            
    print("==================================================\n")
    

def inspect_point_h5_file(h5_path: Path):
    if not h5_path.exists():
        print(f"❌ Error: Point H5 file not found at {h5_path}")
        return

    print("==================================================")
    print(f"📍 PROBING POINT H5 DATASET: {h5_path.name}")
    print("==================================================")

    with h5py.File(h5_path, "r") as f:
        datasets = []
        
        # Gather all internal dataset paths dynamically
        f.visititems(lambda name, obj: datasets.append((name, obj)) if isinstance(obj, h5py.Dataset) else None)
        
        print(f"🔍 Discovered {len(datasets)} raw datasets inside the point file structure.\n")
        print("-" * 50)

        for name, dset in datasets:
            shape = dset.shape
            dtype = dset.dtype
            data_array = np.array(dset)
            
            print(f"📄 Dataset Path: /{name}")
            print(f"   Array Shape: {shape} | Data Type: {dtype}")
            
            # 1. Vertex Coordinates Array
            if len(shape) == 2 and shape[1] == 3 and np.issubdtype(dtype, np.floating):
                print("   🎯 WHAT THIS IS: The physical coordinates of your tracking points.")
                print("   📋 Full Coordinate List:")
                for idx, coords in enumerate(data_array):
                    # print(f"      Index [{idx}] ──> (X: {coords[0]:.2f}, Y: {coords[1]:.2f}, Z: {coords[2]:.2f})")
                    pass
                    
            # 2. Integer Arrays (Cell map vs Physical Tags)
            elif np.issubdtype(dtype, np.integer):
                print("   🎯 WHAT THIS IS: Integer topology map or physical tag indices.")
                print(f"   📋 Sample Data (Showing up to 5 rows):")
                
                # Flatten the data representation for clean terminal printing
                flat_sample = data_array.flatten()[:5]
                for idx, val in enumerate(flat_sample):
                    print(f"      Row [{idx}] value ──> {val}")
                
                # Provide a helper note to help you tell them apart instantly
                unique_vals = np.unique(data_array)
                if len(unique_vals) <= 5:
                    print(f"   💡 Note: This array only contains specific tag values: {unique_vals.tolist()}")
                    print("      ──> Confirmed: This dataset represents your PHYSICAL TAGS!")
                else:
                    print("   💡 Note: This array contains sequential or high-range node index values.")
                    print("      ──> Confirmed: This dataset represents your CELL TOPOLOGY (Master Node IDs)!")

            print("-" * 50)
            
    print("==================================================\n")


# --- How to run it ---
if __name__ == "__main__":
    import numpy as np
    import h5py
    # Point this to your file prefix (without _volume.xdmf etc.)
    base_prefix = Path("/mnt/c/Users/saharl/Documents/simulations_api/simulations/test_dxf_exporter/standard_fin3")
    comprehensive_mesh_audit_xdmf(base_prefix)
    blindly_inspect_h5(base_prefix.with_name(f"{base_prefix.name}_volume.h5"))
    inspect_point_h5_file(base_prefix.with_name(f"{base_prefix.name}_point.h5"))
    


# if __name__ == "__main__":
#     path = Path(
#         r"/mnt/c/Users/saharl/Documents/simulations_api/simulations/test_dxf_exporter/standard_fin2.msh"
#     )
#     exporter = Msh2Xdmf(path)
#     exporter.convert()
    
