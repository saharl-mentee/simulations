import gc
import re
import meshio
import pandas as pd
import numpy as np
from typing import Tuple, List, defaultdict
from pathlib import Path

ELEMENT_TYPE_MAPPING = {
    "tetra": "volume",
    "hexahedron": "volume",
    "triangle": "surface",
    "quad": "surface",
    "line": "edge", 
    "vertex": "point"
}

class Msh2Xdmf:
    """Converts Gmsh (.msh) files to synchronized XDMF/H5 files for FEniCSx."""

    def __init__(self, path: Path, save_path: Path = None):
        self.path = path
        self.save_path = save_path if save_path is not None else path
        self.mesh = meshio.read(path)

    # 🔄 CHANGED: Completely rewrote the conversion flow. 
    # The old version looped through 'save_elements' independently for each cell type.
    # The new version forces all boundaries to align to a master 3D grid first.
    def convert(self) -> None:
        self.check_geo_duplicate_physical_groups(self.path.with_suffix(".geo"))
         
        cells_dict = self.mesh.cells_dict
        cell_types = list(cells_dict.keys())
        
        # 1. NEW LOGIC: Identify whether this is a Tetrahedral or Hexahedral 3D mesh
        bulk_type = None
        for t in ["tetra", "hexahedron"]:
            if t in cell_types:
                bulk_type = t
                break
        
        if bulk_type is None:
            raise ValueError("Could not find a 3D bulk element type (tetra or hexahedron) in the mesh.")
            
        # 2. NEW LOGIC: Create a single MASTER point layout using ONLY nodes referenced by 3D cells.
        # This acts as the baseline coordinate map for the entire simulation.
        bulk_cells = cells_dict[bulk_type]
        unique_indices, remapped_flat_bulk = np.unique(bulk_cells, return_inverse=True)
        master_points = self.mesh.points[unique_indices]
        
        # 3. NEW LOGIC: Build a translation lookup dictionary.
        # Maps: original_gmsh_vertex_id -> new_synchronized_vertex_id
        global_to_local = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
        
        # 4. CHANGED: Process and save the primary volume file using the master layout
        remapped_bulk_cells = remapped_flat_bulk.reshape(bulk_cells.shape)
        bulk_tags = self.mesh.cell_data_dict["gmsh:physical"][bulk_type]
        
        volume_mesh = meshio.Mesh(
            points=master_points,
            cells={bulk_type: remapped_bulk_cells},
            cell_data={"Grid": [bulk_tags]}
        )
        vol_path = self.save_path.with_stem(self.save_path.stem + "_volume").with_suffix(".xdmf")
        meshio.write(vol_path, volume_mesh)
        print(f"[Exported] Clean volume mesh: {vol_path.name}")
        
        # 5. CHANGED: Process boundary layers (surfaces/edges) using the EXACT SAME master layout
        for el_type in cell_types:
            if el_type == bulk_type:
                continue
                
            # Keep isolated 0D tracking points completely independent
            if el_type == "vertex":
                pt_tags = self.mesh.cell_data_dict["gmsh:physical"]["vertex"]
                pt_cells = cells_dict["vertex"]
                
                # Look up exactly what index these nodes hold in the master 3D layout
                synchronized_pt_cells = np.array([
                    [global_to_local[old_idx]] for old_idx in pt_cells.flatten()
                ], dtype=np.int32)
                
                pt_mesh = meshio.Mesh(
                    points=master_points,  # Forces identical 3D coordinate layout matching the volume
                    cells={"vertex": synchronized_pt_cells}, # Uses native 'vertex' keys
                    cell_data={"Grid": [pt_tags]}
                )
                pt_path = self.save_path.with_stem(self.save_path.stem + "_point").with_suffix(".xdmf")
                meshio.write(pt_path, pt_mesh)
                
                # Post-Processing: Fix meshio's hardcoded "Polyvertex" text generation
                with open(pt_path, "r") as f:
                    xml_content = f.read()
                
                # Dynamically swap the words in the generated XML text
                fixed_xml = xml_content.replace('TopologyType="Polyvertex"', 'TopologyType="Vertex"')
                
                with open(pt_path, "w") as f:
                    f.write(fixed_xml)
                    
                print(f"[Exported] Perfectly synchronized native FEA Vertex file: {pt_path.name}")
                continue
                
            # 6. NEW LOGIC: Remap the boundary cells (triangles, quads, or lines) 
            # to point to the master volume's vertex indices instead of their own unique indices.
            bnd_cells = cells_dict[el_type]
            bnd_tags = self.mesh.cell_data_dict["gmsh:physical"][el_type]
            
            valid_cells = []
            valid_tags = []
            
            for cell, tag in zip(bnd_cells, bnd_tags):
                # Only save boundary elements whose vertices exist inside the 3D volume grid
                if all(v in global_to_local for v in cell):
                    # Translate the old vertex ID to the new synchronized master index
                    remapped_cell = [global_to_local[v] for v in cell]
                    valid_cells.append(remapped_cell)
                    valid_tags.append(tag)
            
            if valid_cells:
                # 7. CRITICAL CHANGE: We pass 'master_points' here instead of generating a unique subset.
                # This ensures the surface file and volume file share an identical node index structure.
                bnd_mesh = meshio.Mesh(
                    points=master_points,  
                    cells={el_type: np.array(valid_cells, dtype=np.int32)},
                    cell_data={"Grid": [np.array(valid_tags, dtype=np.int32)]}
                )
                bnd_path = self.save_path.with_stem(self.save_path.stem + f"_{ELEMENT_TYPE_MAPPING[el_type]}").with_suffix(".xdmf")
                meshio.write(bnd_path, bnd_mesh)
                print(f"[Exported] Synchronized boundary layer: {bnd_path.name}")
                
        gc.collect()

    # (The validation checks remain the same)
    def check_geo_duplicate_physical_groups(self, path) -> None:
        pattern = r'Physical (Surface|Volume)\("([^"]+)",\s*(\d+)?\)\s*=\s*\{([^}]+)\};'
        with open(path, "r") as f:
            content = f.read()
            matches = re.findall(pattern, content)
            entity_to_groups = self.regroup_geometry_entities(matches)
            duplicates = [[s_id, groups] for s_id, groups in entity_to_groups.items() if len(groups) > 1]
            if duplicates:
                df = pd.DataFrame(duplicates, columns=["Surface No", "Common Physical Group"])
                raise ValueError(f"Geometry entities assigned to MULTIPLE groups:\n {df.to_string(index=False)}")

    @staticmethod
    def regroup_geometry_entities(matches) -> defaultdict:
        entity_to_groups = defaultdict(list)
        if not matches:
            raise ValueError("No physical groups found in the file layout.")
        for geometry_type, group_name, group_tag, entity_id in matches:
            entity_ids = [int(e_id.strip()) for e_id in entity_id.split(",")]
            for e_id in entity_ids:
                key = f"{e_id} ({geometry_type})"
                entity_to_groups[key].append([group_name, group_tag])
        return entity_to_groups

if __name__ == "__main__":
    path = Path(r"/mnt/c/Users/saharl/Documents/simulations_api/simulations/test_dxf_exporter/standard_fin2.msh")
    exporter = Msh2Xdmf(path)
    exporter.convert()