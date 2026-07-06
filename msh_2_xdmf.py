import gc
import re
import meshio
import pandas as pd
from typing import Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np


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
        # 1. Grab raw cells and physical tags for this element type
        cells = self.mesh.cells_dict[element_type]
        tags = self.mesh.cell_data_dict["gmsh:physical"][element_type]
        
        # 2. Extract ONLY the unique points actually referenced by these cells
        unique_point_indices, remapped_flat_cells = np.unique(cells, return_inverse=True)
        pruned_points = self.mesh.points[unique_point_indices]
        
        # 3. Reshape the remapped cell indexing back to match original cell topology shape
        remapped_cells = remapped_flat_cells.reshape(cells.shape)
        
        mesh_subset = meshio.Mesh(
            points=pruned_points,
            cells={element_type: remapped_cells},
            cell_data={"Grid": [tags]},
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


if __name__ == "__main__":
    path = Path(
        r"/mnt/c/Users/saharl/Documents/simulations_api/simulations/test_dxf_exporter/standard_fin2.msh"
    )
    exporter = Msh2Xdmf(path)
    exporter.convert()
    
