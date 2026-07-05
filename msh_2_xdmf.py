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
    """convert the msh file to xdmf and h5 files that can be read by fenicsx."""


    def __init__(self, path: Path, save_path: Path = None):
        """Initialize the MeshConverter and load the mesh data.

        Parameters
        ----------
        path : Path
            the path to the .msh file to be converted.
        to_save : Path, optional
            new place to save the files to. defaults to the same folder.
            if not None, should have full path, with wanted the file name, with no sufix.
            the function will add sufix and volume/surface to the files respectively.
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
        Execute the full mesh conversion workflow from .msh to .xdmf.

        The process includes:
        1. Validating the associated .geo file for duplicate physical groups.
        2. Identifying the element types (e.g., tetrahedral vs. hexahedral).
        3. Extracting and saving volume elements to a dedicated XDMF and h5 file.
        4. Extracting and saving surface elements to a dedicated XDMF and h5 file.

        Raises
        ------
        ValueError
            If duplicate physical groups are found in the .geo file 
            or if the mesh element type is unsupported.
        FileNotFoundError
            If the required .geo file is missing.
        """
        self.check_geo_duplicate_physical_groups(self.path.with_suffix(".geo"))
         
        cells = list(self.mesh.cells_dict.keys())
        [self.save_elements(el_type) for el_type in cells]


    def check_geo_duplicate_physical_groups(self, path) -> None:
        """verifier that the user didnt mistakeably put a geometry entity into more
        than one physical group.

        Parameters
        ----------
        path : Path
            Path to the .geo file that generated the .msh file. should be in the same
            folder and with the same name for convenience.

        Raises
        ------
        ValueError
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
        """Extract and save the mesh elements of a specific type to an XDMF file.

        Parameters
        ----------
        element_type : str
            The type of mesh elements to extract (e.g., 'tetra', 'hexahedron', 'triangle', 'quad').
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
        Groups geometry entities with their assigned physical groups.

        Parameters
        ----------
        matches : list of tuple
            A list of regex matches where each tuple contains:
            (geometry_type, group_name, group_tag, entity_id).

        Returns
        -------
        defaultdict(list)
            The updated entity_to_groups dictionary where keys are formatted
            as "{id} ({type})".
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
    
