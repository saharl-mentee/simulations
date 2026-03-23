import re
import meshio
import pandas as pd
from typing import Tuple
from pathlib import Path
from collections import defaultdict


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

        vol_type, surface_type = self.determine_elements_type()
        self.save_volume_elements(vol_type)
        self.save_surface_elements(surface_type)

    def determine_elements_type(self) -> Tuple[str, str]:
        """detects the type of elements in the msh file. whether they are hexagonal or tetraheadral

        Returns
        -------
        Tuple[str, str]
            volumetric type and surface type of the mesh.
        """
        if "hexahedron" in self.mesh.cells_dict:
            volume_type, surface_type = "hexahedron", "quad"
        elif "tetra" in self.mesh.cells_dict:
            volume_type, surface_type = "tetra", "triangle"
        else:
            raise ValueError("type of mesh is not hexa nor tetra")
        return volume_type, surface_type

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

    def save_volume_elements(self, vol_type: str) -> None:
        """Extract and save the 3D Volume (The Mesh)
        This includes the tetrahedrons (Type 4) and hexahesron (Type 5)

        Parameters
        ----------
        vol_type : str
        """
        vol_mesh = meshio.Mesh(
            points=self.mesh.points,
            cells={vol_type: self.mesh.cells_dict[vol_type]},
            cell_data={"Grid": [self.mesh.cell_data_dict["gmsh:physical"][vol_type]]},
        )
        save_path = self.save_path.with_stem(self.save_path.stem + "_volume").with_suffix(".xdmf")
        meshio.write(save_path, vol_mesh)

    def save_surface_elements(self, surface_type: str) -> None:
        """Extract and save the 2D Facets (The Boundary Tags)
        This includes the triangles (Type 2) and quadrilaterals (Type 3)

        Parameters
        ----------
        surface_type : str
        """
        facet_mesh = meshio.Mesh(
            points=self.mesh.points,
            cells={surface_type: self.mesh.cells_dict[surface_type]},
            cell_data={"Grid": [self.mesh.cell_data_dict["gmsh:physical"][surface_type]]},
        )
        save_path = self.save_path.with_stem(save_path.stem + "_surface").with_suffix(".xdmf")
        meshio.write(save_path, facet_mesh)
        
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
        r"/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/fin_assembly/fin_asm2.msh"
    )
    exporter = Msh2Xdmf(path)
    exporter.convert()
    
