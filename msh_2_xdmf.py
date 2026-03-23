from typing import Tuple


import meshio
from pathlib import Path
import re
from collections import defaultdict
import pandas as pd




def msh_to_xdmf(path: Path, to_save: Path = None) -> None:
    """convert the msh file to xdmf and h5 files that can be read by fenicsx.


    Parameters
    ----------
    path : Path
        the path to the .msh file to be converted.
    to_save: Path, default to None
        new place to save the files to. defaults to the same folder.
    """


    check_geo_duplicate_physical_groups(path.with_suffix(".geo"))


    msh = meshio.read(path)
    vol_type, surface_type = determine_elements_type(msh)


    save_volume_elements(msh, vol_type)
    save_surface_elements(msh, surface_type)




def determine_elements_type(mesh) -> Tuple[str, str]:
    """detects the type of elements in the msh file. whether they are hexagonal or tetraheadral


    Parameters
    ----------
    mesh : meshio.mesh.Mesh
        the loaded mesh from the .msh file.


    Returns
    -------
    Tuple[str, str]
        volumetric type and surface type of the mesh.
    """
    if "hexahedron" in mesh.cells_dict:
        vol_type, surf_type = "hexahedron", "quad"
    else:
        vol_type, surf_type = "tetra", "triangle"
    return vol_type, surf_type




def check_geo_duplicate_physical_groups(path):
    # Regex to find Physical Surface("Name") = {ID1, ID2, ...};
    pattern = r'Physical (Surface|Volume)\("([^"]+)",\s*(\d+)?\)\s*=\s*\{([^}]+)\};'


    surface_to_groups = defaultdict(list)


    with open(path, "r") as f:
        content = f.read()
        matches = re.findall(pattern, content)


        for group_type, group_name, group_num, ids_str in matches:
            # Clean up the IDs (remove spaces and split by comma)
            ids = [int(id.strip()) for id in ids_str.split(",")]
            for s_id in ids:
                surface_to_groups[f"{s_id}, ({group_type})"].append(
                    [group_name, group_num]
                )


        duplicate_faces = [
            [s_id, groups]
            for s_id, groups in surface_to_groups.items()
            if len(groups) > 1
        ]
        headlines = ["Surface No", "Common Physical Group"]
        df = pd.DataFrame(duplicate_faces, columns=headlines)
        if len(duplicate_faces) > 1:
            raise ValueError(
                f"Surfaces are assigned to MULTIPLE groups:\n {df.to_string(index=False)}"
            )




def save_volume_elements(mesh, vol_type):
    # 1. Extract and save the 3D Volume (The Mesh)
    # This includes the tetrahedrons (Type 4)
    vol_mesh = meshio.Mesh(
        points=mesh.points,
        cells={vol_type: mesh.cells_dict[vol_type]},
        cell_data={"Grid": [mesh.cell_data_dict["gmsh:physical"][vol_type]]},
    )
    path_vol = path.with_stem(path.stem + "_volume").with_suffix(".xdmf")
    meshio.write(path_vol, vol_mesh)




def save_surface_elements(mesh, surface_type):
    # 2. Extract and save the 2D Facets (The Boundary Tags)
    # This includes the triangles (Type 2) and their Physical Group IDs
    facet_mesh = meshio.Mesh(
        points=mesh.points,
        cells={surface_type: mesh.cells_dict[surface_type]},
        cell_data={"Grid": [mesh.cell_data_dict["gmsh:physical"][surface_type]]},
    )
    path_surface = path.with_stem(path.stem + "_surface").with_suffix(".xdmf")
    meshio.write(path_surface, facet_mesh)




if __name__ == "__main__":
    path = Path(
        r"/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/fin_assembly/fin_asm2.msh"
    )
    msh_to_xdmf(path)



