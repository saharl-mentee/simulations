import meshio
from pathlib import Path
import re
from collections import defaultdict
import pandas as pd


def msh_to_xdmf(path: Path):
    # Make sure to export msh in ASCII 2 format! otherwise wont work.
    # when meshing in gmsh, dont forget to make physical groups of all boundaris,
    # and of the inner volume!

    # Load the .msh file
    msh = meshio.read(path)

    # Extract cells and data for the volume (Tetrahedrons)
    # If your mesh is 2D, change 'tetra' to 'triangle'
    tetra_mesh = meshio.Mesh(
        points=msh.points,
        cells={"tetra": msh.cells_dict["tetra"]},
        cell_data={"name_to_read": [msh.cell_data_dict["gmsh:physical"]["tetra"]]},
    )

    # Write to XDMF
    meshio.write(path.with_suffix(".xdmf"), tetra_mesh)


def msh_to_xdmf2(path: Path):
    msh = meshio.read(path)

    # 1. Extract and save the 3D Volume (The Mesh)
    # This includes the tetrahedrons (Type 4)
    tetra_mesh = meshio.Mesh(
        points=msh.points, cells={"tetra": msh.cells_dict["tetra"]}
    )
    path_vol = path.with_stem(path.stem + "_volume").with_suffix(".xdmf")
    meshio.write(path_vol, tetra_mesh)

    # 2. Extract and save the 2D Facets (The Boundary Tags)
    # This includes the triangles (Type 2) and their Physical Group IDs
    facet_mesh = meshio.Mesh(
        points=msh.points,
        cells={"triangle": msh.cells_dict["triangle"]},
        cell_data={"name_to_read": [msh.cell_data_dict["gmsh:physical"]["triangle"]]},
    )
    path_surface = path.with_stem(path.stem + "_surface").with_suffix(".xdmf")
    meshio.write(path_surface, facet_mesh)


def msh_to_xdmf_universal(path: Path):
    check_geo_duplicate_physical_groups(path.with_suffix(".geo"))
    msh = meshio.read(path)

    # Determine which 3D elements exist
    if "hexahedron" in msh.cells_dict:
        vol_type, surf_type = "hexahedron", "quad"
    else:
        vol_type, surf_type = "tetra", "triangle"

    # 1. Extract and save the 3D Volume (The Mesh)
    # This includes the tetrahedrons (Type 4)
    vol_mesh = meshio.Mesh(
        points=msh.points,
        cells={vol_type: msh.cells_dict[vol_type]},
        cell_data={"Grid": [msh.cell_data_dict["gmsh:physical"][vol_type]]},
    )
    path_vol = path.with_stem(path.stem + "_volume").with_suffix(".xdmf")
    meshio.write(path_vol, vol_mesh)

    # 2. Extract and save the 2D Facets (The Boundary Tags)
    # This includes the triangles (Type 2) and their Physical Group IDs
    facet_mesh = meshio.Mesh(
        points=msh.points,
        cells={surf_type: msh.cells_dict[surf_type]},
        cell_data={"Grid": [msh.cell_data_dict["gmsh:physical"][surf_type]]},
    )
    path_surface = path.with_stem(path.stem + "_surface").with_suffix(".xdmf")
    meshio.write(path_surface, facet_mesh)


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


if __name__ == "__main__":
    path = Path(
        r"/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/fin_assembly/fin_asm2.msh"
    )
    msh_to_xdmf_universal(path)
