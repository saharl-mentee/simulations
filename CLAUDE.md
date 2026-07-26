# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A collection of standalone FEniCSx (dolfinx) finite element scripts for thermal and structural
analysis of heatsink/fin geometries (CAD → mesh → solve → visualize). There is no package
structure, build system, or test suite — each `.py` file is run directly via its
`if __name__ == "__main__":` block, usually with a hardcoded absolute path to a specific mesh
under one of the data directories.

## Environment

Use the dedicated virtualenv, not system Python:

```bash
source /home/saharl/fenicsx_venv/bin/activate
python3 <script>.py
```

Key packages: `fenics-dolfinx` 0.10 (dolfinx API, NOT legacy `dolfin`/`fenics`), `fenics-ufl`,
`fenics-basix`, `gmsh`/`pygmsh`, `meshio`, `petsc4py`, `pyvista` (3D viz), `h5py`, `pandas`, `numpy`.

`fe_thermal_fins.py` and `evaluate_heat_transfer_no_fins.py` are older/reference scripts —
the former imports legacy `dolfin`/`fenics` (pre-X FEniCS) and is not compatible with the
dolfinx environment above; the latter is a hand-calc analytical check using `pint`, unrelated
to the FEM pipeline.

## Pipeline: CAD → mesh → simulation → plot

1. **CAD/geometry** — SolidWorks files (`.SLDPRT`/`.SLDASM`) exported to `.STEP`, then meshed in
   Gmsh to produce a `.geo` (physical group definitions) and `.msh` file. Each data directory
   (`fin_assembly/`, `motor_fin/`, `solid_circular/`, `solid_test*/`, `test_dxf_exporter/`,
   `test_linear_elasticity/`) holds one geometry's full artifact chain.
2. **`.msh` → `.xdmf`/`.h5`** — dolfinx cannot read `.msh` directly. Convert first with
   `Msh2Xdmf` in **`msh2xdmf_2.py`** (current version — synchronizes all boundary element types
   against one master volume point layout so that node indices line up across files). Produces
   `<name>_volume.xdmf/.h5` plus one file per other element type found in the mesh
   (`_surface`, `_edge`, `_point`). `msh_2_xdmf.py` is the older per-element-type independent
   exporter kept for reference/comparison — prefer `msh2xdmf_2.py` for new work.
   Both validate that no Gmsh physical group is assigned to overlapping geometry entities by
   parsing the sibling `.geo` file (`check_geo_duplicate_physical_groups`) before converting.
3. **Loading** — `mesh_loader.py`'s `Mesh3DLoader` takes the path *stem* (no suffix) shared by
   the `_volume`/`_surface`/`_edge`/`_point` XDMF files, picks the volume mesh as primary (falls
   back to surface-only), builds full topology connectivity for every sub-dimension so BCs can
   be located on any entity dimension, and optionally rescales mm → m (`in_mm=True`, default).
   Returns `(mesh, tags)` where `tags` is a dict keyed by `"cell"/"surface"/"edge"/"point"` →
   `MeshTags | None`. Point tags are matched by nearest-neighbor proximity (not exact topology,
   since Gmsh point exports don't preserve dolfinx's internal vertex numbering) — see
   `_match_physical_group_to_mesh_nodes` (tolerance 0.1 mm).
4. **Simulation** — `thermal_simulator.py` (`ThermalSimulator`, steady-state heat conduction,
   Dirichlet/Robin/Neumann BCs by physical-group tag) and `statics_simulation.py`
   (`StaticStructuralSimulator`, linear elasticity, same BC pattern via displacement/traction
   tuples). Both follow the same constructor shape: `(geometry_path, materials, dirichlet_bcs,
   ..., elements_order=2)` where `materials` and BC args are tuples of
   `(physical_group_tag, value)`. Material properties (conductivity, or E/nu → Lamé constants)
   come from `material_library.py`'s `ThermalMaterialLibrary`/`ElasticMaterialLibrary` registries,
   looked up by name and painted onto a DG0 function per volume tag. `StaticStructuralSimulator`
   additionally auto-detects which loaded tag dict entry is facet-dimensional for the traction
   measure, and its Dirichlet BC application scans across *all* tag dimensions (cell/surface/
   edge/point) to find whichever entity the tag lives on — thermal's Dirichlet BC assumes the
   tag is always on a facet.
5. **Visualization** — `thermal_plotter.py` (`VolumeViewer`/`SliceViewer`/`FluxViewer`) and
   `static_structural_plotter.py` (`StressVolumeViewer`, `plot_deflection`, `plot_von_mises`,
   `compute_von_mises_from_simulator`) both build on pyvista, sharing a `BaseMeshViewer` pattern
   that converts a dolfinx function space to a `pyvista.UnstructuredGrid`, applies a 1000x scale
   (m → mm display), and optionally wires up interactive point-probing
   (`enable_point_picking`, press `P`).

## Working with a new geometry

To run a simulation on a new part: mesh it in Gmsh with physical groups for every volume and
BC surface, then run `Msh2Xdmf(path).convert()` from `msh2xdmf_2.py` before instantiating
`ThermalSimulator`/`StaticStructuralSimulator` — both simulators' `load_mesh()` expects the
`_volume.xdmf` (and sibling) files to already exist next to the `.msh`, they do not convert
on the fly. Physical group tags used in `materials`/BC tuples must match the integer tags
assigned in the `.geo` file. Use `mesh_sanity_checker.py`'s `MeshSanityChecker` to verify
point/coordinate synchronization across the exported `_volume`/`_surface`/`_edge`/`_point`
files (and vertex tag alignment) before debugging a solver failure — most "BC tag not found"
or misaligned-geometry issues trace back to a conversion mismatch here, not the solver.

## Conventions

- Geometry files are authored in millimeters; `Mesh3DLoader(..., in_mm=True)` (default) rescales
  to meters for the physics. Don't rescale twice.
- Physical group tags (ints) are the join key between `.geo`/`.msh`, the tags dict returned by
  `Mesh3DLoader`, and the `materials`/BC tuples passed into the simulators.
- `elements_order=1` is used for quick/coarse debug runs (e.g. structural on a raw `.msh`
  directly in some `__main__` blocks); production runs default to order 2.
