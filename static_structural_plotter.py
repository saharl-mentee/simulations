import os
import numpy as np
import pyvista
import ufl
from dolfinx import plot, fem


class BaseStructuralViewer:
    """Shared scaffolding for structural plots: mesh/displacement extraction,
    grid formatting, and interactive point-probing. Mirrors the
    BaseMeshViewer pattern in thermal_plotter.py, adapted for a
    vector-valued (displacement) field instead of a scalar one.
    """

    AXIS_MAP = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}

    def __init__(self, u, function_space, title="Structural Viewer", off_screen=None):
        # off_screen=None defers to pyvista's global OFF_SCREEN setting, matching
        # thermal_plotter's BaseMeshViewer; pass an explicit bool to override it.
        plotter_kwargs = {"title": title}
        if off_screen is not None:
            plotter_kwargs["off_screen"] = off_screen
        self.plotter = pyvista.Plotter(**plotter_kwargs)

        # Extract the DOLFINx mesh topology and geometry
        topology, cell_types, geometry = plot.vtk_mesh(function_space)
        self.grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        # Attach the displacement vector field and its magnitude
        num_components = function_space.dofmap.index_map_bs
        displacement_vectors = u.x.array.reshape((-1, num_components))
        self.grid["Displacement"] = displacement_vectors
        self.grid.set_active_vectors("Displacement")
        self.grid["Total Deflection"] = np.linalg.norm(displacement_vectors, axis=1)

        self._probe_grid = None
        self._probe_field = None
        self._probe_unit = ""

    def warp(self, scaling_factor=1.0):
        """Returns a copy of the grid deflected by the displacement field, to
        visually exaggerate structural deformation."""
        return self.grid.warp_by_vector(factor=scaling_factor)

    def apply_standard_grid_formatting(self):
        """Applies consistent grid boundary formatting across all plotters."""
        self.plotter.show_grid(
            xtitle="X-Axis",
            ytitle="Y-Axis",
            ztitle="Z-Axis",
            grid=True,
            location="outer",
            ticks="both",
            font_size=14,
            color="black",
            fmt="%.2f",
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True,
        )

    def enable_probing(self, target_grid, field_name, unit=""):
        """Wires up interactive point-picking on `target_grid`, reporting the
        `field_name` scalar at the picked location — mirrors thermal_plotter's
        probe_callback pattern.
        """
        self._probe_grid = target_grid
        self._probe_field = field_name
        self._probe_unit = unit
        self.plotter.enable_point_picking(
            callback=self._probe_callback, show_message=True, font_size=12
        )
        print("Instructions: Hover over the mesh and press 'P' to pick a point.")

    def _probe_callback(self, point):
        idx = self._probe_grid.find_closest_point(point)
        value = self._probe_grid.point_data[self._probe_field][idx]
        actual_mesh_coord = self._probe_grid.points[idx]

        self.plotter.add_point_labels(
            [actual_mesh_coord],
            [f"{value:.3e}{self._probe_unit}"],
            name="probe",
            font_size=20,
            point_size=10,
            always_visible=True,
        )

        formatted_loc = np.array2string(
            actual_mesh_coord, formatter={'float_kind': lambda x: f'{x:.2f}'}, separator=', '
        )
        print(f"Clicked Point ID: {idx} | Location: {formatted_loc} | {self._probe_field}: {value:.3e}{self._probe_unit}")


class StressVolumeViewer(BaseStructuralViewer):
    """Renders the undeformed structural mesh colored by total deflection
    magnitude — a quick raw-data viewer, analogous to thermal_plotter's
    VolumeViewer."""

    def __init__(self, u, function_space_V, to_probe=True):
        super().__init__(u, function_space_V, title="3D Solution Viewer")

        sargs = dict(fmt="%.3e", title_font_size=20, label_font_size=15, color="black")
        self.plotter.add_mesh(
            self.grid, scalars="Total Deflection", show_edges=True, cmap="rainbow", scalar_bar_args=sargs
        )

        if to_probe:
            self.enable_probing(self.grid, "Total Deflection", unit=" m")

        self.plotter.add_axes()
        self.apply_standard_grid_formatting()


class DeflectionViewer(BaseStructuralViewer):
    """Plots the deformed shape of a 3D structural body, colored by total
    deflection magnitude.

    Parameters
    ----------
    u : dolfinx.fem.Function
        The displacement vector field solution from the solver.
    function_space : dolfinx.fem.FunctionSpace
        The vector function space used for the simulation.
    scaling_factor : float, optional
        A multiplier to exaggerate the displacement visually. Real-world
        structural deflections are often microscopic (e.g., 0.0001 meters).
        Setting this to 100.0 or 1000.0 makes the deformation clearly
        visible. Default is 1.0.
    """

    def __init__(self, u, function_space, scaling_factor=1.0, to_probe=True):
        super().__init__(u, function_space, title="Deformed Shape Viewer")

        self.warped_grid = self.warp(scaling_factor)

        self.plotter.add_mesh(
            self.warped_grid,
            scalars="Total Deflection",
            cmap="turbo",
            show_edges=True,
            scalar_bar_args={"title": "Total Deflection (m)"},
        )
        # Faint wireframe of the original undeformed shape for visual reference
        self.plotter.add_mesh(self.grid, color="white", style="wireframe", opacity=0.15)
        self.plotter.title = f"Deformed Shape (Scaling Factor: {scaling_factor}x)"

        if to_probe:
            self.enable_probing(self.warped_grid, "Total Deflection", unit=" m")

        self.apply_standard_grid_formatting()


def compute_von_mises_from_simulator(u: fem.Function, lambda_field, mu_field) -> fem.Function:
    """Computes the scalar Von Mises stress field by directly reusing the
    spatially varying Lamé fields already defined in the Simulator.
    """
    mesh = u.function_space.mesh

    # 1. Define Strain Tensor using UFL
    def epsilon(v):
        return ufl.sym(ufl.grad(v))

    # 2. Hooke's Law using the pre-existing, multi-material Lamé fields
    def sigma(v):
        return lambda_field * ufl.tr(epsilon(v)) * ufl.Identity(len(v)) + 2.0 * mu_field * epsilon(v)

    # 3. Compute Deviatoric Stress tensor components
    s = sigma(u) - (1.0 / 3.0) * ufl.tr(sigma(u)) * ufl.Identity(len(u))
    von_mises_expr = ufl.sqrt(3.0 / 2.0 * ufl.inner(s, s))

    # 4. Project the expression into a continuous CG1 space for smooth visualization
    V_scalar = fem.functionspace(mesh, ("Lagrange", 1))
    von_mises_field = fem.Function(V_scalar, name="Von_Mises_Stress_Pa")

    expr = fem.Expression(von_mises_expr, V_scalar.element.interpolation_points)
    von_mises_field.interpolate(expr)

    return von_mises_field


class VonMisesViewer(BaseStructuralViewer):
    """Plots Von Mises stress mapped onto the deformed structural geometry,
    with automatic headless-environment detection (bypasses Xvfb errors on
    WSLg vs. headless servers).

    Call `.show()` to render — it opens an interactive window when a display
    is available, or saves a screenshot when running headless.
    """

    def __init__(self, u, von_mises, scaling_factor=1.0, to_probe=True, interactive=True):
        self.is_root = u.function_space.mesh.comm.rank == 0
        if not self.is_root:
            self.plotter = None
            return

        has_display = "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ
        self.use_offscreen = not interactive or not has_display

        super().__init__(u, u.function_space, title="Von Mises Stress Viewer", off_screen=self.use_offscreen)
        self.grid["VonMises_Stress"] = von_mises.x.array
        self.warped_grid = self.warp(scaling_factor)

        if self.use_offscreen:
            print("📺 [Plotter] Headless environment detected. Rendering mesh directly to system memory...")
        else:
            print("🚀 [Plotter] Native display link detected (WSLg). Launching interactive GUI window...")

        self.plotter.add_text("Static Structural Analysis: Von Mises Stress Field", font_size=12, color="white")
        self.plotter.add_mesh(
            self.warped_grid,
            scalars="VonMises_Stress",
            cmap="jet",  # Classic FEA blue-to-red stress gradient map
            show_edges=True,
            edge_color="black",
            line_width=0.5,
            scalar_bar_args={"title": "Von Mises Stress [Pa]", "vertical": True},
        )
        # Light wireframe outline of the un-deformed geometry for visual baseline reference
        self.plotter.add_mesh(self.grid, style="wireframe", color="white", opacity=0.15, label="Original Profile")
        self.plotter.add_axes()
        self.plotter.view_isometric()

        if to_probe and not self.use_offscreen:
            self.enable_probing(self.warped_grid, "VonMises_Stress", unit=" Pa")

        self.apply_standard_grid_formatting()

    def show(self, output_image="von_mises_stress_result.png"):
        """Renders interactively, or saves a screenshot when running headless."""
        if not self.is_root:
            return
        if self.use_offscreen:
            self.plotter.screenshot(output_image)
            self.plotter.close()
            print(f"✅ [Plotter Success] Headless visualization completed. Image saved to:\n   👉 {os.path.abspath(output_image)}")
        else:
            self.plotter.show()
