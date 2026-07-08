import numpy as np
import pyvista


class BaseProbeableViewer:
    """Shared scaffolding for pyvista-based FEM result viewers: plotter setup,
    consistent grid formatting, and interactive point-probing.

    Subclasses build their own grid and attach field data (scalar or vector)
    in their own __init__, then call `enable_probing` once the grid they want
    to probe is ready.
    """

    AXIS_MAP = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}

    def __init__(self, title="Mesh Viewer", off_screen=None):
        # off_screen=None defers to pyvista's global OFF_SCREEN setting;
        # pass an explicit bool to override it for a specific viewer.
        plotter_kwargs = {"title": title}
        if off_screen is not None:
            plotter_kwargs["off_screen"] = off_screen
        self.plotter = pyvista.Plotter(**plotter_kwargs)

        self._probe_grid = None
        self._probe_field = None
        self._probe_unit = ""
        self._probe_fmt = ".3e"
        self._probe_vector_field = None

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

    def enable_probing(self, target_grid, field_name, unit="", value_fmt=".3e", vector_field=None):
        """Wires up interactive point-picking on `target_grid`.

        Reports the `field_name` scalar (and optionally a `vector_field`, e.g.
        heat flux or displacement) at the picked location. Press 'P' while
        hovering the mesh to probe a point.
        """
        self._probe_grid = target_grid
        self._probe_field = field_name
        self._probe_unit = unit
        self._probe_fmt = value_fmt
        self._probe_vector_field = vector_field
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
            [f"{value:{self._probe_fmt}}{self._probe_unit}"],
            name="probe",
            font_size=20,
            point_size=10,
            always_visible=True,
        )

        formatted_loc = np.array2string(
            actual_mesh_coord, formatter={'float_kind': lambda x: f'{x:.2f}'}, separator=', '
        )
        message = (
            f"Clicked Point ID: {idx} | Location: {formatted_loc} | "
            f"{self._probe_field}: {value:{self._probe_fmt}}{self._probe_unit}"
        )

        if self._probe_vector_field is not None:
            vector = self._probe_grid.point_data[self._probe_vector_field][idx]
            formatted_vec = np.array2string(
                vector, formatter={'float_kind': lambda x: f'{x:.2e}'}, separator=', '
            )
            message += f" | Vector: {formatted_vec}"

        print(message)
