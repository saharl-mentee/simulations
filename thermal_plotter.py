import pyvista
from dolfinx import plot
import numpy as np


class BaseMeshViewer:
    """Base class to encapsulate shared mesh parsing, data attachment, and formatting."""
    
    AXIS_MAP = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}

    def __init__(self, u, function_space_V, title="Mesh Viewer"):
        self.plotter = pyvista.Plotter(title=title)
        
        # Extract the DOLFINx mesh topology and geometry
        topology, cell_types, geometry = plot.vtk_mesh(function_space_V)
        self.grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        
        # Attach the data vector
        self.grid.point_data["Data"] = u.x.array.real
        self.grid.set_active_scalars("Data")
        
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


class VolumeViewer(BaseMeshViewer):
    def __init__(self, u, function_space_V, to_probe=True):
        super().__init__(u, function_space_V, title="3D Solution Viewer")
        
        # Scale the grid (1000x scaling factor)
        self.plot_grid = self.grid.scale([1000, 1000, 1000], inplace=False)
        
        sargs = dict(fmt="%.3f", title_font_size=20, label_font_size=15, color="black")
        self.plotter.add_mesh(
            self.plot_grid, show_edges=True, cmap="rainbow", scalar_bar_args=sargs
        )
        
        if to_probe:
            self.plotter.enable_point_picking(
                callback=self.probe_callback, show_message=True, font_size=12
            )
            print("Instructions: Hover over the mesh and press 'P' to pick a point.")
            
        self.plotter.add_axes()
        self.apply_standard_grid_formatting()

    def probe_callback(self, point):
        """Probes the physical 3D volume grid directly."""
        idx = self.plot_grid.find_closest_point(point)
        value = self.plot_grid.point_data["Data"][idx]
        actual_mesh_coord = self.plot_grid.points[idx]

        self.plotter.add_point_labels(
            [actual_mesh_coord], 
            [f"{value:.3f}°C"], 
            name="probe", 
            font_size=20,
            point_size=10,
            always_visible=True 
        )

        formatted_loc = np.array2string(
            actual_mesh_coord, formatter={'float_kind': lambda x: f'{x:.2f}'}, separator=', '
        )
        print(f"Clicked Point ID: {idx} | Location: {formatted_loc} | Data: {value:.3f}°C")
    

class SliceViewer(BaseMeshViewer):
    def __init__(self, u, function_space_V, to_probe=True):
        super().__init__(u, function_space_V, title="Normalized Fin Probe (0-100%)")
        
        self.axis = "x"
        self.norm_val = 0.5
        self.current_slice = None
        self.actor = None
        self.widgets = {}
        
        # Background reference wireframe
        self.plotter.add_mesh(
            self.grid, style="wireframe", opacity=0.1, color="white", pickable=False
        )
        
        # UI Elements Configuration
        self.widgets["slider"] = self.plotter.add_slider_widget(
            self.update_slice_view,
            [0.0, 1.0],
            value=self.norm_val,
            title="Position",
            fmt="",  # Hides the unhelpful internal [0.00 - 1.00] floating value text
            pointa=(0.4, 0.2),
            pointb=(0.9, 0.2),
        )
        
        self.widgets["x"] = self.plotter.add_checkbox_button_widget(
            lambda flag: self.set_axis("x", flag), value=True, position=(10, 150), size=30, color_on="red"
        )
        self.widgets["y"] = self.plotter.add_checkbox_button_widget(
            lambda flag: self.set_axis("y", flag), value=False, position=(10, 80), size=30, color_on="green"
        )
        self.widgets["z"] = self.plotter.add_checkbox_button_widget(
            lambda flag: self.set_axis("z", flag), value=False, position=(10, 10), size=30, color_on="blue"
        )
        
        if to_probe:
            self.plotter.enable_point_picking(callback=self.probe_callback, show_message=True)
            
        # Initialization
        self.update_slice_view(norm_val=self.norm_val)
        self.apply_standard_grid_formatting()
        
    def set_axis(self, axis_name, flag):
        if flag:
            self.axis = axis_name
            
            # Synchronize radio button logic checkboxes visual states
            for ax in ["x", "y", "z"]:
                state_val = 1 if ax == axis_name else 0
                self.widgets[ax].GetRepresentation().SetState(state_val)
                    
            self.update_slice_view()

    def update_slice_view(self, norm_val=None):
        """Calculates physical position from normalized 0-1 slider state."""
        if norm_val is not None:
            self.norm_val = norm_val

        # Calculate physical position using current bounding limits
        b = self.grid.bounds
        idx_min, idx_max = self.AXIS_MAP[self.axis]
        actual_pos = b[idx_min] + (b[idx_max] - b[idx_min]) * self.norm_val

        # Dynamically push structural updates onto the text slider representation
        if "slider" in self.widgets:
            axis_label = self.axis.upper()
            slider_rep = self.widgets["slider"].GetRepresentation()
            slider_rep.SetTitleText(f"{axis_label} Axis Position: {actual_pos:.3f}")
            self.plotter.render()

        # Evict old cross section visualization slice actor
        if self.actor:
            self.plotter.remove_actor(self.actor)

        origin = [0, 0, 0]
        origin[{"x": 0, "y": 1, "z": 2}[self.axis]] = actual_pos

        try:
            sargs = dict(fmt="%.3f", title_font_size=20, label_font_size=15, color="black")
            slc = self.grid.slice(normal=self.axis, origin=origin)
            
            if slc.n_points > 0:
                self.current_slice = slc
                self.actor = self.plotter.add_mesh(
                    slc, cmap="rainbow", name="active_slice", 
                    scalar_bar_args=sargs, pickable=True
                )
        except Exception as e:
            print(f"Slice failed at {actual_pos}: {e}")

    def probe_callback(self, point):
        """Probes data points localized strictly on the active visible slice plane."""
        if self.current_slice is None:
            return  

        idx = self.current_slice.find_closest_point(point)
        value = self.current_slice.point_data["Data"][idx]
        actual_mesh_coord = self.current_slice.points[idx]

        self.plotter.add_point_labels(
            [actual_mesh_coord], 
            [f"{value:.3f}°C"], 
            name="probe", 
            font_size=20,
            point_size=10,
            always_visible=True  
        )

        axis_label = self.axis.upper()
        print(f"Slice Pick | Axis: {axis_label} | Data: {value:.3f}°C")