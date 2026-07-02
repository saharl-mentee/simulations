import pyvista
from dolfinx import plot
import numpy as np
from dolfinx import fem


class BaseMeshViewer:
    """Base class to encapsulate shared mesh parsing, data attachment, and formatting."""
    
    AXIS_MAP = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}

    def __init__(self, u, function_space_V, title="Mesh Viewer"):
        self.plotter = pyvista.Plotter(title=title)
        
        # Extract the DOLFINx mesh topology and geometry
        topology, cell_types, geometry = plot.vtk_mesh(function_space_V)
        self.grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        
        # Attach the data vector
        self.grid.point_data["Temperature"] = u.x.array.real
        self.grid.set_active_scalars("Temperature")
        
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
        value = self.plot_grid.point_data["Temperature"][idx]
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
        print(f"Clicked Point ID: {idx} | Location: {formatted_loc} | Temperature: {value:.3f}°C")
    

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
        value = self.current_slice.point_data["Temperature"][idx]
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
        print(f"Slice Pick | Axis: {axis_label} | Temperature: {value:.3f}°C")
        
        
class FluxViewer(BaseMeshViewer):
    """Computes and visualizes spatial Heat Flux vectors using local material conductivity."""
    
    def __init__(self, u, function_space_V, conduction_coeff, to_probe=True):
        super().__init__(u, function_space_V, title="3D Flux Vector Field Viewer")
        
        self.arrow_actor = None
        self.widgets = {}
        self.current_factor = 1  # Keep track of the current scalar multiplier
        
        # 1. Project/Interpolate the DG0 conductivity function into the Lagrange space
        # This aligns the cell-based conductivity directly to the PyVista nodes (points)
        k_lagrange = fem.Function(function_space_V)
        k_lagrange.interpolate(conduction_coeff)
        self.grid.point_data["Conductivity"] = k_lagrange.x.array.real
        
        # 2. Scale the grid (1000x factor consistent with your original VolumeViewer)
        self.plot_grid = self.grid.scale([1000, 1000, 1000], inplace=False)
        
        # 3. Compute the spatial gradient of the Temperature data
        self.derived_grid = self.plot_grid.compute_derivative(scalars="Temperature", gradient="Temperature_Gradient")
        
        # 4. Calculate Heat Flux: q = -k * grad(T)
        # Using numpy broadcasting [:, None] to multiply the (N,) conductivity array by the (N,3) gradient
        k_spatial = self.derived_grid.point_data["Conductivity"]
        grad_t = self.derived_grid.point_data["Temperature_Gradient"]
        self.derived_grid.point_data["Heat_Flux"] = -k_spatial[:, None] * grad_t
        
        # 5. Compute vector magnitudes for scalar color mapping
        self.derived_grid.point_data["Flux_Magnitude"] = np.linalg.norm(
            self.derived_grid.point_data["Heat_Flux"], axis=1
        )
        self.derived_grid.set_active_scalars("Flux_Magnitude")
        self.derived_grid.set_active_vectors("Heat_Flux")

        # 6. Render the volume colored by Heat Flux Magnitude
        sargs = dict(fmt="%.3e", title_font_size=18, label_font_size=14, color="black")
        self.plotter.add_mesh(
            self.derived_grid, show_edges=True, cmap="viridis", scalar_bar_args=sargs, opacity=0.7
        )
        
        # 7. KEYBOARD ROUTING: Bind the 'S' key to pull up an input dialog box
        self.plotter.add_key_event("s", self.prompt_for_scale)
        
        print("\n" + "="*60)
        print("INSTRUCTIONS:")
        print("  • Hover over the mesh and Right-Click (or press 'P') to probe a data point.")
        print("  • Press 'S' on your keyboard to TYPE a manual arrow scale factor.")
        print("="*60 + "\n")
        
        # Generate the initial arrow field layout automatically
        self.update_arrow_scale(self.current_factor)

        if to_probe:
            self.plotter.enable_point_picking(
                callback=self.probe_callback, show_message=True, font_size=12
            )

        self.plotter.add_axes()
        self.apply_standard_grid_formatting()
        
    def prompt_for_scale(self):
        """Pops up a standard native entry dialog box to let you type an exact scale number."""
        import tkinter as tk
        from tkinter import simpledialog
        
        # Setup a temporary hidden framework root so a stray background window doesn't hang
        root = tk.Tk()
        root.withdraw()
        
        # Open a text numerical entry prompt modal on top of the screen
        new_factor = simpledialog.askfloat(
            title="Arrow Scale Configuration", 
            prompt="Enter a precise scale multiplier factor for the vector arrows:", 
            initialvalue=self.current_factor
        )
        
        # If the user typed a number and clicked 'OK' (didn't press 'Cancel')
        if new_factor is not None:
            print(f"Applying new arrow scale factor: {new_factor:.2e}")
            self.current_factor = new_factor
            self.update_arrow_scale(new_factor)
            
        # Safely collapse the hidden background frame
        root.destroy()

    def update_arrow_scale(self, factor):
        """Re-draws arrows cleanly with the chosen configuration sizing."""
        if self.arrow_actor:
            self.plotter.remove_actor(self.arrow_actor)
            
        arrows = self.derived_grid.glyph(
            orient="Heat_Flux", 
            scale="Flux_Magnitude", 
            factor=factor, 
            tolerance=0.03
        )
        
        self.arrow_actor = self.plotter.add_mesh(arrows, color="orange", name="flux_vectors")
        self.plotter.render()

    def probe_callback(self, point):
        """Probes the heat flux vector field details at any clicked point."""
        idx = self.derived_grid.find_closest_point(point)
        
        # 2. Extract the values from self.derived_grid (not plot_grid!)
        magnitude = self.derived_grid.point_data["Flux_Magnitude"][idx]
        vector = self.derived_grid.point_data["Heat_Flux"][idx]
        actual_mesh_coord = self.derived_grid.points[idx]

        # 3. Add the label flag to the viewport
        self.plotter.add_point_labels(
            [actual_mesh_coord], 
            [f"{magnitude:.3e} W/m²"], 
            name="probe", font_size=18, point_size=10, always_visible=True 
        )

        # 4. Print the exact metrics out to the terminal log
        formatted_vec = np.array2string(vector, formatter={'float_kind': lambda x: f'{x:.2e}'}, separator=', ')
        print(f"Flux Probe | ID: {idx} | Magnitude: {magnitude:.3e} | Vector: {formatted_vec}")
        