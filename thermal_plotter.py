import pyvista
from dolfinx import plot
import numpy as np
from dolfinx import fem

from base_plotter import BaseProbeableViewer


class BaseMeshViewer(BaseProbeableViewer):
    """Base class to encapsulate shared mesh parsing and data attachment for
    thermal (scalar temperature) fields. Builds on BaseProbeableViewer
    (plotter setup, grid formatting, point-probing)."""

    def __init__(self, u, function_space_V, title="Mesh Viewer"):
        super().__init__(title=title)

        # Extract the DOLFINx mesh topology and geometry
        topology, cell_types, geometry = plot.vtk_mesh(function_space_V)
        self.grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        # Attach the data vector
        self.grid.point_data["Temperature"] = u.x.array.real
        self.grid.set_active_scalars("Temperature")


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
            self.enable_probing(self.plot_grid, "Temperature", unit="°C", value_fmt=".3f")

        self.plotter.add_axes()
        self.apply_standard_grid_formatting()


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
        
        # Initialization
        self.update_slice_view(norm_val=self.norm_val)

        if to_probe:
            self.enable_probing(self.current_slice, "Temperature", unit="°C", value_fmt=".3f")

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
                # Keep the probe target in sync with whichever slice is visible
                self._probe_grid = slc
                self.actor = self.plotter.add_mesh(
                    slc, cmap="rainbow", name="active_slice",
                    scalar_bar_args=sargs, pickable=True
                )
        except Exception as e:
            print(f"Slice failed at {actual_pos}: {e}")


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
            self.enable_probing(
                self.derived_grid, "Flux_Magnitude", unit=" W/m²", value_fmt=".3e", vector_field="Heat_Flux"
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


import dolfinx
from dolfinx import fem
import ufl
import numpy as np
import pyvista as pv
from mpi4py import MPI

def compute_von_mises(u: fem.Function, E: float, nu: float) -> fem.Function:
    """Computes the scalar Von Mises stress field from a displacement field u.
    Outputs a continuous CG1 scalar function for smooth visual plotting.
    """
    mesh = u.function_space.mesh
    
    # 1. Compute Lamé constants from Young's Modulus and Poisson's Ratio
    mu = E / (2.0 * (1.0 + nu))
    lambda_ = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    # 2. Define Strain and Stress Tensors via UFL Expressions
    def epsilon(v):
        return ufl.sym(ufl.grad(v))
    
    def sigma(v):
        return lambda_ * ufl.tr(epsilon(v)) * ufl.Identity(len(v)) + 2.0 * mu * epsilon(v)
    
    s = sigma(u) - (1.0 / 3.0) * ufl.tr(sigma(u)) * ufl.Identity(len(u))
    von_mises_expr = ufl.sqrt(3.0 / 2.0 * ufl.inner(s, s))
    
    # 3. Create a continuous linear scalar space (CG1) to store the stress values
    V_scalar = fem.functionspace(mesh, ("Lagrange", 1))
    von_mises_field = fem.Function(V_scalar, name="Von_Mises_Stress_Pa")
    
    # 4. Interpolate the complex mathematical expression into our scalar function
    expr = fem.Expression(von_mises_expr, V_scalar.element.interpolation_points())
    von_mises_field.interpolate(expr)
    
    return von_mises_field


def plot_von_mises(u: fem.Function, von_mises: fem.Function, scaling_factor: float = 1.0):
    """Generates an interactive 3D PyVista plot showing Von Mises stress distributions
    mapped directly onto the physically deformed grid.
    """
    mesh = u.function_space.mesh
    
    # Prune plotting if running on non-zero MPI ranks
    if mesh.comm.rank != 0:
        return

    # 1. Initialize PyVista unstructured grid exporter
    pv.start_xvfb() # Ensures headless environments don't crash
    pv_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim))
    
    # 2. Extract and attach the displacement data vectors
    num_points = mesh.geometry.x.shape[0]
    u_values = u.x.array.reshape((num_points, 3))
    pv_grid.point_data["Displacement"] = u_values
    
    # 3. Extract and attach our computed Von Mises stress values
    pv_grid.point_data["VonMises_Stress"] = von_mises.x.array
    
    # 4. Warp the visual mesh based on the deflection vectors
    warped_grid = pv_grid.warp_by_vector("Displacement", factor=scaling_factor)
    
    # 5. Build the interactive visual render window
    plotter = pv.Plotter(window_size=[1024, 768])
    plotter.add_text("Static Structural Analysis: Von Mises Stress Field", font_size=12, color="white")
    
    # Add the deformed mesh colored by stress intensity
    plotter.add_mesh(
        warped_grid,
        scalars="VonMises_Stress",
        cmap="jet",                 # Classic FEA blue-to-red stress gradient map
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        scalar_bar_args={"title": "Von Mises Stress [Pa]", "vertical": True}
    )
    
    # Overlay an outline of the original, un-deformed CAD geometry for scale contrast
    plotter.add_mesh(pv_grid, style="wireframe", color="white", opacity=0.15, label="Original Profile")
    
    plotter.add_axes()
    plotter.view_isometric()
    plotter.show()