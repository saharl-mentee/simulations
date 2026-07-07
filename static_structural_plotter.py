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

class StressVolumeViewer(BaseMeshViewer):
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
   
        
def plot_deflection(u, function_space, scaling_factor=1.0):
    """Plots the total deflection of a 3D structural body using PyVista.
    
    Parameters
    ----------
    u : dolfinx.fem.Function
        The displacement vector field solution from the solver.
    function_space : dolfinx.fem.FunctionSpace
        The vector function space used for the simulation.
    scaling_factor : float, optional
        A multiplier to exaggerate the displacement visually. Real-world structural 
        deflections are often microscopic (e.g., 0.0001 meters). Setting this to 
        100.0 or 1000.0 makes the deformation clearly visible. Default is 1.0.
    """
    # 1. Extract mesh topology, cell types, and geometry compatible with VTK/PyVista
    # This automatically handles both linear (order 1) and quadratic (order 2) elements.
    topology, cell_types, x = plot.vtk_mesh(function_space)
    
    # 2. Create the PyVista Unstructured Grid object
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    
    # 3. Extract and reshape the flat FEniCSx array into clean 3D vectors [u_x, u_y, u_z]
    num_components = function_space.dofmap.index_map_bs  # Should be 3 for 3D vector space
    displacement_vectors = u.x.array.reshape((-1, num_components))
    
    # Assign the 3D vectors to the PyVista grid array
    grid["Displacement"] = displacement_vectors
    
    # 4. Calculate the combined X, Y, Z values (Magnitude / Total Deflection)
    # This computes sqrt(ux^2 + uy^2 + uz^2) for every single node
    total_deflection = np.linalg.norm(displacement_vectors, axis=1)
    grid["Total Deflection"] = total_deflection
    
    # 5. Warp (deflect) the geometry by the displacement vectors
    grid.set_active_vectors("Displacement")
    warped_grid = grid.warp_by_vector(factor=scaling_factor)
    
    # 6. Set up the PyVista Plotter environment
    plotter = pyvista.Plotter()
    
    # Add the deflected body colored by the combined total deflection scalar
    plotter.add_mesh(
        warped_grid, 
        scalars="Total Deflection", 
        cmap="turbo",               # "turbo" or "jet" are standard for engineering analysis
        show_edges=True, 
        scalar_bar_args={"title": "Total Deflection (m)"}
    )
    
    # Optional: Draw a faint wireframe of the original undeformed shape for visual reference
    plotter.add_mesh(grid, color="white", style="wireframe", opacity=0.15)
    
    plotter.title = f"Deformed Shape (Scaling Factor: {scaling_factor}x)"
    plotter.show()