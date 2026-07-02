from functools import partial
import pyvista
from dolfinx import plot
import numpy as np


def plot_3d_solution(u, function_space_V, to_probe=True):
    # 1. Create a PyVista plotter
    plotter = pyvista.Plotter()

    # 2. Extract the mesh topology and geometry for PyVista
    topology, cell_types, geometry = plot.vtk_mesh(function_space_V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # 3. Attach your temperature solution (uh) to the grid
    grid.point_data["Data"] = u.x.array.real
    grid.set_active_scalars("Data")
    plot_grid = grid.scale([1000, 1000, 1000], inplace=False)
    
    # 4. Add the mesh to the plotter with a color map
    sargs = dict(
        fmt="%.3f",  # This forces 3 decimal places
        title_font_size=20,
        label_font_size=15,
        color="black",  # Ensure labels are visible if background is white
    )
    plotter.add_mesh(plot_grid, show_edges=True, cmap="rainbow", scalar_bar_args=sargs)
    
    if to_probe:
        # 5. Set up a callback for mouse clicks
        plotter.enable_point_picking(callback=partial(probe, grid=plot_grid, plotter=plotter), show_message=True, font_size=12)
        
    print("Instructions: Hover over the mesh and press 'P' to pick a point.")
    plotter.add_axes()
    # This is the most common way to show coordinates in PyVista
    plotter.show_grid(
        xtitle="X-Axis",
        ytitle="Y-Axis",
        ztitle="Z-Axis",
        grid=True,  # Show the actual grid lines
        location="outer",  # Keep labels outside the mesh
        ticks="both",  # Show ticks on both sides
        font_size=14,  # Increase if they are too small
        color="black",  # Change to 'black' if your background is white
        fmt="%.2f",  # Force 2 decimal places (prevents scientific notation)
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=True,
    )
        
    return plotter


def plot_3d_solution_slice(u, function_space_V, to_probe=True):
    # 1. Setup Plotter and Mesh
    plotter = pyvista.Plotter(title="Normalized Fin Probe (0-100%)")

    # 2. Extract the mesh topology and geometry for PyVista
    # The 'v' here refers to your solution uh (or just 'mesh' if you only want the grid)
    topology, cell_types, geometry = plot.vtk_mesh(function_space_V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # 3. Attach your temperature solution (uh) to the grid
    grid.point_data["Data"] = u.x.array.real
    grid.set_active_scalars("Data")

    # plotter.add_mesh(grid, style="wireframe", opacity=0.1, color="white")
    plotter.add_mesh(
        grid, style="wireframe", opacity=0.1, color="white", pickable=False
    )

    # Get physical bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
    state = {"axis": "x", "norm_val": 0.5, "actor": None, "current_slice": None, "widgets": {}}
    update_slice_callback = partial(update_slice_view, plotter, grid, state)
    
    set_axis_x = lambda flag: set_axis("x", flag, state, update_slice_callback)
    set_axis_y = lambda flag: set_axis("y", flag, state, update_slice_callback)
    set_axis_z = lambda flag: set_axis("z", flag, state, update_slice_callback)
    
    # 3. UI Elements
    # Slider from 0 to 1 (Normalized)
    state["widgets"]["slider"] =plotter.add_slider_widget(
        update_slice_callback,
        [0.0, 1.0],
        value=state["norm_val"],
        title="Position",
        pointa=(0.4, 0.1),
        pointb=(0.9, 0.1),
    )

    state["widgets"]["x"] = plotter.add_checkbox_button_widget(
        set_axis_x, value=True, position=(10, 150), size=30, color_on="red"
    )
    state["widgets"]["y"] = plotter.add_checkbox_button_widget(
        set_axis_y, value=False, position=(10, 80), size=30, color_on="green"
    )
    state["widgets"]["z"] = plotter.add_checkbox_button_widget(
        set_axis_z, value=False, position=(10, 10), size=30, color_on="blue"
    )
    
    if to_probe:
        plotter.enable_point_picking(callback=partial(probe, plotter=plotter, grid=grid, state=state), show_message=True)
    update_slice_callback(norm_val=state["norm_val"])
    
    plotter.show_grid(
        xtitle="X-Axis",
        ytitle="Y-Axis",
        ztitle="Z-Axis",
        grid=True,  # Show the actual grid lines
        location="outer",  # Keep labels outside the mesh
        ticks="both",  # Show ticks on both sides
        font_size=14,  # Increase if they are too small
        color="black",  # Change to 'black' if your background is white
        fmt="%.2f",  # Force 2 decimal places (prevents scientific notation)
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=True,
    )
    
    return plotter


def set_axis(axis_name, flag, state, update_callback):
    # Only proceed if the user is turning a button ON
    if flag:
        state["axis"] = axis_name
        
        # Make the checkboxes act like Radio Buttons
        if "widgets" in state:
            for ax in ["x", "y", "z"]:
                if ax != axis_name:
                    # Force the other buttons visually OFF (State 0)
                    state["widgets"][ax].GetRepresentation().SetState(0)
                else:
                    # Force the active button visually ON (State 1)
                    state["widgets"][ax].GetRepresentation().SetState(1)

        # Update the slice view using the current slider value
        update_callback(norm_val=state["norm_val"])
    

def update_slice_view(plotter, grid, state, norm_val=None):
    """Calculates physical position from normalized 0-1 value"""
    if norm_val is not None:
        state["norm_val"] = norm_val

    # 1. Get bounds and calculate physical position from percentile
    b = grid.bounds
    ax_map = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}
    idx_min, idx_max = ax_map[state["axis"]]
    actual_pos = b[idx_min] + (b[idx_max] - b[idx_min]) * state["norm_val"]

    # 2. Update the slider title to show the REAL location
    if "widgets" in state and "slider" in state["widgets"]:
        axis_label = state["axis"].upper()
        slider_rep = state["widgets"]["slider"].GetRepresentation()
        slider_rep.SetTitleText(f"{axis_label} Axis Position: {actual_pos:.3f}")

    # Remove previous slice
    if state["actor"]:
        plotter.remove_actor(state["actor"])

    origin = [0, 0, 0]
    origin[{"x": 0, "y": 1, "z": 2}[state["axis"]]] = actual_pos

    # Create and add the slice
    try:
        sargs = dict(fmt="%.3f", title_font_size=20, label_font_size=15, color="black")
        slc = grid.slice(normal=state["axis"], origin=origin)
        
        if slc.n_points > 0:
            state["current_slice"] = slc
            state["actor"] = plotter.add_mesh(
                slc, cmap="rainbow", name="active_slice", 
                scalar_bar_args=sargs, pickable=True
            )
    except Exception as e:
        print(f"Slice failed at {actual_pos}: {e}")

    
def probe(point, plotter, grid=None, state=None, array_name="Data", label_name="probe"):
    # 1. Resolve the active mesh (Dynamic from state, or static from grid)
    if state is not None:
        mesh = state.get("current_slice")
        if mesh is None:
            return  # Early exit if the slice isn't visible yet
    else:
        mesh = grid

    # 2. Find the closest point index and extract the value
    idx = mesh.find_closest_point(point)
    value = mesh.point_data[array_name][idx]
    
    # NEW: Get the exact 3D coordinate of that node on the mesh
    actual_mesh_coord = mesh.points[idx]

    # 3. Add the label to the 3D plot
    plotter.add_point_labels(
        [actual_mesh_coord], # Pass the actual mesh node as a list
        [f"{value:.3f}°C"], 
        name=label_name, 
        font_size=20,
        point_size=10,
        always_visible=True  # NEW: Forces the text to render on top of the mesh!
    )

    # 4. Print terminal output based on context
    if state is not None:
        axis = state.get('axis', 'UNKNOWN').upper()
        print(f"Slice Pick | Axis: {axis} | {array_name}: {value:.3f}°C")
    else:
        print(f"Clicked Point ID: {idx} | Location: {np.array2string(actual_mesh_coord, formatter={'float_kind': lambda x: f'{x:.2f}'}, separator=', ')} | {array_name}: {value:.3f}°C")
        