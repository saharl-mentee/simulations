import ufl
from dolfinx import fem, default_scalar_type, io
from mpi4py import MPI
from pathlib import Path
from dolfinx import io
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import LinearProblem


def plot_3d(u_t, V, ):
    import pyvista
    from dolfinx import plot

    # 1. Create a PyVista plotter
    plotter = pyvista.Plotter()

    # 2. Extract the mesh topology and geometry for PyVista
    # The 'v' here refers to your solution uh (or just 'mesh' if you only want the grid)
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # 3. Attach your temperature solution (uh) to the grid
    grid.point_data["Temperature"] = u_t.x.array.real
    grid.set_active_scalars("Temperature")
    plot_grid = grid.scale([1000, 1000, 1000], inplace=False)

    # 4. Add the mesh to the plotter with a color map
    sargs = dict(
        fmt="%.3f",  # This forces 3 decimal places
        title_font_size=20,
        label_font_size=15,
        color="black",  # Ensure labels are visible if background is white
    )
    plotter.add_mesh(plot_grid, show_edges=True, cmap="rainbow", scalar_bar_args=sargs)

    def callback(point):
        # 1. Find the index of the closest point to where you clicked
        point_id = plot_grid.find_closest_point(point)

        # 2. Get the temperature value at that index
        temp_value = plot_grid.point_data["Temperature"][point_id]

        # 3. Print to terminal (so you can see it in WSL/VS Code)
        print(
            f"Clicked Point ID: {point_id} | Location: {point} | Temperature: {temp_value:.3f}°C"
        )

        # 4. Add a label to the 3D plot at that location
        plotter.add_point_labels(
            point,
            [f"{temp_value:.2f}°C"],
            point_size=10,
            font_size=20,
            name="temp_label",
        )

    # Enable picking
    plotter.enable_point_picking(callback=callback, show_message=True)

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
        font_size=12,  # Increase if they are too small
        color="black",  # Change to 'black' if your background is white
        fmt="%.2f",  # Force 2 decimal places (prevents scientific notation)
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=True,
    )
    plotter.show()

    import pyvista as pv
    import numpy as np
    from dolfinx import plot
    import pyvista

    # 1. Setup Plotter and Mesh
    plotter = pv.Plotter(title="Normalized Fin Probe (0-100%)")

    # 2. Extract the mesh topology and geometry for PyVista
    # The 'v' here refers to your solution uh (or just 'mesh' if you only want the grid)
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # 3. Attach your temperature solution (uh) to the grid
    grid.point_data["Temperature"] = u_t.x.array.real
    grid.set_active_scalars("Temperature")

    # plotter.add_mesh(grid, style="wireframe", opacity=0.1, color="white")
    plotter.add_mesh(
        grid, style="wireframe", opacity=0.1, color="white", pickable=False
    )

    # Get physical bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
    b = grid.bounds
    state = {"axis": "x", "norm_val": 0.0, "actor": None, "current_slice": None}

    def update_view(norm_val=None):
        """Calculates physical position from normalized 0-1 value"""
        if norm_val is not None:
            state["norm_val"] = norm_val

        # Remove previous slice
        if state["actor"]:
            plotter.remove_actor(state["actor"])

        # Map 0.0-1.0 to the physical min/max of the chosen axis
        ax_map = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}
        idx_min, idx_max = ax_map[state["axis"]]

        # Interpolate: physical_pos = min + (max - min) * normalized_val
        actual_pos = b[idx_min] + (b[idx_max] - b[idx_min]) * state["norm_val"]

        origin = [0, 0, 0]
        origin[{"x": 0, "y": 1, "z": 2}[state["axis"]]] = actual_pos

        # Create and add the slice
        try:
            sargs = dict(
                fmt="%.3f",  # This forces 3 decimal places
                title_font_size=20,
                label_font_size=15,
                color="black",  # Ensure labels are visible if background is white
            )

            # slc = grid.slice(normal=state["axis"], origin=origin)
            # # Check if slice is empty before adding to avoid the warning/error
            # if slc.n_points > 0:
            #     state["actor"] = plotter.add_mesh(slc, cmap="rainbow", name="active_slice", scalar_bar_args=sargs)
            slc = grid.slice(normal=state["axis"], origin=origin)
            # if slc.n_points > 0:
            #     state["current_slice"] = slc  # <--- SAVE THIS HERE
            #     state["actor"] = plotter.add_mesh(slc, cmap="rainbow", name="active_slice", scalar_bar_args=sargs)
            if slc.n_points > 0:
                state["current_slice"] = slc
                state["actor"] = plotter.add_mesh(
                    slc,
                    cmap="rainbow",
                    name="active_slice",
                    scalar_bar_args=sargs,
                    pickable=True,  # <--- FORCE THIS
                )
        except Exception as e:
            print(f"Slice failed at {actual_pos}: {e}")

    # 2. Exclusive Selection Functions
    def set_axis_x(flag):
        if flag:
            state["axis"] = "x"
            update_view()

    def set_axis_y(flag):
        if flag:
            state["axis"] = "y"
            update_view()

    def set_axis_z(flag):
        if flag:
            state["axis"] = "z"
            update_view()

    # 3. UI Elements
    # Slider from 0 to 1 (Normalized)
    plotter.add_slider_widget(
        update_view,
        [0.0, 1.0],
        value=0.5,
        title="Normalized Depth",
        pointa=(0.4, 0.1),
        pointb=(0.9, 0.1),
    )

    # Axis buttons (X, Y, Z)
    plotter.add_checkbox_button_widget(
        set_axis_x, value=True, position=(10, 150), size=30, color_on="red"
    )
    plotter.add_checkbox_button_widget(
        set_axis_y, value=False, position=(10, 80), size=30, color_on="green"
    )
    plotter.add_checkbox_button_widget(
        set_axis_z, value=False, position=(10, 10), size=30, color_on="blue"
    )

    # 4. Point Picking (Press 'P')
    def p_callback(point):
        # idx = grid.find_closest_point(point)
        # temp = grid.point_data["Temperature"][idx]
        # plotter.add_point_labels(point, [f"{temp:.3f}°C"], name="probe", font_size=20)
        # print(f"Axis: {state['axis'].upper()} | Normalized Pos: {state['norm_val']:.3f} | Temp: {temp:.3f}°C")
        # Only proceed if we actually have a slice visible
        if state["current_slice"] is None:
            return

        # Use the slice to find the closest point, NOT the full grid
        idx = state["current_slice"].find_closest_point(point)
        temp = state["current_slice"].point_data["Temperature"][idx]

        plotter.add_point_labels(point, [f"{temp:.3f}°C"], name="probe", font_size=20)
        print(f"Slice Pick | Axis: {state['axis'].upper()} | Temp: {temp:.3f}°C")

    plotter.enable_point_picking(callback=p_callback, show_message=True)

    # Initial draw
    update_view(0.5)

    # This adds a 3D "Cage" with X, Y, Z values
    plotter.add_axes()
    plotter.show()
    

def fin_cylindrical(path):
    # 1. Create the communicator (usually MPI.COMM_WORLD for parallel)
    comm = MPI.COMM_WORLD

    # 2. Open the file in Read mode
    # Load the 3D Structure
    path_vol = Path(str(path) + "_volume.xdmf")
    with XDMFFile(comm, path_vol, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    # 1. Determine the facet dimension (2 for a 3D mesh)
    facet_dim = mesh.topology.dim - 1

    # 2. CREATE THE ENTITIES (This fixes your current error)
    mesh.topology.create_entities(facet_dim)

    # 3. CREATE THE CONNECTIVITY (This fixes the "Missing dims 2->3" error)
    mesh.topology.create_connectivity(facet_dim, mesh.topology.dim)

    # Load the 2D Boundary Markers
    # We tell dolfinx to map these triangles onto the 3D mesh we just loaded
    path_surface = Path(str(path) + "_surface.xdmf")
    with XDMFFile(comm, path_surface, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")

    mesh.geometry.x[:, :] *= 0.001

    # 2. Define Function Space
    V = fem.functionspace(mesh, ("Lagrange", 2))

    # 3. Apply Dirichlet BC to the "Base"
    # Assuming 'Base' was the first physical surface you created (ID 1)
    base_id = 1
    dobs = facet_tags.find(base_id)
    bc = fem.dirichletbc(default_scalar_type(120.0), fem.locate_dofs_topological(V, facet_dim, dobs), V)

    # 4. Define Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Parameters
    k = fem.Constant(mesh, default_scalar_type(167.0))  # Thermal conductivity
    h = fem.Constant(mesh, default_scalar_type(5.0))  # Convection coeff
    T_amb = fem.Constant(mesh, default_scalar_type(25.0))

    # Integration measure for surfaces (ds) using the facet_tags from Gmsh

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    # Weak form: conduction + convection on specific surfaces (e.g., ID 2)
    convection_id = 2
    a = k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + h * u * v * ds(convection_id)
    L = h * T_amb * v * ds(convection_id)

    # # for Neuman base BC
    # input_watt = 8
    # q_val = fem.Constant(mesh, default_scalar_type( input_watt / (2 * 437e-6)))
    # L = h * T_amb * v * ds(convection_id) + q_val * v * ds(base_id)

    # # 1. Define the integration form for the base flux
    # # q_base is your Constant (W/m^2), ds(base_id) is the surface measure
    # Q_in_form = fem.form(q_val * ds(base_id))

    # # 2. Assemble the scalar value
    # # This integrates q * dA across the specified surface
    # local_Qin = fem.assemble_scalar(Q_in_form)

    # # 3. Sum across processors (important if running in parallel)
    # total_Qin = mesh.comm.allreduce(local_Qin, op=MPI.SUM)

    # print(f"Total Heat Rate entered (Neumann): {total_Qin:.4f} Watts")

    # 5. Solve
    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="fin_heat_",
    )  # <--- This is the missing piece
    u_t = problem.solve()
    print("Success!")

    # Define the flux expression for convection
    # h and T_amb are your constants, u_t is the solution
    convection_flux_expr = h * (u_t - T_amb) * ds(convection_id)

    # Compute the integral
    # assemble_scalar requires the form to be compiled via fem.form
    Q_conv_form = fem.form(convection_flux_expr)
    Q_conv = mesh.comm.allreduce(fem.assemble_scalar(Q_conv_form), op=MPI.SUM)

    print(f"Total convective heat transfer rate: {Q_conv:.4f} W")

    # Define the facet normal
    n = ufl.FacetNormal(mesh)

    # Define the flux expression: -k * grad(u) . n
    # We use ds(13) for the base
    base_flux_expr = -k * ufl.dot(ufl.grad(u_t), n) * ds(base_id)

    # Compute the integral
    Q_base_form = fem.form(base_flux_expr)
    Q_base = mesh.comm.allreduce(fem.assemble_scalar(Q_base_form), op=MPI.SUM)

    print(f"Total heat rate entering at base: {Q_base:.4f} W")
    print(f"DIFF: {Q_conv+Q_base:.4f} W")
    
    plot_3d(u_t, V)
    
    
def fin_cylindrical_with_base(path):
    # 1. Create the communicator (usually MPI.COMM_WORLD for parallel)
    comm = MPI.COMM_WORLD

    # 2. Open the file in Read mode
    # Load the 3D Structure
    path_vol = Path(str(path) + "_volume.xdmf")
    with XDMFFile(comm, path_vol, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(mesh, name="Grid")

    # 1. Determine the facet dimension (2 for a 3D mesh)
    facet_dim = mesh.topology.dim - 1

    # 2. CREATE THE ENTITIES (This fixes your current error)
    mesh.topology.create_entities(facet_dim)

    # 3. CREATE THE CONNECTIVITY (This fixes the "Missing dims 2->3" error)
    mesh.topology.create_connectivity(facet_dim, mesh.topology.dim)

    # Load the 2D Boundary Markers
    # We tell dolfinx to map these triangles onto the 3D mesh we just loaded
    path_surface = Path(str(path) + "_surface.xdmf")
    with XDMFFile(comm, path_surface, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")

    mesh.geometry.x[:, :] *= 0.001

    # 2. Define Function Space
    V = fem.functionspace(mesh, ("Lagrange", 2))

    # 3. Apply Dirichlet BC to the "Base"
    # Assuming 'Base' was the first physical surface you created (ID 1)
    # base_id = 539
    base_id = 746
    dobs = facet_tags.find(base_id)
    # print(f"Number of facets found for BC: {len(dobs)}")
    bc = fem.dirichletbc(default_scalar_type(120.0), fem.locate_dofs_topological(V, facet_dim, dobs), V)

    # 4. Define Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Parameters
    # Create a Discontinuous Galerkin space (one value per element)
    # base_v_id = 537
    base_v_id = 743
    # fin_v_group_id = 536
    fin_v_group_id = 742
    Q = fem.functionspace(mesh, ("DG", 0))
    k_func = fem.Function(Q)
    # Get the array of values for k_func
    k_values = k_func.x.array

    # Assign values based on the cell tags from Gmsh
    # Volume ID 1 (Base), Volume ID 2 (Fin A), Volume ID 3 (Fin B)
    k_values[cell_tags.find(base_v_id)] = 167.0  # e.g., Aluminum
    k_values[cell_tags.find(fin_v_group_id)] = 167.0  # e.g., Aluminum
    
    h = fem.Constant(mesh, default_scalar_type(5.0))  # Convection coeff
    T_amb = fem.Constant(mesh, default_scalar_type(25.0))

    # Integration measure for surfaces (ds) using the facet_tags from Gmsh

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    # Weak form: conduction + convection on specific surfaces (e.g., ID 2)
    # convection_id = 538
    convection_id = 744
    a = k_func * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + h * u * v * ds(convection_id)
    L = h * T_amb * v * ds(convection_id)

    # # for Neuman base BC
    # input_watt = 8
    # q_val = fem.Constant(mesh, default_scalar_type( input_watt / (2 * 437e-6)))
    # L = h * T_amb * v * ds(convection_id) + q_val * v * ds(base_id)

    # # 1. Define the integration form for the base flux
    # # q_base is your Constant (W/m^2), ds(base_id) is the surface measure
    # Q_in_form = fem.form(q_val * ds(base_id))

    # # 2. Assemble the scalar value
    # # This integrates q * dA across the specified surface
    # local_Qin = fem.assemble_scalar(Q_in_form)

    # # 3. Sum across processors (important if running in parallel)
    # total_Qin = mesh.comm.allreduce(local_Qin, op=MPI.SUM)

    # print(f"Total Heat Rate entered (Neumann): {total_Qin:.4f} Watts")

    # 5. Solve
    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="fin_heat_",
    )  # <--- This is the missing piece
    u_t = problem.solve()
    print("Success!")

    # Define the flux expression for convection
    # h and T_amb are your constants, u_t is the solution
    convection_flux_expr = h * (u_t - T_amb) * ds(convection_id)

    # Compute the integral
    # assemble_scalar requires the form to be compiled via fem.form
    Q_conv_form = fem.form(convection_flux_expr)
    Q_conv = mesh.comm.allreduce(fem.assemble_scalar(Q_conv_form), op=MPI.SUM)

    print(f"Total convective heat transfer rate: {Q_conv:.4f} W")

    # Define the facet normal
    n = ufl.FacetNormal(mesh)

    # Define the flux expression: -k * grad(u) . n
    # We use ds(13) for the base
    base_flux_expr = -k_func * ufl.dot(ufl.grad(u_t), n) * ds(base_id)

    # Compute the integral
    Q_base_form = fem.form(base_flux_expr)
    Q_base = mesh.comm.allreduce(fem.assemble_scalar(Q_base_form), op=MPI.SUM)

    print(f"Total heat rate entering at base: {Q_base:.4f} W")
    print(f"DIFF: {Q_conv+Q_base:.4f} W")
    
    plot_3d(u_t, V)

    


if __name__ == "__main__":
    # path = Path(
    #     "/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/motor_fin/symmetric/MP-05971"
    # )
    # fin_cylindrical(path)
    
    path = Path(
        "/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/fin_assembly/fin_asm2"
        # "/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/fin_assembly/archive/working_no_contact_resistance/fin_asm"
    )
    fin_cylindrical_with_base(path)
