from pathlib import Path

import dolfin
from fenics import *
import matplotlib.pyplot as plt


def main_1d():
    # 1. Create mesh and define function space
    L = 0.1                 # Length of fin (meters)
    mesh = IntervalMesh(50, 0, L) 
    V = FunctionSpace(mesh, 'P', 1) # Linear Lagrange elements

    # 2. Physical Constants
    k = Constant(200.0)     # Thermal conductivity (W/m-K) - Aluminum
    h = Constant(50.0)      # Convection coeff (W/m^2-K)
    Ac = Constant(0.0001)   # Cross-sectional area (m^2)
    P = Constant(0.04)      # Perimeter (m)
    T_amb = Constant(25.0)  # Ambient temperature (C)
    T_base = Constant(100.0)# Temperature at the wall (C)

    # 3. Boundary Conditions
    # Fixed temperature at x = 0
    def base_boundary(x, on_boundary):
        return on_boundary and near(x[0], 0)

    bc = DirichletBC(V, T_base, base_boundary)

    # 4. Variational Problem (The Math)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Left hand side (a) and Right hand side (L)
    a = (k * Ac * dot(grad(u), grad(v)) + h * P * u * v) * dx
    L_func = h * P * T_amb * v * dx

    # 5. Solve
    T = Function(V)
    solve(a == L_func, T, bc)

    # 6. Plotting
    plot(T, title="Temperature Profile of a Fin")
    plt.xlabel("Distance from base (m)")
    plt.ylabel("Temperature (C)")
    plt.grid(True)

    # IMPORTANT: Save the plot because WSL doesn't always show windows
    plt.savefig("fin_plot.png")
    plt.show()
    print("Simulation complete! Check fin_plot.png for results.")


def main_2d():
    # 1. Mesh and Function Space
    L, W = 0.1, 0.01  # 10cm long, 1cm thick
    mesh = RectangleMesh(Point(0, 0), Point(L, W), 50, 10)
    V = FunctionSpace(mesh, 'P', 1)

    # 2. Physical Constants
    k = Constant(200.0)    # Aluminum conductivity
    h = Constant(50.0)     # Convection coefficient
    T_amb = Constant(25.0) # Air temperature
    T_base = Constant(100.0)

    # 3. Define Boundaries
    # Base (x=0)
    def base(x, on_boundary): return on_boundary and near(x[0], 0)

    # Convection surfaces (Top, Bottom, and Tip)
    # We define a 'Measure' to integrate over the boundary
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary_parts.set_all(0)

    # Mark everything EXCEPT the base as convection surfaces
    class ConvectionBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and not near(x[0], 0)

    convection_boundary = ConvectionBoundary()
    convection_boundary.mark(boundary_parts, 1)
    ds = Measure("ds", subdomain_data=boundary_parts)

    # 4. Variational Problem
    u = TrialFunction(V)
    v = TestFunction(V)

    # Interior conduction
    a = k * dot(grad(u), grad(v)) * dx
    # Surface convection (only where marked as 1)
    a += h * u * v * ds(1)
    L = h * T_amb * v * ds(1)

    # 5. Boundary Condition (Base)
    bc = DirichletBC(V, T_base, base)

    # 6. Solve
    T = Function(V)
    solve(a == L, T, bc)
    
    # Define the flux as a vector (since heat has a direction)
    # We use 'Project' to calculate -k * grad(T)
    flux_vector = project(-k * grad(T), VectorFunctionSpace(mesh, 'P', 1))
    
    # If you just want the magnitude (how much heat, regardless of direction)
    flux_magnitude = project(sqrt(dot(flux_vector, flux_vector)), FunctionSpace(mesh, 'P', 1))
    
    # Define the normal vector (pointing out of the surface)
    n = FacetNormal(mesh)

    # Total Heat Transfer (Watts) through the base
    # We integrate the dot product of flux and the normal vector
    total_heat_in = assemble(dot(flux_vector, n) * ds(base)) 
    print(f"Total Heat Flow: {abs(total_heat_in):.4f} Watts")

    # 7. Visualize
    p = plot(T, title="2D Fin Temperature Gradient")
    plt.colorbar(p)
    plt.xlabel("Length (m)")
    plt.ylabel("Thickness (m)")
    plt.savefig("fin_2d_plot.png")
    plt.show()
    print("2D Simulation complete! Open fin_2d_plot.png")


def main_3d_solid():
    import numpy as np
    import ufl
    from dolfinx import fem, mesh, plot
    from dolfinx import io
    from mpi4py import MPI
    from petsc4py import PETSc
    
    import meshio

    def create_mesh(mesh, cell_type):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        return meshio.Mesh(points=mesh.points, 
                        cells={cell_type: cells},
                        cell_data={"name_to_read": [cell_data]})

    
    from dolfinx.io import gmshio    
    # Load directly from the .msh file
    # 3 is the dimension (3D), MPI.COMM_WORLD is for parallel support
    domain, cell_tags, facet_tags = gmshio.read_from_msh("finger_heat_transfer/solid_test/standard_fin.msh", MPI.COMM_WORLD, 0, gdim=3)

    print("Mesh loaded directly from GMSH!")
    
    # 1. Read your GMSH file
    msh = meshio.read(r"finger_heat_transfer/solid_test/standard_fin.msh")

    # 2. Extract Volume (Tetrahedrons) and Surfaces (Triangles)
    vol_mesh = create_mesh(msh, "tetra")
    surf_mesh = create_mesh(msh, "triangle")

    # 3. Write them out for FEniCSx
    meshio.write("mesh_volume.xdmf", vol_mesh)
    meshio.write("mesh_facets.xdmf", surf_mesh)

    # 1. Load the Volume (The Fin itself)
    with io.XDMFFile(MPI.COMM_WORLD, "mesh_volume.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    # 2. Load the Surface Tags (The IDs for Base and Convection)
    # We must use 'read_meshtags' to get the boundary markers
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    with io.XDMFFile(MPI.COMM_WORLD, "mesh_facets.xdmf", "r") as xdmf:
        facet_tags = xdmf.read_meshtags(domain, name="Grid")

    # 3. Define Function Space
    V = fem.FunctionSpace(domain, ("Lagrange", 1))

    # 4. Apply BCs using the tags from GMSH
    # Assuming ID 1 = Base, ID 2 = Outer Surface
    T_base = fem.Constant(domain, 373.15)
    dofs_base = facet_tags.find(1)
    bc = fem.dirichletbc(T_base, fem.locate_dofs_topological(V, 2, dofs_base), V)

    # 4. Variational Form
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k = fem.Constant(domain, 200.0)  # Thermal conductivity (e.g., Aluminum)
    h = fem.Constant(domain, 10.0)   # Convection coeff
    T_amb = fem.Constant(domain, 293.15) # 20°C

    # Surface integration for convection (Surface ID 2)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ct)
    a = k * ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx + h * T * v * ds(2)
    L = h * T_amb * v * ds(2)

    # 5. Solve
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc])
    uh = problem.solve()
    
    import pyvista
    from dolfinx import plot

    # Initialize the pyvista plotter
    plotter = pyvista.Plotter()
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach the temperature data (uh) to the grid
    grid.point_data["Temperature"] = uh.x.array.real
    grid.set_active_scalars("Temperature")

    # View it immediately
    plotter.add_mesh(grid, show_edges=True)
    plotter.show()
    
    
def main_new_attempt():
    import ufl
    from dolfinx import fem, default_scalar_type, io
    from mpi4py import MPI
    from pathlib import Path
    from dolfinx import io
    from mpi4py import MPI
    from dolfinx.io import XDMFFile
    from dolfinx.fem.petsc import LinearProblem
    
    # 1. Create the communicator (usually MPI.COMM_WORLD for parallel)
    comm = MPI.COMM_WORLD
    
    # 2. Open the file in Read mode
    # Load the 3D Structure
    path_vol = Path("/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/solid_test/standard_fin_volume.xdmf")
    with XDMFFile(comm, path_vol, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    
    # 2. CREATE THE ENTITIES (This fixes your current error)
    mesh.topology.create_entities(mesh.topology.dim - 1)

    # 3. CREATE THE CONNECTIVITY (This fixes the "Missing dims 2->3" error)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # Load the 2D Boundary Markers
    # We tell dolfinx to map these triangles onto the 3D mesh we just loaded
    path_surface = Path('/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/solid_test/standard_fin_facets.xdmf')
    with XDMFFile(comm, path_surface, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")

    
    
    # 2. Define Function Space
    V = fem.functionspace(mesh, ("Lagrange", 1))

    # 3. Apply Dirichlet BC to the "Base" 
    # Assuming 'Base' was the first physical surface you created (ID 1)
    base_id = 25
    dobs = facet_tags.find(base_id)
    bc = fem.dirichletbc(default_scalar_type(120.0), fem.locate_dofs_topological(V, 2, dobs), V)

    # 4. Define Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Parameters
    k = fem.Constant(mesh, default_scalar_type(10.0))  # Thermal conductivity
    h = fem.Constant(mesh, default_scalar_type(1.0))  # Convection coeff
    T_amb = fem.Constant(mesh, default_scalar_type(20.0))

    # Integration measure for surfaces (ds) using the facet_tags from Gmsh
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    # Weak form: conduction + convection on specific surfaces (e.g., ID 2)
    convection_id = 26
    a = k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + h * u * v * ds(convection_id)
    L = h * T_amb * v * ds(convection_id)

    # 5. Solve
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"},petsc_options_prefix="fin_heat_") # <--- This is the missing piece
    u_t = problem.solve()
    print('Success!')
    
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

    # 4. Add the mesh to the plotter with a color map
    plotter.add_mesh(grid, show_edges=True, cmap="rainbow")
    
    def callback(point):
        # 1. Find the index of the closest point to where you clicked
        point_id = grid.find_closest_point(point)
        
        # 2. Get the temperature value at that index
        temp_value = grid.point_data["Temperature"][point_id]
        
        # 3. Print to terminal (so you can see it in WSL/VS Code)
        print(f"Clicked Point ID: {point_id} | Location: {point} | Temperature: {temp_value:.2f}°C")
        
        # 4. Add a label to the 3D plot at that location
        plotter.add_point_labels(point, [f"{temp_value:.1f}°C"], 
                                point_size=10, font_size=20, name="temp_label")
        
    # Enable picking
    plotter.enable_point_picking(callback=callback, show_message=True)

    print("Instructions: Hover over the mesh and press 'P' to pick a point.")
    plotter.show()

    
    
    
    import pyvista as pv
    import numpy as np

    # 1. Setup Plotter and Mesh
    plotter = pv.Plotter(title="Normalized Fin Probe (0-100%)")
    
    # 2. Extract the mesh topology and geometry for PyVista
    # The 'v' here refers to your solution uh (or just 'mesh' if you only want the grid)
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # 3. Attach your temperature solution (uh) to the grid
    grid.point_data["Temperature"] = u_t.x.array.real
    grid.set_active_scalars("Temperature")
    
    plotter.add_mesh(grid, style="wireframe", opacity=0.1, color="white")

    # Get physical bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
    b = grid.bounds
    state = {"axis": 'x', "norm_val": 0.0, "actor": None}

    def update_view(norm_val=None):
        """Calculates physical position from normalized 0-1 value"""
        if norm_val is not None:
            state["norm_val"] = norm_val
        
        # Remove previous slice
        if state["actor"]:
            plotter.remove_actor(state["actor"])
        
        # Map 0.0-1.0 to the physical min/max of the chosen axis
        ax_map = {'x': (0, 1), 'y': (2, 3), 'z': (4, 5)}
        idx_min, idx_max = ax_map[state["axis"]]
        
        # Interpolate: physical_pos = min + (max - min) * normalized_val
        actual_pos = b[idx_min] + (b[idx_max] - b[idx_min]) * state["norm_val"]
        
        origin = [0, 0, 0]
        origin[{'x':0, 'y':1, 'z':2}[state["axis"]]] = actual_pos
        
        # Create and add the slice
        try:
            slc = grid.slice(normal=state["axis"], origin=origin)
            # Check if slice is empty before adding to avoid the warning/error
            if slc.n_points > 0:
                state["actor"] = plotter.add_mesh(slc, cmap="inferno", name="active_slice")
        except Exception as e:
            print(f"Slice failed at {actual_pos}: {e}")

    # 2. Exclusive Selection Functions
    def set_axis_x(flag):
        if flag:
            state["axis"] = 'x'
            update_view()

    def set_axis_y(flag):
        if flag:
            state["axis"] = 'y'
            update_view()

    def set_axis_z(flag):
        if flag:
            state["axis"] = 'z'
            update_view()

    # 3. UI Elements
    # Slider from 0 to 1 (Normalized)
    plotter.add_slider_widget(
        update_view, 
        [0.0, 1.0], 
        value=0.5,
        title="Normalized Depth",
        pointa=(0.4, 0.1), 
        pointb=(0.9, 0.1)
    )

    # Axis buttons (X, Y, Z)
    plotter.add_checkbox_button_widget(set_axis_x, value=True, position=(10, 150), size=30, color_on='red')
    plotter.add_checkbox_button_widget(set_axis_y, value=False, position=(10, 80), size=30, color_on='green')
    plotter.add_checkbox_button_widget(set_axis_z, value=False, position=(10, 10), size=30, color_on='blue')

    # 4. Point Picking (Press 'P')
    def p_callback(point):
        idx = grid.find_closest_point(point)
        temp = grid.point_data["Temperature"][idx]
        plotter.add_point_labels(point, [f"{temp:.1f}°C"], name="probe", font_size=20)
        print(f"Axis: {state['axis'].upper()} | Normalized Pos: {state['norm_val']:.2f} | Temp: {temp:.2f}°C")

    plotter.enable_point_picking(callback=p_callback, show_message=True)

    # Initial draw
    update_view(0.5)
    plotter.show()


def main_new_attempt_2_quad(path):
    import ufl
    from dolfinx import fem, default_scalar_type, io
    from mpi4py import MPI
    from pathlib import Path
    from dolfinx import io
    from mpi4py import MPI
    from dolfinx.io import XDMFFile
    from dolfinx.fem.petsc import LinearProblem
    
    # 1. Create the communicator (usually MPI.COMM_WORLD for parallel)
    comm = MPI.COMM_WORLD
    
    # 2. Open the file in Read mode
    # Load the 3D Structure
    path_vol = path / Path("standard_fin_volume.xdmf")
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
    path_surface = path / Path('standard_fin_surface.xdmf')
    with XDMFFile(comm, path_surface, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")

    mesh.geometry.x[:, 1] *= 0.25
    mesh.geometry.x[:, :] *= 0.001
    
    
    # 2. Define Function Space
    # V = fem.functionspace(mesh, ("Lagrange", 1))
    V = fem.functionspace(mesh, ("Serendipity", 2))

    # 3. Apply Dirichlet BC to the "Base" 
    # Assuming 'Base' was the first physical surface you created (ID 1)
    base_id = 13
    dobs = facet_tags.find(base_id)
    bc = fem.dirichletbc(default_scalar_type(120.0), fem.locate_dofs_topological(V, facet_dim, dobs), V)
    
    # 3. Apply Dirichlet BC to the "Base" 
    # Assuming 'Base' was the first physical surface you created (ID 1)
    base_id = 16
    dobs = facet_tags.find(base_id)
    bc_2 = fem.dirichletbc(default_scalar_type(50.0), fem.locate_dofs_topological(V, facet_dim, dobs), V)

    # 4. Define Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Parameters
    k = fem.Constant(mesh, default_scalar_type(100.0))  # Thermal conductivity
    h = fem.Constant(mesh, default_scalar_type(5.0))  # Convection coeff
    T_amb = fem.Constant(mesh, default_scalar_type(20.0))

    # Integration measure for surfaces (ds) using the facet_tags from Gmsh
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    # Weak form: conduction + convection on specific surfaces (e.g., ID 2)
    convection_id = 14
    a = k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + h * u * v * ds(convection_id)
    L = h * T_amb * v * ds(convection_id)

    # 5. Solve
    problem = LinearProblem(a, L, bcs=[bc, bc_2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"},petsc_options_prefix="fin_heat_") # <--- This is the missing piece
    u_t = problem.solve()
    print('Success!')
    
    # Define the flux expression for convection
    # h and T_amb are your constants, u_t is the solution
    convection_flux_expr = h * (u_t - T_amb) * ds(convection_id)

    # Compute the integral
    # assemble_scalar requires the form to be compiled via fem.form
    Q_conv_form = fem.form(convection_flux_expr)
    Q_conv = mesh.comm.allreduce(fem.assemble_scalar(Q_conv_form), op=MPI.SUM)

    print(f"Total convective heat transfer rate: {Q_conv:.4f} W")
    
    n = ufl.FacetNormal(mesh)

    # Define the flux expression: -k * grad(u) . n
    # This represents the heat flow across the boundary
    flux_expr = -k * ufl.dot(ufl.grad(u_t), n)
    
    def calculate_heat_rate(boundary_id):
        # Create the form for the specific surface
        form = fem.form(flux_expr * ds(boundary_id))
        
        # Assemble (integrate) the value on this processor
        local_q = fem.assemble_scalar(form)
        
        # Sum the values from all processors (essential for MPI/parallel)
        total_q = mesh.comm.allreduce(local_q, op=MPI.SUM)
        return total_q

    # Extract rates for your two specific faces
    q_base_13 = calculate_heat_rate(13)
    q_base_16 = calculate_heat_rate(16)

    print(f"Heat rate at Face 13 (T=120): {q_base_13:.4f} W")
    print(f"Heat rate at Face 16 (T=50): {q_base_16:.4f} W")
    print(f'total heat entering: {Q_conv+q_base_16:.4f} W')
    print(f'DIFF: {Q_conv+q_base_16 + q_base_13:.4f} W')
    
    
    
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
        fmt="%.3f",        # This forces 3 decimal places
        title_font_size=20, 
        label_font_size=15,
        color="black"      # Ensure labels are visible if background is white
    )
    plotter.add_mesh(plot_grid, show_edges=True, cmap="rainbow", scalar_bar_args=sargs)
    
    def callback(point):
        # 1. Find the index of the closest point to where you clicked
        point_id = plot_grid.find_closest_point(point)
        
        # 2. Get the temperature value at that index
        temp_value = plot_grid.point_data["Temperature"][point_id]
        
        # 3. Print to terminal (so you can see it in WSL/VS Code)
        print(f"Clicked Point ID: {point_id} | Location: {point} | Temperature: {temp_value:.3f}°C")
        
        # 4. Add a label to the 3D plot at that location
        plotter.add_point_labels(point, [f"{temp_value:.2f}°C"], 
                                point_size=10, font_size=20, name="temp_label")
        
    # Enable picking
    plotter.enable_point_picking(callback=callback, show_message=True)

    print("Instructions: Hover over the mesh and press 'P' to pick a point.")
    plotter.add_axes()
    # This is the most common way to show coordinates in PyVista
    plotter.show_grid(
        xtitle="X-Axis",
        ytitle="Y-Axis", 
        ztitle="Z-Axis",
        grid=True,               # Show the actual grid lines
        location='outer',        # Keep labels outside the mesh
        ticks='both',            # Show ticks on both sides
        font_size=12,            # Increase if they are too small
        color='black',           # Change to 'black' if your background is white
        fmt="%.2f",              # Force 2 decimal places (prevents scientific notation)
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=True
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
    plotter.add_mesh(grid, style="wireframe", opacity=0.1, color="white", pickable=False)
    

    # Get physical bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
    b = grid.bounds
    state = {"axis": 'x', "norm_val": 0.0, "actor": None, "current_slice": None}

    def update_view(norm_val=None):
        """Calculates physical position from normalized 0-1 value"""
        if norm_val is not None:
            state["norm_val"] = norm_val
        
        # Remove previous slice
        if state["actor"]:
            plotter.remove_actor(state["actor"])
        
        # Map 0.0-1.0 to the physical min/max of the chosen axis
        ax_map = {'x': (0, 1), 'y': (2, 3), 'z': (4, 5)}
        idx_min, idx_max = ax_map[state["axis"]]
        
        # Interpolate: physical_pos = min + (max - min) * normalized_val
        actual_pos = b[idx_min] + (b[idx_max] - b[idx_min]) * state["norm_val"]
        
        origin = [0, 0, 0]
        origin[{'x':0, 'y':1, 'z':2}[state["axis"]]] = actual_pos
        
        # Create and add the slice
        try:
            sargs = dict(
                fmt="%.3f",        # This forces 3 decimal places
                title_font_size=20, 
                label_font_size=15,
                color="black"      # Ensure labels are visible if background is white
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
                    pickable=True  # <--- FORCE THIS
                )
        except Exception as e:
            print(f"Slice failed at {actual_pos}: {e}")

    # 2. Exclusive Selection Functions
    def set_axis_x(flag):
        if flag:
            state["axis"] = 'x'
            update_view()

    def set_axis_y(flag):
        if flag:
            state["axis"] = 'y'
            update_view()

    def set_axis_z(flag):
        if flag:
            state["axis"] = 'z'
            update_view()

    # 3. UI Elements
    # Slider from 0 to 1 (Normalized)
    plotter.add_slider_widget(
        update_view, 
        [0.0, 1.0], 
        value=0.5,
        title="Normalized Depth",
        pointa=(0.4, 0.1), 
        pointb=(0.9, 0.1)
    )

    # Axis buttons (X, Y, Z)
    plotter.add_checkbox_button_widget(set_axis_x, value=True, position=(10, 150), size=30, color_on='red')
    plotter.add_checkbox_button_widget(set_axis_y, value=False, position=(10, 80), size=30, color_on='green')
    plotter.add_checkbox_button_widget(set_axis_z, value=False, position=(10, 10), size=30, color_on='blue')

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
    import ufl
    from dolfinx import fem, default_scalar_type, io
    from mpi4py import MPI
    from pathlib import Path
    from dolfinx import io
    from mpi4py import MPI
    from dolfinx.io import XDMFFile
    from dolfinx.fem.petsc import LinearProblem
    
    # 1. Create the communicator (usually MPI.COMM_WORLD for parallel)
    comm = MPI.COMM_WORLD
    
    # 2. Open the file in Read mode
    # Load the 3D Structure
    path_vol = path / Path("fin_volume.xdmf")
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
    path_surface = path / Path('fin_surface.xdmf')
    with XDMFFile(comm, path_surface, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")

    mesh.geometry.x[:, :] *= 0.001
    
    
    # 2. Define Function Space
    V = fem.functionspace(mesh, ("Lagrange",2))

    # 3. Apply Dirichlet BC to the "Base" 
    # Assuming 'Base' was the first physical surface you created (ID 1)
    base_id = 49
    dobs = facet_tags.find(base_id)
    # bc = fem.dirichletbc(default_scalar_type(120.0), fem.locate_dofs_topological(V, facet_dim, dobs), V)

    # 4. Define Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Parameters
    k = fem.Constant(mesh, default_scalar_type(100.0))  # Thermal conductivity
    h = fem.Constant(mesh, default_scalar_type(5.0))  # Convection coeff
    T_amb = fem.Constant(mesh, default_scalar_type(20.0))

    # Integration measure for surfaces (ds) using the facet_tags from Gmsh
    
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    # 1. Define the radial coordinate 'r'
    x = ufl.SpatialCoordinate(mesh)
    r = x[0] 

    # Weak form: conduction + convection on specific surfaces (e.g., ID 2)
    convection_id = 50
    a = k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + h * u * v * ds(convection_id)
    # L = h * T_amb * v * ds(convection_id)
    
    # for Neuman base BC
    q_val = fem.Constant(mesh, default_scalar_type(100 / 2450e-6))
    L = h * T_amb * v * ds(convection_id) + q_val * v * ds(base_id)
    
    # 1. Define the integration form for the base flux
    # q_base is your Constant (W/m^2), ds(base_id) is the surface measure
    Q_in_form = fem.form(q_val * ds(base_id))

    # 2. Assemble the scalar value
    # This integrates q * dA across the specified surface
    local_Qin = fem.assemble_scalar(Q_in_form)

    # 3. Sum across processors (important if running in parallel)
    total_Qin = mesh.comm.allreduce(local_Qin, op=MPI.SUM)

    print(f"Total Heat Rate entered (Neumann): {total_Qin:.4f} Watts")

    # 5. Solve
    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"},petsc_options_prefix="fin_heat_") # <--- This is the missing piece
    u_t = problem.solve()
    print('Success!')
    
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
    print(f'DIFF: {Q_conv+Q_base:.4f} W')
    
    
    
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
        fmt="%.3f",        # This forces 3 decimal places
        title_font_size=20, 
        label_font_size=15,
        color="black"      # Ensure labels are visible if background is white
    )
    plotter.add_mesh(plot_grid, show_edges=True, cmap="rainbow", scalar_bar_args=sargs)
    
    def callback(point):
        # 1. Find the index of the closest point to where you clicked
        point_id = plot_grid.find_closest_point(point)
        
        # 2. Get the temperature value at that index
        temp_value = plot_grid.point_data["Temperature"][point_id]
        
        # 3. Print to terminal (so you can see it in WSL/VS Code)
        print(f"Clicked Point ID: {point_id} | Location: {point} | Temperature: {temp_value:.3f}°C")
        
        # 4. Add a label to the 3D plot at that location
        plotter.add_point_labels(point, [f"{temp_value:.2f}°C"], 
                                point_size=10, font_size=20, name="temp_label")
        
    # Enable picking
    plotter.enable_point_picking(callback=callback, show_message=True)

    print("Instructions: Hover over the mesh and press 'P' to pick a point.")
    plotter.add_axes()
    # This is the most common way to show coordinates in PyVista
    plotter.show_grid(
        xtitle="X-Axis",
        ytitle="Y-Axis", 
        ztitle="Z-Axis",
        grid=True,               # Show the actual grid lines
        location='outer',        # Keep labels outside the mesh
        ticks='both',            # Show ticks on both sides
        font_size=12,            # Increase if they are too small
        color='black',           # Change to 'black' if your background is white
        fmt="%.2f",              # Force 2 decimal places (prevents scientific notation)
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=True
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
    plotter.add_mesh(grid, style="wireframe", opacity=0.1, color="white", pickable=False)
    

    # Get physical bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
    b = grid.bounds
    state = {"axis": 'x', "norm_val": 0.0, "actor": None, "current_slice": None}

    def update_view(norm_val=None):
        """Calculates physical position from normalized 0-1 value"""
        if norm_val is not None:
            state["norm_val"] = norm_val
        
        # Remove previous slice
        if state["actor"]:
            plotter.remove_actor(state["actor"])
        
        # Map 0.0-1.0 to the physical min/max of the chosen axis
        ax_map = {'x': (0, 1), 'y': (2, 3), 'z': (4, 5)}
        idx_min, idx_max = ax_map[state["axis"]]
        
        # Interpolate: physical_pos = min + (max - min) * normalized_val
        actual_pos = b[idx_min] + (b[idx_max] - b[idx_min]) * state["norm_val"]
        
        origin = [0, 0, 0]
        origin[{'x':0, 'y':1, 'z':2}[state["axis"]]] = actual_pos
        
        # Create and add the slice
        try:
            sargs = dict(
                fmt="%.3f",        # This forces 3 decimal places
                title_font_size=20, 
                label_font_size=15,
                color="black"      # Ensure labels are visible if background is white
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
                    pickable=True  # <--- FORCE THIS
                )
        except Exception as e:
            print(f"Slice failed at {actual_pos}: {e}")

    # 2. Exclusive Selection Functions
    def set_axis_x(flag):
        if flag:
            state["axis"] = 'x'
            update_view()

    def set_axis_y(flag):
        if flag:
            state["axis"] = 'y'
            update_view()

    def set_axis_z(flag):
        if flag:
            state["axis"] = 'z'
            update_view()

    # 3. UI Elements
    # Slider from 0 to 1 (Normalized)
    plotter.add_slider_widget(
        update_view, 
        [0.0, 1.0], 
        value=0.5,
        title="Normalized Depth",
        pointa=(0.4, 0.1), 
        pointb=(0.9, 0.1)
    )

    # Axis buttons (X, Y, Z)
    plotter.add_checkbox_button_widget(set_axis_x, value=True, position=(10, 150), size=30, color_on='red')
    plotter.add_checkbox_button_widget(set_axis_y, value=False, position=(10, 80), size=30, color_on='green')
    plotter.add_checkbox_button_widget(set_axis_z, value=False, position=(10, 10), size=30, color_on='blue')

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
    
    

    
if __name__ == '__main__':
    path = Path('/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/solid_test_2/prescribed_end')
    # main_1d()
    # main_2d()
    # main_3d_solid()
    # main_new_attempt()
    # main_new_attempt_2_quad(path)
    fin_cylindrical('/mnt/c/Users/saharl/Documents/V3.2/hand/finger_heat_transfer/solid_circular')