import os

import numpy as np
import pyvista
from dolfinx import fem, plot

from base_plotter import BaseProbeableViewer
from static_structural_plotter import compute_von_mises


class ModeShapeViewer(BaseProbeableViewer):
    """Interactive viewer for natural frequency mode shapes.

    Displays the structure deformed by the currently selected mode shape,
    colored by normalized modal amplitude, with the natural frequency shown
    on screen. Switch between modes with the slider or the Left/Right arrow
    keys.

    Parameters
    ----------
    modes : List[Tuple[float, dolfinx.fem.Function]]
        (natural_frequency_hz, mode_shape) pairs as returned by
        ModalSimulator.run(). Mode shapes are expected to be normalized to a
        maximum displacement magnitude of 1.
    function_space : dolfinx.fem.FunctionSpace
        The vector function space used for the simulation.
    scaling_factor : float, optional
        Physical displacement (in meters) applied at the point of maximum
        modal amplitude. Defaults to 10% of the geometry's bounding-box
        diagonal so every mode is clearly visible.
    """

    def __init__(self, modes, function_space, scaling_factor=None, to_probe=True):
        super().__init__(title="Modal Analysis Viewer")

        if len(modes) == 0:
            raise ValueError("No modes supplied — nothing to display.")
        self.modes = modes
        self.current_mode_index = 0

        # Extract the DOLFINx mesh topology and geometry
        topology, cell_types, geometry = plot.vtk_mesh(function_space)
        self.grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        self.base_points = self.grid.points.copy()

        num_components = function_space.dofmap.index_map_bs
        self.mode_displacements = [
            mode_shape.x.array.reshape((-1, num_components)) for _, mode_shape in modes
        ]

        if scaling_factor is None:
            bounds = np.asarray(self.grid.bounds).reshape(3, 2)
            bbox_diagonal = np.linalg.norm(bounds[:, 1] - bounds[:, 0])
            scaling_factor = 0.1 * bbox_diagonal
        self.scaling_factor = scaling_factor

        # The warped grid is mutated in-place when switching modes; pyvista
        # re-renders automatically because the underlying VTK arrays change.
        self.warped_grid = self.grid.copy()
        self.warped_grid["Mode Amplitude"] = np.zeros(self.grid.n_points)

        self.plotter.add_mesh(
            self.warped_grid,
            scalars="Mode Amplitude",
            cmap="turbo",
            show_edges=True,
            clim=(0.0, 1.0),
            scalar_bar_args={"title": "Normalized Amplitude", "fmt": "%.2f", "vertical": True},
        )
        # Faint wireframe of the original undeformed shape for visual reference
        self.plotter.add_mesh(self.grid, color="white", style="wireframe", opacity=0.15)

        self._slider = None
        if len(self.modes) > 1:
            self._slider = self.plotter.add_slider_widget(
                self._on_slider_change,
                [1, len(self.modes)],
                value=1,
                title="Mode #",
                fmt="%.0f",
                pointa=(0.35, 0.08),
                pointb=(0.86, 0.08),
                style="modern",
            )
        self.plotter.add_key_event("Right", lambda: self._step_mode(+1))
        self.plotter.add_key_event("Left", lambda: self._step_mode(-1))
        print("Instructions: Use the slider or Left/Right arrow keys to switch modes.")

        if to_probe:
            self.enable_probing(self.warped_grid, "Mode Amplitude", value_fmt=".3f")

        self.plotter.add_axes()
        self.plotter.view_isometric()
        self.apply_standard_grid_formatting()
        self.set_mode(0)

    def set_mode(self, mode_index):
        """Displays the given mode: warps the grid by its shape and updates
        the on-screen natural frequency label."""
        mode_index = int(np.clip(mode_index, 0, len(self.modes) - 1))
        self.current_mode_index = mode_index

        frequency, _ = self.modes[mode_index]
        displacements = self.mode_displacements[mode_index]

        self.warped_grid.points = self.base_points + self.scaling_factor * displacements
        self.warped_grid["Mode Amplitude"] = np.linalg.norm(displacements, axis=1)

        self.plotter.add_text(
            f"Mode {mode_index + 1} / {len(self.modes)}   |   f = {frequency:.2f} Hz",
            name="mode_label",
            position="upper_edge",
            font_size=14,
        )
        self.plotter.render()

    def _on_slider_change(self, value):
        mode_index = int(round(value)) - 1
        if mode_index != self.current_mode_index:
            self.set_mode(mode_index)

    def _step_mode(self, step):
        mode_index = (self.current_mode_index + step) % len(self.modes)
        if self._slider is not None:
            # Keep the slider in sync; its callback triggers set_mode.
            self._slider.GetRepresentation().SetValue(mode_index + 1)
        self.set_mode(mode_index)


class ModeAnimationViewer(BaseProbeableViewer):
    """Continuous animation ("small video") of a single vibration mode.

    Oscillates the geometry through a full vibration cycle,
    u(t) = A * sin(phase) * phi  with phase sweeping [0, 2*pi), so the loop is
    seamless. The animation phase is normalized — it shows the mode's *shape*
    cycling, not real time (the true frequency, hundreds of Hz, is impossible
    to watch in real time; it is printed on screen instead).

    Call `.show()` to play. With a display available it plays continuously via
    a repeating timer; headless (or when `interactive=False`) it writes an
    animated GIF/MP4 of one period.

    Parameters
    ----------
    modes : List[Tuple[float, dolfinx.fem.Function]]
        (natural_frequency_hz, mode_shape) pairs from ModalSimulator.run().
    function_space : dolfinx.fem.FunctionSpace
        The vector function space used for the simulation.
    mode_index : int, optional
        Which mode to animate (0-based), by default 0.
    scaling_factor : float, optional
        Peak physical displacement (m) at the point of maximum amplitude.
        Defaults to 10% of the bounding-box diagonal.
    n_frames : int, optional
        Frames per full vibration period, by default 48.
    interactive : bool, optional
        Force off-screen GIF/MP4 export when False, by default True.
    """

    def __init__(self, modes, function_space, mode_index=0, scaling_factor=None,
                 n_frames=48, interactive=True):
        if len(modes) == 0:
            raise ValueError("No modes supplied — nothing to animate.")
        mode_index = int(np.clip(mode_index, 0, len(modes) - 1))

        has_display = "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ
        self.use_offscreen = not interactive or not has_display
        super().__init__(title="Mode Animation Viewer", off_screen=self.use_offscreen)

        self.frequency, mode_shape = modes[mode_index]
        self.mode_index = mode_index
        self.num_modes = len(modes)
        self.n_frames = n_frames
        # One full period, endpoint excluded so the last frame flows back into
        # the first without a duplicate — keeps the loop seamless.
        self.phases = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)

        topology, cell_types, geometry = plot.vtk_mesh(function_space)
        self.grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        self.base_points = self.grid.points.copy()

        num_components = function_space.dofmap.index_map_bs
        self.displacement = mode_shape.x.array.reshape((-1, num_components))
        self.amplitude = np.linalg.norm(self.displacement, axis=1)

        if scaling_factor is None:
            bounds = np.asarray(self.grid.bounds).reshape(3, 2)
            bbox_diagonal = np.linalg.norm(bounds[:, 1] - bounds[:, 0])
            scaling_factor = 0.1 * bbox_diagonal
        self.scaling_factor = scaling_factor

        self.warped_grid = self.grid.copy()
        self.warped_grid["Mode Amplitude"] = np.zeros(self.grid.n_points)
        self.plotter.add_mesh(
            self.warped_grid,
            scalars="Mode Amplitude",
            cmap="turbo",
            show_edges=True,
            clim=(0.0, 1.0),
            scalar_bar_args={"title": "Normalized Amplitude", "fmt": "%.2f", "vertical": True},
        )
        # Faint wireframe of the undeformed shape for a fixed visual reference.
        self.plotter.add_mesh(self.grid, color="white", style="wireframe", opacity=0.15)
        self.plotter.add_axes()
        self.plotter.view_isometric()
        self.apply_standard_grid_formatting()
        self._set_phase(0.0)

    def _set_phase(self, phase):
        """Warps the grid to the given phase of the vibration cycle."""
        modulation = np.sin(phase)
        self.warped_grid.points = self.base_points + self.scaling_factor * modulation * self.displacement
        # Instantaneous amplitude, so the color pulses with the motion.
        self.warped_grid["Mode Amplitude"] = np.abs(modulation) * self.amplitude
        self.plotter.add_text(
            f"Mode {self.mode_index + 1} / {self.num_modes}   |   f = {self.frequency:.2f} Hz",
            name="mode_label",
            position="upper_edge",
            font_size=14,
        )

    def _on_tick(self, step):
        self._set_phase(self.phases[step % self.n_frames])
        self.plotter.render()

    def show(self, save_path="mode_animation.gif", fps=20):
        """Plays the animation, or writes a GIF/MP4 when running headless.

        The file format follows `save_path`'s extension ('.gif' or '.mp4').
        """
        if self.use_offscreen:
            if save_path.lower().endswith(".mp4"):
                self.plotter.open_movie(save_path, framerate=fps)
            else:
                self.plotter.open_gif(save_path, fps=fps)
            for phase in self.phases:
                self._set_phase(phase)
                self.plotter.write_frame()
            self.plotter.close()
            print(f"✅ [Animation] Saved vibration loop to:\n   👉 {os.path.abspath(save_path)}")
        else:
            # A large step budget keeps the timer firing effectively forever;
            # the callback wraps the phase index, so it loops continuously.
            self.plotter.add_timer_event(
                max_steps=1_000_000, duration=int(1000 / fps), callback=self._on_tick
            )
            print("Instructions: the mode animates continuously; close the window to stop.")
            self.plotter.show()


class ModalStressViewer(BaseProbeableViewer):
    """Static plot of the (relative) Von Mises stress of a single mode shape.

    Mode shapes are normalized to unit maximum displacement, so the stress
    *magnitude* is arbitrary (a free-vibration mode has no external load, only
    a shape). The stress *distribution*, however, is meaningful: it shows where
    a given mode concentrates strain energy — the likely crack/fatigue sites if
    the structure is excited near that natural frequency.

    Parameters
    ----------
    modes : List[Tuple[float, dolfinx.fem.Function]]
        (natural_frequency_hz, mode_shape) pairs from ModalSimulator.run().
    function_space : dolfinx.fem.FunctionSpace
        The vector function space used for the simulation.
    E, nu : float
        Young's modulus and Poisson's ratio for the constitutive stress model.
    mode_index : int, optional
        Which mode to visualize (0-based), by default 0.
    scaling_factor : float, optional
        Peak displacement (m) used to warp the geometry so the stress is drawn
        on the deformed shape. Defaults to 10% of the bounding-box diagonal.
    interactive : bool, optional
        Force off-screen screenshot export when False, by default True.
    """

    def __init__(self, modes, function_space, E, nu, mode_index=0, scaling_factor=None,
                 to_probe=True, interactive=True):
        if len(modes) == 0:
            raise ValueError("No modes supplied — nothing to display.")
        mode_index = int(np.clip(mode_index, 0, len(modes) - 1))

        has_display = "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ
        self.use_offscreen = not interactive or not has_display
        super().__init__(title="Modal Stress Viewer", off_screen=self.use_offscreen)

        frequency, mode_shape = modes[mode_index]

        # Von Mises on a continuous CG1 scalar space (smooth for plotting).
        von_mises = compute_von_mises(mode_shape, E, nu)
        mesh = mode_shape.function_space.mesh
        gdim = mesh.geometry.dim

        # Build the plot grid from the CG1 stress space, and pull the mode's
        # displacement onto the SAME CG1 nodes so point counts line up for the
        # warp (the mode shape may live on a higher-order vector space).
        topology, cell_types, geometry = plot.vtk_mesh(von_mises.function_space)
        self.grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        self.grid["VonMises_Stress"] = von_mises.x.array

        displacement_cg1 = fem.Function(fem.functionspace(mesh, ("Lagrange", 1, (gdim,))))
        displacement_cg1.interpolate(mode_shape)
        displacements = displacement_cg1.x.array.reshape((-1, gdim))

        if scaling_factor is None:
            bounds = np.asarray(self.grid.bounds).reshape(3, 2)
            bbox_diagonal = np.linalg.norm(bounds[:, 1] - bounds[:, 0])
            scaling_factor = 0.1 * bbox_diagonal
        self.warped_grid = self.grid.copy()
        self.warped_grid.points = self.grid.points + scaling_factor * displacements

        self.plotter.add_text(
            f"Modal Stress (relative) — Mode {mode_index + 1} | f = {frequency:.2f} Hz",
            font_size=12,
        )
        self.plotter.add_mesh(
            self.warped_grid,
            scalars="VonMises_Stress",
            cmap="jet",
            show_edges=True,
            edge_color="black",
            line_width=0.5,
            scalar_bar_args={"title": "Von Mises (relative)", "vertical": True},
        )
        # Faint wireframe of the undeformed shape for visual reference.
        self.plotter.add_mesh(self.grid, color="white", style="wireframe", opacity=0.15)

        if to_probe and not self.use_offscreen:
            self.enable_probing(self.warped_grid, "VonMises_Stress")

        self.plotter.add_axes()
        self.plotter.view_isometric()
        self.apply_standard_grid_formatting()

    def show(self, output_image="modal_stress.png"):
        """Renders interactively, or saves a screenshot when running headless."""
        if self.use_offscreen:
            self.plotter.screenshot(output_image)
            self.plotter.close()
            print(f"✅ [Modal Stress] Headless render saved to:\n   👉 {os.path.abspath(output_image)}")
        else:
            self.plotter.show()
