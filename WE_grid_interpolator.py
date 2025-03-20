import sys
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
import pyvista as pv
from pyvistaqt import QtInteractor

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSlider
)

# -----------------------------
# Synthetic Data Generation
# -----------------------------
np.random.seed(0)

# Parameters
num_grid_points = 64  # total grid points
flange_radius = 100.0  # flange radius (mm)
grid_region_radius = 80.0  # grid points are distributed within 80 mm
num_time_steps = 3  # number of time steps (t = 0,1,2 s)
time_steps = np.linspace(0, 2, num=num_time_steps)  # e.g. 0, 1, 2 s

# --- Grid Points ---
# Generate 64 grid points randomly inside a circle of radius 80 mm.
r_grid = grid_region_radius * np.sqrt(np.random.rand(num_grid_points))
theta_grid = 2 * np.pi * np.random.rand(num_grid_points)
grid_positions_x = r_grid * np.cos(theta_grid)
grid_positions_y = r_grid * np.sin(theta_grid)
grid_positions = np.column_stack((grid_positions_x, grid_positions_y, np.zeros(num_grid_points)))

# --- Grid Displacements ---
# For each time step, compute displacements at each grid point using arbitrary spatial functions.
grid_disp_data = []
for t in time_steps:
    disp = {}
    for i in range(num_grid_points):
        x_val = grid_positions_x[i]
        y_val = grid_positions_y[i]
        # Example displacement functions (in mm)
        dx = 0.1 * x_val + 0.5 * np.sin(t + x_val / 20)
        dy = 0.1 * y_val + 0.5 * np.cos(t + y_val / 20)
        dz = 0.2 * t + 0.05 * x_val  # variation with time and x position
        disp[f'Grid_{i + 1}_x'] = dx
        disp[f'Grid_{i + 1}_y'] = dy
        disp[f'Grid_{i + 1}_z'] = dz
    disp['Time'] = t
    grid_disp_data.append(disp)
grid_disp_df = pd.DataFrame(grid_disp_data)

# --- Mesh Nodes ---
# Create a structured grid of points in the square [-100, 100] mm and keep only those inside a circle of radius 100 mm.
x_coords = np.linspace(-flange_radius, flange_radius, 31)
y_coords = np.linspace(-flange_radius, flange_radius, 31)
node_list = []
for x in x_coords:
    for y in y_coords:
        if x ** 2 + y ** 2 <= flange_radius ** 2:
            node_list.append((x, y, 0.0))
nodes = np.array(node_list)  # shape (num_nodes, 3)
num_nodes = nodes.shape[0]

# --- Interpolation ---
# Precompute interpolated displacement fields for each time step.
disp_dict = {}  # key: time index (0,1,2), value: dict with keys 'disp_x', 'disp_y', 'disp_z'
for idx, row in grid_disp_df.iterrows():
    grid_disp_x = np.array([row[f'Grid_{i + 1}_x'] for i in range(num_grid_points)])
    grid_disp_y = np.array([row[f'Grid_{i + 1}_y'] for i in range(num_grid_points)])
    grid_disp_z = np.array([row[f'Grid_{i + 1}_z'] for i in range(num_grid_points)])

    # Create RBF interpolators for each displacement component.
    rbf_x = Rbf(grid_positions_x, grid_positions_y, grid_disp_x, function='multiquadric')
    rbf_y = Rbf(grid_positions_x, grid_positions_y, grid_disp_y, function='multiquadric')
    rbf_z = Rbf(grid_positions_x, grid_positions_y, grid_disp_z, function='multiquadric')

    node_x = nodes[:, 0]
    node_y = nodes[:, 1]
    interp_disp_x = rbf_x(node_x, node_y)
    interp_disp_y = rbf_y(node_x, node_y)
    interp_disp_z = rbf_z(node_x, node_y)

    disp_dict[idx] = {
        'disp_x': interp_disp_x,
        'disp_y': interp_disp_y,
        'disp_z': interp_disp_z,
        'time': row['Time']
    }


# -----------------------------
# PyQt5 GUI with PyVista Visualization
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flange Displacement Visualization")
        self.setGeometry(100, 100, 1200, 800)

        # Main widget and layout
        self.frame = QWidget()
        self.setCentralWidget(self.frame)
        layout = QVBoxLayout()
        self.frame.setLayout(layout)

        # PyVista interactor widget
        self.plotter = QtInteractor(self.frame)
        layout.addWidget(self.plotter.interactor)

        # Top Controls: Mode and Time
        top_controls = QHBoxLayout()
        layout.addLayout(top_controls)

        top_controls.addWidget(QLabel("Deformation Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["X Deformation", "Y Deformation", "Z Deformation", "Total Deformation"])
        top_controls.addWidget(self.mode_combo)

        top_controls.addWidget(QLabel("Time Step:"))
        self.time_slider = QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(num_time_steps - 1)
        self.time_slider.setTickInterval(1)
        self.time_slider.setTickPosition(QSlider.TicksBelow)
        top_controls.addWidget(self.time_slider)
        self.time_label = QLabel("0 s")
        top_controls.addWidget(self.time_label)

        self.mode_combo.currentIndexChanged.connect(self.update_plot)
        self.time_slider.valueChanged.connect(self.on_time_changed)

        # Additional Controls: Grid toggle, Mesh Point Size, and Animation Buttons
        additional_controls = QHBoxLayout()
        layout.addLayout(additional_controls)

        # Grid points toggle
        self.grid_checkbox = QtWidgets.QCheckBox("Show Grid Points")
        self.grid_checkbox.setChecked(True)
        additional_controls.addWidget(self.grid_checkbox)
        self.grid_checkbox.stateChanged.connect(self.update_plot)

        # Mesh point size spinbox
        additional_controls.addWidget(QLabel("Mesh Point Size:"))
        self.point_size_spin = QtWidgets.QSpinBox()
        self.point_size_spin.setRange(1, 50)
        self.point_size_spin.setValue(10)
        additional_controls.addWidget(self.point_size_spin)
        self.point_size_spin.valueChanged.connect(self.update_plot)

        # Animation buttons: Play, Pause, Stop
        self.play_button = QtWidgets.QPushButton("Play")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.stop_button = QtWidgets.QPushButton("Stop")
        additional_controls.addWidget(self.play_button)
        additional_controls.addWidget(self.pause_button)
        additional_controls.addWidget(self.stop_button)

        self.play_button.clicked.connect(self.play_animation)
        self.pause_button.clicked.connect(self.pause_animation)
        self.stop_button.clicked.connect(self.stop_animation)

        # Timer for animation
        self.timer = QtCore.QTimer()
        self.timer.setInterval(500)  # update every 500 ms
        self.timer.timeout.connect(self.animate)

        # Initial plot update
        self.update_plot()

    def on_time_changed(self, value):
        t_val = disp_dict[value]['time']
        self.time_label.setText(f"{t_val:.2f} s")
        self.update_plot()

    def update_plot(self):
        self.plotter.clear()
        time_idx = self.time_slider.value()
        mode = self.mode_combo.currentText()
        disp_data = disp_dict[time_idx]
        disp_x = disp_data['disp_x']
        disp_y = disp_data['disp_y']
        disp_z = disp_data['disp_z']

        orig = nodes.copy()  # original node positions

        if mode == "X Deformation":
            new_points = np.column_stack((orig[:, 0] + disp_x, orig[:, 1], orig[:, 2]))
            scalars = disp_x
            scalar_label = "X Disp (mm)"
        elif mode == "Y Deformation":
            new_points = np.column_stack((orig[:, 0], orig[:, 1] + disp_y, orig[:, 2]))
            scalars = disp_y
            scalar_label = "Y Disp (mm)"
        elif mode == "Z Deformation":
            new_points = np.column_stack((orig[:, 0], orig[:, 1], orig[:, 2] + disp_z))
            scalars = disp_z
            scalar_label = "Z Disp (mm)"
        elif mode == "Total Deformation":
            new_points = np.column_stack((orig[:, 0] + disp_x, orig[:, 1] + disp_y, orig[:, 2] + disp_z))
            scalars = np.sqrt(disp_x ** 2 + disp_y ** 2 + disp_z ** 2)
            scalar_label = "Total Disp (mm)"
        else:
            new_points = orig.copy()
            scalars = np.zeros_like(disp_x)
            scalar_label = "No Data"

        # Create PolyData for mesh nodes and assign scalars using point_data.
        mesh = pv.PolyData(new_points)
        mesh.point_data[scalar_label] = scalars

        # Get mesh point size from spinbox.
        point_size = self.point_size_spin.value()
        self.plotter.add_mesh(mesh, scalars=scalar_label,
                              render_points_as_spheres=True,
                              point_size=point_size,
                              cmap="jet",
                              clim=[np.min(scalars), np.max(scalars)])

        # Optionally add grid points if checkbox is checked.
        if self.grid_checkbox.isChecked():
            grid_pd = pv.PolyData(grid_positions)
            self.plotter.add_mesh(grid_pd, color="red",
                                  render_points_as_spheres=True,
                                  point_size=20)

        self.plotter.add_scalar_bar(title=scalar_label, n_labels=5)
        self.plotter.reset_camera()
        self.plotter.render()

    def animate(self):
        current = self.time_slider.value()
        if current < self.time_slider.maximum():
            self.time_slider.setValue(current + 1)
        else:
            self.time_slider.setValue(0)

    def play_animation(self):
        self.timer.start()

    def pause_animation(self):
        self.timer.stop()

    def stop_animation(self):
        self.timer.stop()
        self.time_slider.setValue(0)
        self.update_plot()


# -----------------------------
# Main Application Entry Point
# -----------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
