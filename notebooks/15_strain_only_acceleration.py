import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

from tensor_vis.fluids.velocity_field import VelocityField
from tensor_vis.kinematics.jacobian import calculate_jacobian
from tensor_vis.core.interp import bilinear_interpolate

# --- Setup Data ---
bounds = ((-2, 2), (-2, 2))
shape = (15, 15)
v_field = VelocityField(shape, bounds)

x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')

# Taylor-Green vortex (2D)
Vx = np.sin(X) * np.cos(Y)
Vy = -np.cos(X) * np.sin(Y)
v_field.set_data(np.stack([Vx, Vy], axis=0))

J_field = calculate_jacobian(v_field)

# --- Compute Strain-Only Acceleration: a_strain = D · v ---
A_strain_x = np.zeros_like(X)
A_strain_y = np.zeros_like(Y)

for i in range(shape[0]):
    for j in range(shape[1]):
        v_vec = v_field.data[:, i, j]
        L = J_field.data[:, :, i, j]
        D = 0.5 * (L + L.T)
        a_strain = D @ v_vec
        A_strain_x[i, j] = a_strain[0]
        A_strain_y[i, j] = a_strain[1]

# --- Particle Trajectory (for animation) ---
dt = 0.03
steps = 350
trajectory = []
vel_samples = []
strain_samples = []

pos = np.array([-1.2, -0.8])

def sample_local(px, py):
    v_local = bilinear_interpolate(v_field.data, bounds, (px, py))
    J_local = bilinear_interpolate(J_field.data, bounds, (px, py))
    D_local = 0.5 * (J_local + J_local.T)
    a_local = D_local @ v_local
    return v_local, a_local

for _ in range(steps):
    px, py = pos
    if px < bounds[0][0] or px > bounds[0][1] or py < bounds[1][0] or py > bounds[1][1]:
        break
    v_local, a_local = sample_local(px, py)
    trajectory.append(pos.copy())
    vel_samples.append(v_local)
    strain_samples.append(a_local)
    pos = pos + v_local * dt

trajectory = np.array(trajectory)
vel_samples = np.array(vel_samples)
strain_samples = np.array(strain_samples)

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Strain-Only Action on Velocity: $\\mathbf{a}_{strain} = \\mathbf{D} \\cdot \\mathbf{v}$\n(antisymmetric part removed)", fontsize=16)

# 1. Velocity Field
ax1.set_title("Velocity Field $\\mathbf{v}$")
ax1.set_xlim(-2.2, 2.2); ax1.set_ylim(-2.2, 2.2)
ax1.set_aspect('equal')
ax1.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray', density=1.2)
ax1.quiver(X, Y, Vx, Vy, color='black', pivot='mid', scale=20)
path_line, = ax1.plot([], [], 'k--', alpha=0.4)
probe_v, = ax1.plot([], [], 'ro', markersize=7)
q_v = ax1.quiver([0], [0], [0], [0], color='red', scale=1, scale_units='xy', angles='xy', width=0.01)

# 2. Strain-Only Acceleration
ax2.set_title("Strain Contribution $\\mathbf{a}_{strain} = \\mathbf{D} \\cdot \\mathbf{v}$")
ax2.set_xlim(-2.2, 2.2); ax2.set_ylim(-2.2, 2.2)
ax2.set_aspect('equal')
ax2.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray', density=1.2)
ax2.quiver(X, Y, A_strain_x, A_strain_y, color='green', pivot='mid', scale=10)
probe_a, = ax2.plot([], [], 'ro', markersize=7)
q_a = ax2.quiver([0], [0], [0], [0], color='darkgreen', scale=1, scale_units='xy', angles='xy', width=0.01)

v_scale = 0.5
a_scale = 0.8

def update(frame):
    if frame >= len(trajectory):
        return path_line, probe_v, q_v, probe_a, q_a

    px, py = trajectory[frame]
    v_local = vel_samples[frame]
    a_local = strain_samples[frame]

    path_line.set_data(trajectory[:frame + 1, 0], trajectory[:frame + 1, 1])
    probe_v.set_data([px], [py])
    probe_a.set_data([px], [py])

    q_v.set_offsets([[px, py]])
    q_v.set_UVC(v_local[0] * v_scale, v_local[1] * v_scale)

    q_a.set_offsets([[px, py]])
    q_a.set_UVC(a_local[0] * a_scale, a_local[1] * a_scale)

    return path_line, probe_v, q_v, probe_a, q_a

ani = FuncAnimation(fig, update, frames=len(trajectory), interval=50, blit=False)
plt.show()
