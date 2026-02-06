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

# --- Compute Symmetric Part D ---
D_field = 0.5 * (J_field.data + np.transpose(J_field.data, (1, 0, 2, 3)))

# --- Build "Force-Like" Field: F = D · v ---
Fx = np.zeros_like(X)
Fy = np.zeros_like(Y)

for i in range(shape[0]):
    for j in range(shape[1]):
        D = D_field[:, :, i, j]
        v = v_field.data[:, i, j]
        f = D @ v
        Fx[i, j] = f[0]
        Fy[i, j] = f[1]

# --- Particle Trajectory for Animation ---
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
    f_local = D_local @ v_local
    return v_local, f_local

for _ in range(steps):
    px, py = pos
    if px < bounds[0][0] or px > bounds[0][1] or py < bounds[1][0] or py > bounds[1][1]:
        break
    v_local, f_local = sample_local(px, py)
    trajectory.append(pos.copy())
    vel_samples.append(v_local)
    strain_samples.append(f_local)
    pos = pos + v_local * dt

trajectory = np.array(trajectory)
vel_samples = np.array(vel_samples)
strain_samples = np.array(strain_samples)

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Symmetric Tensor Field (Strain) as a Linear Map\nLeft: Principal Strain Directions | Right: $\\mathbf{F} = \\mathbf{D}\\,\\mathbf{v}$", fontsize=16)

# Panel 1: Principal Strain Directions (Eigenvectors of D)
ax1.set_title("Principal Strain Directions (Symmetric Part $\\mathbf{D}$)")
ax1.set_xlim(-2.2, 2.2); ax1.set_ylim(-2.2, 2.2)
ax1.set_aspect('equal')
ax1.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray', density=1.0)

scale = 0.35
for i in range(shape[0]):
    for j in range(shape[1]):
        D = D_field[:, :, i, j]
        evals, evecs = np.linalg.eigh(D)

        # Max stretch (red)
        vec1 = evecs[:, 1]
        val1 = evals[1]
        dx1 = vec1[0] * val1 * scale
        dy1 = vec1[1] * val1 * scale

        # Max compression (blue)
        vec2 = evecs[:, 0]
        val2 = evals[0]
        dx2 = vec2[0] * np.abs(val2) * scale
        dy2 = vec2[1] * np.abs(val2) * scale

        cx, cy = X[i, j], Y[i, j]
        ax1.plot([cx - dx1, cx + dx1], [cy - dy1, cy + dy1], color='red', lw=1.5)
        ax1.plot([cx - dx2, cx + dx2], [cy - dy2, cy + dy2], color='blue', lw=1.5)

# Panel 2: Strain-Only "Force-Like" Field
ax2.set_title("Strain-Only Linear Map $\\mathbf{F} = \\mathbf{D}\\,\\mathbf{v}$")
ax2.set_xlim(-2.2, 2.2); ax2.set_ylim(-2.2, 2.2)
ax2.set_aspect('equal')
ax2.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray', density=1.0)
ax2.quiver(X, Y, Fx, Fy, color='green', pivot='mid', scale=8)

# Animated probe (left panel)
path_line_left, = ax1.plot([], [], 'k--', alpha=0.4)
probe_left, = ax1.plot([], [], 'ro', markersize=7)
q_v_left = ax1.quiver([0], [0], [0], [0], color='black', scale=1, scale_units='xy', angles='xy', width=0.01)
q_f_left = ax1.quiver([0], [0], [0], [0], color='darkgreen', scale=1, scale_units='xy', angles='xy', width=0.01)

# Animated probe (right panel)
path_line, = ax2.plot([], [], 'k--', alpha=0.4)
probe, = ax2.plot([], [], 'ro', markersize=7)
q_v = ax2.quiver([0], [0], [0], [0], color='black', scale=1, scale_units='xy', angles='xy', width=0.01)
q_f = ax2.quiver([0], [0], [0], [0], color='darkgreen', scale=1, scale_units='xy', angles='xy', width=0.01)

v_scale = 0.5
f_scale = 0.8

def update(frame):
    if frame >= len(trajectory):
        return (
            path_line_left,
            probe_left,
            q_v_left,
            q_f_left,
            path_line,
            probe,
            q_v,
            q_f,
        )

    px, py = trajectory[frame]
    v_local = vel_samples[frame]
    f_local = strain_samples[frame]

    path_line_left.set_data(trajectory[:frame + 1, 0], trajectory[:frame + 1, 1])
    probe_left.set_data([px], [py])

    q_v_left.set_offsets([[px, py]])
    q_v_left.set_UVC(v_local[0] * v_scale, v_local[1] * v_scale)

    q_f_left.set_offsets([[px, py]])
    q_f_left.set_UVC(f_local[0] * f_scale, f_local[1] * f_scale)

    path_line.set_data(trajectory[:frame + 1, 0], trajectory[:frame + 1, 1])
    probe.set_data([px], [py])

    q_v.set_offsets([[px, py]])
    q_v.set_UVC(v_local[0] * v_scale, v_local[1] * v_scale)

    q_f.set_offsets([[px, py]])
    q_f.set_UVC(f_local[0] * f_scale, f_local[1] * f_scale)

    return (
        path_line_left,
        probe_left,
        q_v_left,
        q_f_left,
        path_line,
        probe,
        q_v,
        q_f,
    )

ani = FuncAnimation(fig, update, frames=len(trajectory), interval=50, blit=False)

plt.show()
