import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

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
shape = (20, 20)
v_field = VelocityField(shape, bounds)

x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')

# Define Field: v = (sin(y), cos(x))
Vx = np.sin(Y)
Vy = np.cos(X)
v_field.set_data(np.stack([Vx, Vy], axis=0))

J_field = calculate_jacobian(v_field)

# --- Visualization ---
fig = plt.figure(figsize=(14, 8))
fig.suptitle("Basis Transformation: $\\nabla \\mathbf{v}$ maps Basis to Derivatives\n$\\mathbf{J} \\cdot \\hat{i} = \\partial_x \\mathbf{v}$  |  $\\mathbf{J} \\cdot \\hat{j} = \\partial_y \\mathbf{v}$", fontsize=16)

gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# Left: Global Field
ax_global = fig.add_subplot(gs[0])
ax_global.set_title("Global Field & Probe")
ax_global.set_xlim(-2, 2); ax_global.set_ylim(-2, 2)
ax_global.set_aspect('equal')
ax_global.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray')

# Probe
probe_scatter = ax_global.scatter([0], [0], color='black', s=100, zorder=5)
# Path
theta_path = np.linspace(0, 2*np.pi, 200)
path_r = 1.0
path_x = path_r * np.cos(theta_path)
path_y = path_r * np.sin(theta_path)
ax_global.plot(path_x, path_y, 'k--', alpha=0.5)

# Right: Basis View
ax_local = fig.add_subplot(gs[1])
ax_local.set_title("Local Basis Transformation\nInput (Dashed) $\\to$ Output (Solid)")
ax_local.set_xlim(-1.5, 1.5); ax_local.set_ylim(-1.5, 1.5)
ax_local.set_aspect('equal')
ax_local.grid(True)
ax_local.axhline(0, color='k', linewidth=0.5)
ax_local.axvline(0, color='k', linewidth=0.5)

# Standard Basis (Input) - Dashed
# i_hat = (1, 0)
# j_hat = (0, 1)
ax_local.arrow(0, 0, 1, 0, color='red', width=0.015, ls='--', alpha=0.5, label='$\\hat{i}$ (Input)')
ax_local.arrow(0, 0, 0, 1, color='green', width=0.015, ls='--', alpha=0.5, label='$\\hat{j}$ (Input)')

# Transformed Basis (Output) - Solid
# These will be updated
q_out_i = ax_local.quiver([0], [0], [1], [0], color='red', scale=1, scale_units='xy', angles='xy', width=0.02, label='$\\partial_x \\mathbf{v}$ (Output)')
q_out_j = ax_local.quiver([0], [0], [0], [1], color='green', scale=1, scale_units='xy', angles='xy', width=0.02, label='$\\partial_y \\mathbf{v}$ (Output)')

ax_local.legend(loc='upper right')

def get_interpolated_J(px, py):
    # Bilinear interpolation for smoother probe motion
    return bilinear_interpolate(J_field.data, bounds, (px, py))

def update(frame):
    px = path_x[frame % len(path_x)]
    py = path_y[frame % len(path_y)]
    probe_scatter.set_offsets(np.c_[px, py])
    
    J = get_interpolated_J(px, py)
    
    # Transform Basis
    # i_new = J * [1, 0]^T = Column 0 of J
    # j_new = J * [0, 1]^T = Column 1 of J
    
    i_out = J[:, 0]
    j_out = J[:, 1]
    
    q_out_i.set_UVC(i_out[0], i_out[1])
    q_out_j.set_UVC(j_out[0], j_out[1])
    
    return probe_scatter, q_out_i, q_out_j

ani = FuncAnimation(fig, update, frames=len(path_x), interval=50, blit=False)
plt.show()
