import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

from tensor_vis.fluids.velocity_field import VelocityField
from tensor_vis.kinematics.jacobian import calculate_jacobian
from tensor_vis.core.interp import bilinear_interpolate

# --- 1. Define Non-Linear Velocity Field ---
# We need a field with curvature (non-zero second derivatives) 
# so the Linear Approximation has a visible error.
# v = (sin(x) cos(y), -cos(x) sin(y)) [Taylor-Green Vortex]
bounds = ((-np.pi, np.pi), (-np.pi, np.pi))
shape = (40, 40)
v_field = VelocityField(shape, bounds)

x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')

Vx = np.sin(X) * np.cos(Y)
Vy = -np.cos(X) * np.sin(Y)
v_field.set_data(np.stack([Vx, Vy], axis=0))

J_field = calculate_jacobian(v_field)

# --- 2. Trajectory Setup ---
dt = 0.05
steps = 200
trajectory = []
curr_pos = np.array([0.5, 0.5]) # Start inside a vortex
trajectory.append(curr_pos.copy())

for _ in range(steps):
    # Bilinear interp for velocity (simplified nearest for demo speed)
    dx = (bounds[0][1] - bounds[0][0]) / (shape[0] - 1)
    dy = (bounds[1][1] - bounds[1][0]) / (shape[1] - 1)
    idx_x = int((curr_pos[0] - bounds[0][0]) / dx)
    idx_y = int((curr_pos[1] - bounds[1][0]) / dy)
    
    # Boundary check
    idx_x = max(0, min(shape[0]-1, idx_x))
    idx_y = max(0, min(shape[1]-1, idx_y))
    
    v_at_pos = v_field.data[:, idx_x, idx_y]
    curr_pos = curr_pos + v_at_pos * dt
    trajectory.append(curr_pos.copy())
    
trajectory = np.array(trajectory)

# --- 3. Local Grid Setup (window around particle) ---
local_window_size = 1.0 # Radius of view
local_res = 11
lx = np.linspace(-local_window_size/2, local_window_size/2, local_res)
ly = np.linspace(-local_window_size/2, local_window_size/2, local_res)
LX, LY = np.meshgrid(lx, ly, indexing='ij')
local_points = np.stack([LX.ravel(), LY.ravel()], axis=1) # (N, 2) Relative coords

# --- 4. Visualization ---
fig = plt.figure(figsize=(16, 8))
fig.suptitle(r"Local Reconstruction: Taylor Expansion $\mathbf{v}(\mathbf{x}) \approx \mathbf{v}_0 + \mathbf{J} \cdot \Delta \mathbf{r}$", fontsize=16)

# Layout
gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1])

# Left: Global Map
ax_global = fig.add_subplot(gs[0])
ax_global.set_title("1. Global View (Trajectory)")
ax_global.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray')
line_traj, = ax_global.plot([], [], 'b--', alpha=0.6)
pt_global, = ax_global.plot([], [], 'ro', markersize=8)
ax_global.set_aspect('equal')
ax_global.set_xlim(bounds[0])
ax_global.set_ylim(bounds[1])

# Center: True Field (Interpolated)
ax_true = fig.add_subplot(gs[1])
ax_true.set_title("2. Local Reality\n(Actual Field)")
q_true = ax_true.quiver(LX, LY, np.zeros_like(LX), np.zeros_like(LY), color='black', scale=10, width=0.008)
pt_true, = ax_true.plot([0], [0], 'ro', markersize=8) # Particle is always at (0,0) in local view
ax_true.set_xlim(-local_window_size/2, local_window_size/2)
ax_true.set_ylim(-local_window_size/2, local_window_size/2)
ax_true.set_aspect('equal')
ax_true.grid(True, alpha=0.3)

# Right: Linear Approximation
ax_linear = fig.add_subplot(gs[2])
ax_linear.set_title("3. Linear Reconstruction\n(Using only $\mathbf{v}_0$ and $\mathbf{J}$)")
q_linear = ax_linear.quiver(LX, LY, np.zeros_like(LX), np.zeros_like(LY), color='red', scale=10, width=0.008)
pt_linear, = ax_linear.plot([0], [0], 'ro', markersize=8)
ax_linear.set_xlim(-local_window_size/2, local_window_size/2)
ax_linear.set_ylim(-local_window_size/2, local_window_size/2)
ax_linear.set_aspect('equal')
ax_linear.grid(True, alpha=0.3)

def get_data_at_pos(px, py):
    # Helper to get V and J at position (bilinear interpolation)
    v0 = bilinear_interpolate(v_field.data, bounds, (px, py))
    J0 = bilinear_interpolate(J_field.data, bounds, (px, py))
    return v0, J0

def get_true_local_field(px, py):
    # Sample actual field at relative points
    vecs = []
    for i in range(len(local_points)):
        rx, ry = local_points[i]
        # Query global field at px+rx, py+ry
        qx, qy = px + rx, py + ry

        vecs.append(bilinear_interpolate(v_field.data, bounds, (qx, qy)))
    return np.array(vecs).T # (2, N)

def update(frame):
    if frame >= len(trajectory): return pt_global,
    
    pos = trajectory[frame]
    px, py = pos
    
    # 1. Update Global
    line_traj.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
    pt_global.set_data([px], [py])
    
    # 2. Get Local Data
    v0, J0 = get_data_at_pos(px, py)
    
    # 3. Compute Linear Approximation for all points in window
    # v_linear = v0 + J . dr
    # dr is local_points (N, 2)
    delta_r = local_points.T # (2, N)
    correction = J0 @ delta_r # (2, N)
    v_linear = v0[:, None] + correction # (2, N) (Broadcast v0)
    
    # 4. Compute True Field (Sampled)
    v_true = get_true_local_field(px, py) # (2, N)
    
    # Update Quivers
    # Reshape to (Res, Res) for plotting if needed or flattened
    U_true = v_true[0, :]
    V_true = v_true[1, :]
    q_true.set_UVC(U_true, V_true)
    
    U_lin = v_linear[0, :]
    V_lin = v_linear[1, :]
    q_linear.set_UVC(U_lin, V_lin)
    
    # Update Title with Error estimate
    error = np.mean(np.linalg.norm(v_true - v_linear, axis=0))
    ax_linear.set_xlabel(f"Mean Approx Error: {error:.3f}")
    
    return pt_global, line_traj, q_true, q_linear

ani = FuncAnimation(fig, update, frames=len(trajectory), interval=50, blit=False)

print("Opening Taylor Expansion visualization...")
plt.show()
