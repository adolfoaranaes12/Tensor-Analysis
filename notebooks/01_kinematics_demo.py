import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

from tensor_vis.fluids.velocity_field import VelocityField
from tensor_vis.kinematics.jacobian import calculate_jacobian

# --- 1. Define Velocity Field ---
# v = (x^2 - y^2, 2xy)
bounds = ((-2, 2), (-2, 2))
shape = (30, 30) # Good density for vectors
v_field = VelocityField(shape, bounds)

x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')

Vx = X**2 - Y**2
Vy = 2*X*Y
v_field.set_data(np.stack([Vx, Vy], axis=0))

# --- 2. Calculate Kinematics ---
J_field = calculate_jacobian(v_field)
J = J_field.data # Shape: (2, 2, Nx, Ny)

# Ax = Vx * J[0,0] + Vy * J[0,1]
# Ay = Vx * J[1,0] + Vy * J[1,1]
Ax = Vx * J[0,0] + Vy * J[0,1]
Ay = Vx * J[1,0] + Vy * J[1,1]

# --- 3. Trajectory Integration using Euler (Lagrangian Path) ---
dt = 0.01
steps = 300
trajectory = []
curr_pos = np.array([-0.1, -1.8]) # Start near bottom center to flow through
trajectory.append(curr_pos.copy())

for _ in range(steps):
    x_val, y_val = curr_pos
    vx = x_val**2 - y_val**2
    vy = 2*x_val*y_val
    
    curr_pos = curr_pos + np.array([vx, vy]) * dt
    trajectory.append(curr_pos.copy())
    
    # Stop if out of bounds
    if abs(curr_pos[0]) > 2 or abs(curr_pos[1]) > 2:
        break
        
trajectory = np.array(trajectory)

# --- 4. Visualization ---
fig = plt.figure(figsize=(16, 8))
fig.suptitle(r"Kinematics Demo: The Tensor Field Experienced by a Particle", fontsize=16)

# Left: Global View
ax_global = fig.add_subplot(121)
ax_global.set_title("Global Velocity Field & Particle Path")
ax_global.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray', density=0.8)
ax_global.plot(trajectory[:,0], trajectory[:,1], 'b--', alpha=0.5, label='Streamline Path')
probe_scatter, = ax_global.plot([], [], 'ro', markersize=8, label='Particle')
ax_global.set_xlim(-2, 2); ax_global.set_ylim(-2, 2)
ax_global.set_aspect('equal')
ax_global.legend(loc='upper right')

# Right: Local Tensor View (The Probe)
ax_local = fig.add_subplot(122)
ax_local.set_title(r"Local Tensor Action: $\mathbf{J} \cdot d\mathbf{r}$" + "\nHow the field distorts a small circle at the particle's location")
ax_local.set_xlim(-1.5, 1.5); ax_local.set_ylim(-1.5, 1.5)
ax_local.set_aspect('equal')
ax_local.grid(True, alpha=0.3)

# Material Element (Circle)
num_vecs = 16
angles = np.linspace(0, 2*np.pi, num_vecs, endpoint=False)
dr_vecs = np.stack([np.cos(angles), np.sin(angles)], axis=1) # Shape (N, 2)
# Visualize the circle itself
ax_local.add_patch(plt.Circle((0, 0), 0.05, color='black', alpha=0.1))

# Quivers
# Gray: Original Shape (dr)
q_dr = ax_local.quiver(np.zeros(num_vecs), np.zeros(num_vecs), 
                       dr_vecs[:, 0], dr_vecs[:, 1], 
                       color='gray', scale=1, scale_units='xy', angles='xy', width=0.005, alpha=0.5)

# Red: Gradient Action (J . dr) - This shows the relative velocity stretching/rotating the circle
q_Jdr = ax_local.quiver(dr_vecs[:, 0], dr_vecs[:, 1], 
                        np.zeros(num_vecs), np.zeros(num_vecs), 
                        color='red', scale=1, scale_units='xy', angles='xy', width=0.01, label='Tensor Action')
ax_local.legend()

def get_interpolated_J(px, py):
    # Map to grid
    dx = 4.0 / (shape[0] - 1)
    dy = 4.0 / (shape[1] - 1)
    idx_x = int((px - bounds[0][0]) / dx)
    idx_y = int((py - bounds[1][0]) / dy)
    idx_x = max(0, min(shape[0]-1, idx_x))
    idx_y = max(0, min(shape[1]-1, idx_y))
    return J_field.data[:, :, idx_x, idx_y] # (2, 2)

def update(frame):
    if frame >= len(trajectory):
        return probe_scatter, q_Jdr
        
    pos = trajectory[frame]
    px, py = pos
    
    # 1. Update Global Position
    probe_scatter.set_data([px], [py])
    
    # 2. Update Local Tensor View
    J_local = get_interpolated_J(px, py)
    
    # Calculate deformation rate: dv = J . dr
    # J is (2,2), dr is (N,2). We want (J @ dr.T).T
    dv = (J_local @ dr_vecs.T).T 
    
    # The arrows start at dr (circle edge) and point in direction of dv
    q_Jdr.set_offsets(dr_vecs)
    q_Jdr.set_UVC(dv[:, 0], dv[:, 1])
    
    # Update Title with numeric values of J
    ax_local.set_xlabel(f"Pos: ({px:.2f}, {py:.2f})\n"
                        f"J = [[{J_local[0,0]:.1f}, {J_local[0,1]:.1f}], [{J_local[1,0]:.1f}, {J_local[1,1]:.1f}]]")
    
    return probe_scatter, q_Jdr

# Output Selection
from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, update, frames=len(trajectory), interval=50, blit=False)

print("Opening interactive animation window...")
plt.show()
