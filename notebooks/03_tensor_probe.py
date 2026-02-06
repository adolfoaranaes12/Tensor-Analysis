import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add src to path correctly regardless of CWD
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

from tensor_vis.fluids.velocity_field import VelocityField
from tensor_vis.kinematics.jacobian import calculate_jacobian
from tensor_vis.core.interp import bilinear_interpolate

# --- Setup Data ---
bounds = ((-2, 2), (-2, 2)) # 2D for clarity in "Probe" view (easier to see distortion)
# We can do 3D but 2D is often clearer for "vectors around a point" teaching concepts first
shape = (20, 20)
v_field = VelocityField(shape, bounds)

x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')

# Define Field: v = (sin(y), cos(x)) - Cellular flow / Periodic
Vx = np.sin(Y)
Vy = np.cos(X)
v_field.set_data(np.stack([Vx, Vy], axis=0))

J_field = calculate_jacobian(v_field)

# --- Visualization Setup ---
fig = plt.figure(figsize=(12, 6))
fig.suptitle("Tensor Probe: Local Gradient Deformation ($J \\cdot d\\vec{r}$)", fontsize=14)

# Left: Global Field with Probe Path
ax_global = fig.add_subplot(121)
ax_global.set_title("Global Velocity Field & Probe Path")
ax_global.set_xlim(-2, 2); ax_global.set_ylim(-2, 2)
ax_global.set_aspect('equal')
# Plot Streamlines
strm = ax_global.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray')

# Probe Point (initially at center)
probe_scatter = ax_global.scatter([0], [0], color='red', s=100, zorder=5, label='Probe')
# Probe Path (Circular path for demo)
theta_path = np.linspace(0, 2*np.pi, 200)
path_r = 1.0
path_x = path_r * np.cos(theta_path)
path_y = path_r * np.sin(theta_path)
ax_global.plot(path_x, path_y, 'r--', alpha=0.5)

# Right: Local Probe View (Zoomed)
# "Sphere" (Circle) center with vectors around it
ax_local = fig.add_subplot(122)
ax_local.set_title("Local Deformation Gradient")
ax_local.set_xlim(-1.5, 1.5); ax_local.set_ylim(-1.5, 1.5)
ax_local.set_aspect('equal')
ax_local.grid(True)

# The "Particle" Element (Sphere/Circle)
circle = plt.Circle((0, 0), 0.2, color='red', alpha=0.3)
ax_local.add_patch(circle)

# Initial vectors around the probe (representing a small material element d_r)
# We will show how these d_r vectors change: d_v = J * d_r
num_vecs = 12
angles = np.linspace(0, 2*np.pi, num_vecs, endpoint=False)
dr_vecs = np.stack([np.cos(angles), np.sin(angles)], axis=1) # Shape (N, 2)

# Quivers for d_r (Material Element shape) - Blue
# Quivers for d_v (Change in velocity / Deformation rate) - Green 
# No, let's show the "Deformed" vectors: d_r_new = d_r + J * d_r * dt?
# Or just show the Gradient vectors J*dr?
# The user said "vectors around it which was the force". In fluids, force ~ pressure grad or viscosity.
# But for kinematics, J * dr is the relative velocity of neighboring points.
# Let's plot J * dr (Relative Velocity) attached to the displacements.

quiver_local_dr = ax_local.quiver(np.zeros(num_vecs), np.zeros(num_vecs), 
                                  dr_vecs[:, 0], dr_vecs[:, 1], 
                                  color='gray', scale=1, scale_units='xy', angles='xy', width=0.005,
                                  label='Material Element ($d\\vec{r}$)')

quiver_local_Jdr = ax_local.quiver(dr_vecs[:, 0], dr_vecs[:, 1], 
                                   np.zeros(num_vecs), np.zeros(num_vecs),
                                   color='green', scale=1, scale_units='xy', angles='xy', width=0.01,
                                   label='Gradient Action ($J \\cdot d\\vec{r}$)')

ax_local.legend(loc='upper right', fontsize='small')

def get_interpolated_J(px, py):
    # Bilinear interpolation for smoother probe motion
    return bilinear_interpolate(J_field.data, bounds, (px, py))

def update(frame):
    # Move Probe along path
    px = path_x[frame % len(path_x)]
    py = path_y[frame % len(path_y)]
    
    # Update Global View
    probe_scatter.set_offsets(np.c_[px, py])
    
    # Update Local View
    # Get J at (px, py)
    J_local = get_interpolated_J(px, py)
    
    # Calculate J * dr for all dr vectors
    # J is (2, 2), dr is (N, 2). We want (J @ dr.T).T -> (N, 2)
    delta_v = (J_local @ dr_vecs.T).T
    
    # Update quivers
    # Arrows start at dr_vecs (the surface of the unit circle/sphere)
    # And point in direction of delta_v
    quiver_local_Jdr.set_offsets(dr_vecs)
    quiver_local_Jdr.set_UVC(delta_v[:, 0], delta_v[:, 1])
    
    ax_local.set_title(f"Local Gradient @ ({px:.2f}, {py:.2f})\nGreen = Relative Velocity of Neighbors")
    
    return probe_scatter, quiver_local_Jdr

ani = FuncAnimation(fig, update, frames=len(path_x), interval=50, blit=False)
plt.show()
