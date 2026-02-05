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

def get_rotation_matrix(theta):
    """
    Creates a 6x6 rotation matrix.
    Rotation between dimension 0 (x1) and dimension 3 (x4).
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(6)
    R[0, 0] = c
    R[0, 3] = -s
    R[3, 0] = s
    R[3, 3] = c
    return R

def calculate_vorticity(jacobian):
    """
    Calculates vorticity vector from Jacobian tensor J_ij = dv_i/dx_j
    w_x = dv_z/dy - dv_y/dz = J_21 - J_12
    w_y = dv_x/dz - dv_z/dx = J_02 - J_20
    w_z = dv_y/dx - dv_x/dy = J_10 - J_01
    Check indices: v=[v0, v1, v2], x=[x0, x1, x2]
    J[i, j] = d v_i / d x_j
    """
    # Assuming J is at a single point for this viz, shape (3, 3)
    J = jacobian
    wx = J[2, 1] - J[1, 2]
    wy = J[0, 2] - J[2, 0]
    wz = J[1, 0] - J[0, 1]
    return np.array([wx, wy, wz])

# --- Setup Data from Physics Engine ---
# 1. Create a Velocity Field (Vortex)
bounds = ((-2, 2), (-2, 2), (-2, 2))
shape = (10, 10, 10)
v_field = VelocityField(shape, bounds)

# Coordinate grid for data generation
coords = [np.linspace(b[0], b[1], s) for b, s in zip(bounds, shape)]
X, Y, Z = np.meshgrid(*coords, indexing='ij')

# Define Field: v = (-y, x, 0) (Simple Vortex in Z-axis)
Vx = -Y
Vy = X
Vz = np.zeros_like(Z)
v_field.set_data(np.stack([Vx, Vy, Vz], axis=0))

# 2. Calculate Jacobian Field
J_field = calculate_jacobian(v_field)

# 3. Pick a sample point for the single-vector visualization
# Let's pick a point at r=(1, 0, 0)
# Index roughly at the middle of Y and Z, and 3/4 of X? 
# Let's just lookup closest index for simplicity or pick arbitrary index
idx = (7, 5, 5) # Sample index
sample_pos = (X[idx], Y[idx], Z[idx])

# Extract Velocity vector at this point
u_vec = v_field.data[:, idx[0], idx[1], idx[2]] # shape (3,)

# Extract Jacobian at this point
J_mat = J_field.data[:, :, idx[0], idx[1], idx[2]] # shape (3, 3)

# Extract Vorticity ("Gradient" representation for vector viz) at this point
w_vec = calculate_vorticity(J_mat)

# Initial 6D State combining Velocity and Vorticity
initial_vector = np.concatenate([u_vec, w_vec])

print(f"Sample Point: {sample_pos}")
print(f"Velocity Vector (u): {u_vec}")
print(f"Vorticity Vector (w): {w_vec}")

# --- Plotting ---
fig = plt.figure(figsize=(12, 6))
fig.suptitle(f"6D Phase Space Viz: Velocity vs Vorticity\nRotation in mixed $x-v_x$ plane", fontsize=14)

# Subplot 1: Velocity Space
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Velocity $\\vec{v}$")
limit = 2.0
ax1.set_xlim(-limit, limit); ax1.set_ylim(-limit, limit); ax1.set_zlim(-limit, limit)
ax1.set_xlabel('vx'); ax1.set_ylabel('vy'); ax1.set_zlabel('vz')

# Subplot 2: Vorticity Space ("Gradient")
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("Vorticity $\\vec{\\omega} = \\nabla \\times \\vec{v}$")
ax2.set_xlim(-limit, limit); ax2.set_ylim(-limit, limit); ax2.set_zlim(-limit, limit)
ax2.set_xlabel('wx'); ax2.set_ylabel('wy'); ax2.set_zlabel('wz')

# Quivers
quiver1 = ax1.quiver(0, 0, 0, u_vec[0], u_vec[1], u_vec[2], color='b', lw=3)
quiver2 = ax2.quiver(0, 0, 0, w_vec[0], w_vec[1], w_vec[2], color='r', lw=3)

# Text
txt1 = ax1.text2D(0.05, 0.95, "", transform=ax1.transAxes)
txt2 = ax2.text2D(0.05, 0.95, "", transform=ax2.transAxes)

def update(frame):
    theta = frame * 0.05
    R = get_rotation_matrix(theta)
    
    # Rotate the 6D vector
    rotated_vector = R @ initial_vector
    
    u = rotated_vector[0:3]
    w = rotated_vector[3:6]
    
    global quiver1, quiver2
    quiver1.remove()
    quiver2.remove()
    
    quiver1 = ax1.quiver(0, 0, 0, u[0], u[1], u[2], color='blue', lw=3)
    quiver2 = ax2.quiver(0, 0, 0, w[0], w[1], w[2], color='red', lw=3)
    
    mag_u = np.linalg.norm(u)
    mag_w = np.linalg.norm(w)
    
    txt1.set_text(f"|v|: {mag_u:.2f}")
    txt2.set_text(f"|w|: {mag_w:.2f}")
    
    return quiver1, quiver2, txt1, txt2

ani = FuncAnimation(fig, update, frames=125, interval=50, blit=False)
plt.show()
