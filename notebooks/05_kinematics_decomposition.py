import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Math Helpers ---
def get_sphere_points(radius=1.0, resolution=15):
    """Generate points on a sphere."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(resolution), np.cos(v))
    return np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

# --- Visualization Setup ---
fig = plt.figure(figsize=(12, 12))
fig.suptitle("Kinematics Decomposition: The 3 Components of Motion", fontsize=16)

# Initial sphere points
initial_points = get_sphere_points(radius=1.0)

# Create 4 subplots
axes = []
titles = [
    "1. Translation ($\mathbf{v}_0$)\n(Linear Velocity)",
    "2. Strain ($\mathbf{D}$)\n(Symmetric / Shape Change)",
    "3. Rotation ($\mathbf{W}$)\n(Antisymmetric / Spin)",
    "4. Total Motion\n(Combined)"
]

scatters = []

for i in range(4):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    ax.set_title(titles[i])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    # Fixed limits to see movement
    limit = 2.5
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    # Reference (Ghost)
    ax.scatter(initial_points[:, 0], initial_points[:, 1], initial_points[:, 2], 
               c='gray', marker='.', alpha=0.1)
    
    # Active Plot
    sc = ax.scatter(initial_points[:, 0], initial_points[:, 1], initial_points[:, 2], 
                    c='b', marker='o', s=20, alpha=0.8)
    axes.append(ax)
    scatters.append(sc)

# --- Deformation Logic ---
def get_deformed_points(t, mode):
    points = initial_points.copy()
    
    # Parameters oscillating with time t
    # Translation vector
    trans_vec = np.array([0.5 * np.sin(t), 0.5 * np.cos(t), 0])
    
    # Strain factor (Oscillate stretch)
    s = 1 + 0.5 * np.sin(t*2)
    strain_tensor = np.array([[s, 0, 0], [0, 1/s, 0], [0, 0, 1]])
    
    # Rotation angle (Continuous spin)
    angle = t 
    c, s_rot = np.cos(angle), np.sin(angle)
    # Rotate around Z
    rot_matrix = np.array([[c, -s_rot, 0], [s_rot, c, 0], [0, 0, 1]])
    
    if mode == 0: # Translation Only
        return points + trans_vec
        
    elif mode == 1: # Strain Only
        return points @ strain_tensor.T
        
    elif mode == 2: # Rotation Only
        return points @ rot_matrix.T
        
    elif mode == 3: # Total (Order matters: usually Strain -> Rotate -> Translate for local affine)
        # 1. Strain
        p = points @ strain_tensor.T
        # 2. Rotate
        p = p @ rot_matrix.T
        # 3. Translate
        p = p + trans_vec
        return p

def update(frame):
    t = frame * 0.05
    
    for i in range(4):
        new_pts = get_deformed_points(t, mode=i)
        scatters[i]._offsets3d = (new_pts[:, 0], new_pts[:, 1], new_pts[:, 2])
        
    return scatters

# Create Animation
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)

print("Opening 4-Panel Kinematics Animation...")
plt.show()
