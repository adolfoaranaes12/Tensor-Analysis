import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Math Helpers ---
def get_sphere_mesh(radius=1.0, resolution=20):
    """Generate mesh grids for a sphere."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(resolution), np.cos(v))
    return x, y, z

# --- Visualization Setup ---
def setup_visualization():
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle("Kinematics Decomposition: The 3 Components of Motion", fontsize=16)

    # Initial sphere mesh
    X0, Y0, Z0 = get_sphere_mesh(radius=1.0)
    initial_mesh = (X0, Y0, Z0)

    # Create 4 subplots
    axes = []
    titles = [
        r"1. Initial State\n(Reference)",
        r"2. Strain ($\mathbf{D}$)\n(Symmetric / Shape Change)",
        r"3. Rotation ($\mathbf{W}$)\n(Antisymmetric / Spin)",
        r"4. Total Motion\n(Combined)"
    ]
    
    # Colors for each panel
    colors = ['cyan', 'orange', 'lime', 'magenta']

    surfaces = [None] * 4

    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_title(titles[i])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        
        # Fixed limits to see movement
        limit = 2.5
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        
        # Reference (Ghost) - Wireframe for better visibility behind solid
        ax.plot_wireframe(X0, Y0, Z0, color='gray', alpha=0.1, rstride=2, cstride=2)
        
        # Initial Surface (placeholder)
        
        axes.append(ax)
    
    return fig, axes, initial_mesh, colors, surfaces

# --- Deformation Logic ---
def get_deformed_mesh(t, mode, initial_mesh):
    X0, Y0, Z0 = initial_mesh
    shape = X0.shape
    
    # Flatten coordinates for matrix operations: (3, N*N)
    points = np.stack([X0.flatten(), Y0.flatten(), Z0.flatten()])
    
    # Parameters oscillating with time t
    # Translation vector
    trans_vec = np.array([0.5 * np.sin(t), 0.5 * np.cos(t), 0]).reshape(3, 1)
    
    # Strain factor (Oscillate stretch)
    s = 1 + 0.5 * np.sin(t*2)
    strain_tensor = np.array([[s, 0, 0], [0, 1/s, 0], [0, 0, 1]])
    
    # Rotation angle (Continuous spin)
    angle = t 
    c, s_rot = np.cos(angle), np.sin(angle)
    # Rotate around Z
    rot_matrix = np.array([[c, -s_rot, 0], [s_rot, c, 0], [0, 0, 1]])
    
    new_points = points.copy()

    if mode == 0: # Initial State (Static)
        return X0, Y0, Z0 # No Change
        
    elif mode == 1: # Strain Only
        new_points = strain_tensor @ points
        
    elif mode == 2: # Rotation Only
        new_points = rot_matrix @ points
        
    elif mode == 3: # Total (Strain -> Rotate -> Translate)
        # 1. Strain
        new_points = strain_tensor @ points
        # 2. Rotate
        new_points = rot_matrix @ new_points
        # 3. Translate
        new_points = new_points + trans_vec

    # Reshape back to mesh grids
    X_new = new_points[0, :].reshape(shape)
    Y_new = new_points[1, :].reshape(shape)
    Z_new = new_points[2, :].reshape(shape)
    
    return X_new, Y_new, Z_new

def main():
    fig, axes, initial_mesh, colors, surfaces = setup_visualization()

    def update(frame):
        t = frame * 0.05
        
        for i in range(4):
            # Remove old surface
            if surfaces[i] is not None:
                surfaces[i].remove()
            
            # Calculate new shape
            X, Y, Z = get_deformed_mesh(t, mode=i, initial_mesh=initial_mesh)
            
            # Plot new surface
            surfaces[i] = axes[i].plot_surface(X, Y, Z, color=colors[i], alpha=0.9, shade=True)
            
        return surfaces

    # Create Animation (frames reduced slightly to keep it responsive if plotting is slow)
    ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)

    print("Opening 4-Panel Kinematics Animation with Solid Surfaces...")
    plt.show()

if __name__ == "__main__":
    main()
