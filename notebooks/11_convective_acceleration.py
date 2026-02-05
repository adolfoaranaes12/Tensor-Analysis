import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

from tensor_vis.fluids.velocity_field import VelocityField
from tensor_vis.kinematics.jacobian import calculate_jacobian

# --- Setup Data ---
bounds = ((-2, 2), (-2, 2))
shape = (15, 15)
v_field = VelocityField(shape, bounds)

x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')

# Define Field: Taylor-Green Vortex Sheet
# This field has interesting curvature and speed changes.
Vx = np.sin(X) * np.cos(Y)
Vy = -np.cos(X) * np.sin(Y)
v_field.set_data(np.stack([Vx, Vy], axis=0))

J_field = calculate_jacobian(v_field)

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Dynamics Analogy: Convective Acceleration $(\mathbf{v} \cdot \\nabla) \mathbf{v}$\nHow the Tensor Maps Velocity $\\mathbf{v}$ to Acceleration $\\mathbf{a}$", fontsize=16)

# --- Plot 1: Total Acceleration Mapping ---
ax1.set_title("Input: Velocity $\\mathbf{v}$ (Black)\nOutput: Acceleration $\\mathbf{a}_{conv}$ (Red)\n$\\mathbf{a} = \\mathbf{L} \\cdot \\mathbf{v}$")
ax1.set_xlim(-2.2, 2.2); ax1.set_ylim(-2.2, 2.2)
ax1.set_aspect('equal')
ax1.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray', density=1.0)

# --- Plot 2: Decomposition (Strain vs Rotation) ---
ax2.set_title("Decomposition of Acceleration\nStrain Push (Green) vs Rotation Push (Blue)\n$\\mathbf{a} = \\mathbf{D}\\cdot\\mathbf{v} + \\mathbf{W}\\cdot\\mathbf{v}$")
ax2.set_xlim(-2.2, 2.2); ax2.set_ylim(-2.2, 2.2)
ax2.set_aspect('equal')
ax2.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray', density=1.0)

# Calculate and Plot Vectors
# We'll subsample slightly to avoid crowding if needed, or use the 15x15 grid.
scale = 0.5

for i in range(shape[0]):
    for j in range(shape[1]):
        # Current Position
        cx, cy = X[i, j], Y[i, j]
        
        # Velocity Vector v
        v_vec = v_field.data[:, i, j]
        
        # Gradient Tensor L
        L = J_field.data[:, :, i, j]
        
        # Convective Acceleration a = L * v
        a_vec = L @ v_vec
        
        # Decomposition
        D = 0.5 * (L + L.T) # Symmetric
        W = 0.5 * (L - L.T) # Antisymmetric
        
        a_strain = D @ v_vec # "Electric-like" force (Speed/Stretch)
        a_rot = W @ v_vec    # "Magnetic-like" force (Turn/Spin)
        
        # --- Plot 1: v -> a ---
        # Plot v (Black)
        ax1.arrow(cx, cy, v_vec[0]*0.2, v_vec[1]*0.2, color='black', alpha=0.3, width=0.005)
        # Plot a (Red) - The resultant force
        ax1.arrow(cx, cy, a_vec[0]*scale, a_vec[1]*scale, color='red', width=0.01)
        
        # --- Plot 2: Decomposition ---
        # Plot Strain part (Green) - Aligns with Principal Axes?
        ax2.arrow(cx, cy, a_strain[0]*scale, a_strain[1]*scale, color='green', width=0.01)
        
        # Plot Rotation part (Blue) - Always Perpendicular to v?
        # Check: v . (W v) = v^T W v. Since W is skew-sym, xWx = 0.
        # So YES, Rotational force is ALWAYS perpendicular to velocity.
        # Just like Magnetic Force F = q(v x B).
        ax2.arrow(cx, cy, a_rot[0]*scale, a_rot[1]*scale, color='blue', width=0.01)

# Legends
from matplotlib.lines import Line2D
legend1 = [
    Line2D([0], [0], color='black', alpha=0.5, lw=2, label='Velocity $\\mathbf{v}$'),
    Line2D([0], [0], color='red', lw=2, label='Total Acceleration $\\mathbf{a}$')
]
ax1.legend(handles=legend1, loc='upper right')

legend2 = [
    Line2D([0], [0], color='green', lw=2, label='Strain Force (Speed Change)'),
    Line2D([0], [0], color='blue', lw=2, label='Rotation Force (Turning)'),
    Line2D([0], [0], color='lightgray', lw=1, label='Resultant Dynamics')
]
ax2.legend(handles=legend2, loc='upper right')

plt.show()
