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
shape = (20, 20)
v_field = VelocityField(shape, bounds)

x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')

# Define Field: v = (x, -y) -> Pure Shear / Stagnation Point Flow
# This has clear principal axes aligned with x and y.
# Let's add rotation to make it interesting:
# v = (x - y, x + y) ??
# Let's stick to the Cellular Flow from before as it has variation.
# v = (sin(x)cos(y), -cos(x)sin(y))
Vx = np.sin(X) * np.cos(Y)
Vy = -np.cos(X) * np.sin(Y)
v_field.set_data(np.stack([Vx, Vy], axis=0))

J_field = calculate_jacobian(v_field)

# --- Visualization ---
fig, ax = plt.subplots(figsize=(10, 10))
fig.suptitle("Principal Directions of Strain Field\nThe 'Tensor Gradient' Equivalent: Directions of Max Stretch/Compression", fontsize=16)

ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
ax.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray', density=1.5)

# Plot Principal Axes at each point
# We will draw a "Cross" at each point.
# Arm 1: Direction of Max Stretch (Eigenvector 1), Length ~ Eigenvalue 1
# Arm 2: Direction of Max Compression (Eigenvector 2), Length ~ Eigenvalue 2

scale = 0.3

for i in range(shape[0]):
    for j in range(shape[1]):
        J = J_field.data[:, :, i, j]
        D = 0.5 * (J + J.T)
        
        # Eigen decomposition
        evals, evecs = np.linalg.eigh(D)
        
        # eigh returns sorted eigenvalues.
        # evals[0] <= evals[1]
        # In 2D flow if incompressible, lambda1 + lambda2 = 0 => one pos, one neg.
        # So evals[1] is stretch (pos), evals[0] is compress (neg).
        
        # Arm 1 (Max Stretch)
        val1 = evals[1]
        vec1 = evecs[:, 1]
        
        # Arm 2 (Max Compress)
        val2 = evals[0]
        vec2 = evecs[:, 0]
        
        # Draw Lines
        # Center
        cx, cy = X[i, j], Y[i, j]
        
        # Stretch Axis (Red)
        dx1 = vec1[0] * val1 * scale
        dy1 = vec1[1] * val1 * scale
        ax.plot([cx - dx1, cx + dx1], [cy - dy1, cy + dy1], color='red', lw=2)
        
        # Compress Axis (Blue)
        # val2 is likely negative, so length abs(val2)
        dx2 = vec2[0] * np.abs(val2) * scale
        dy2 = vec2[1] * np.abs(val2) * scale
        ax.plot([cx - dx2, cx + dx2], [cy - dy2, cy + dy2], color='blue', lw=2)

# Helper Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', lw=2, label='Max Extension Axis ($\\lambda_1 > 0$)'),
    Line2D([0], [0], color='blue', lw=2, label='Max Compression Axis ($\\lambda_2 < 0$)'),
    Line2D([0], [0], color='lightgray', lw=1, label='Streamlines (Flow V)')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.show()
