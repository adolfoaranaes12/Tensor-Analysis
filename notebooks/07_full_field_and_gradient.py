import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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

# Define Field: Taylor-Green Vortex Sheet (Cellular)
# v = (sin(x)cos(y), -cos(x)sin(y))
Vx = np.sin(X) * np.cos(Y)
Vy = -np.cos(X) * np.sin(Y)
v_field.set_data(np.stack([Vx, Vy], axis=0))

J_field = calculate_jacobian(v_field)

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Comparison: Vector Field vs. Tensor Field of its Gradient", fontsize=16)

# --- Plot 1: Vector Field (Velocity) ---
ax1.set_title("Vector Field $\\mathbf{v}$ (Velocity)\nRepresented by Arrows (Quiver)")
ax1.set_xlim(-2.2, 2.2); ax1.set_ylim(-2.2, 2.2)
ax1.set_aspect('equal')

# Streamlines for flow structure
ax1.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray', density=1.5)
# Quiver for vectors
ax1.quiver(X, Y, Vx, Vy, color='blue', pivot='mid', scale=20)


# --- Plot 2: Tensor Field (Gradient) ---
ax2.set_title("Tensor Field $\\nabla \\mathbf{v}$ (Gradient)\nRepresented by Glyphs: Ellipse (Strain) + Color (Vorticity)")
ax2.set_xlim(-2.2, 2.2); ax2.set_ylim(-2.2, 2.2)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# Calculate Max Vorticity for Color Scaling identification
W_field_raw = np.zeros_like(X)
for i in range(shape[0]):
    for j in range(shape[1]):
        J = J_field.data[:, :, i, j]
        W_field_raw[i, j] = J[1, 0] - J[0, 1]
max_w = np.max(np.abs(W_field_raw))

# Draw Glyphs
alpha_scale = 1.0
base_size = 0.15

for i in range(shape[0]):
    for j in range(shape[1]):
        J = J_field.data[:, :, i, j]
        
        # 1. Symmetric Part (Shape)
        D = 0.5 * (J + J.T)
        evals, evecs = np.linalg.eigh(D)
        
        # Ellipse axes
        # Map eigenvalue to stretch factor
        # lambda > 0 : Stretch
        # lambda < 0 : Compress
        # r = base * exp(lambda) is a good mapping for "integrated" deformation
        r1 = base_size * np.exp(evals[0] * alpha_scale)
        r2 = base_size * np.exp(evals[1] * alpha_scale)
        
        # Rotation Angle
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        
        # 2. Antisymmetric Part (Color)
        # Vorticity w = dy/dx - dx/dy
        w_val = J[1, 0] - J[0, 1]
        
        # Color Map: Blue (CW) -> White (0) -> Red (CCW)
        # Normalize -max_w to max_w
        norm_w = 0.5 + 0.5 * (w_val / (max_w + 1e-9))
        color = plt.cm.coolwarm(norm_w)
        
        ell = Ellipse(xy=(X[i, j], Y[i, j]), 
                      width=r1*2, height=r2*2, 
                      angle=angle,
                      edgecolor='black',
                      linewidth=0.5,
                      facecolor=color,
                      alpha=0.9)
        ax2.add_patch(ell)

# Add dummy ScalarMappable for colorbar
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-max_w, vmax=max_w))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Vorticity (Rotation) Magnitude")

plt.show()
