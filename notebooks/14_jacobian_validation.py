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

# --- Setup Taylor-Green Vortex Field ---
bounds = ((-2, 2), (-2, 2))
shape = (50, 50)
v_field = VelocityField(shape, bounds)

x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')

Vx = np.sin(X) * np.cos(Y)
Vy = -np.cos(X) * np.sin(Y)
v_field.set_data(np.stack([Vx, Vy], axis=0))

J_num = calculate_jacobian(v_field).data

# Analytic Jacobian for Taylor-Green:
# J00 = dVx/dx = cos(x) cos(y)
# J01 = dVx/dy = -sin(x) sin(y)
# J10 = dVy/dx = sin(x) sin(y)
# J11 = dVy/dy = -cos(x) cos(y)
J_true = np.zeros_like(J_num)
J_true[0, 0] = np.cos(X) * np.cos(Y)
J_true[0, 1] = -np.sin(X) * np.sin(Y)
J_true[1, 0] = np.sin(X) * np.sin(Y)
J_true[1, 1] = -np.cos(X) * np.cos(Y)

# Quick comparison for one component (J00)
extent = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Jacobian Check (Numeric vs Analytic) for $J_{00} = \\partial_x v_x$", fontsize=14)

im0 = axes[0].imshow(J_num[0, 0].T, origin='lower', extent=extent, cmap='viridis')
axes[0].set_title("Numeric $J_{00}$")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(J_true[0, 0].T, origin='lower', extent=extent, cmap='viridis')
axes[1].set_title("Analytic $J_{00}$")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

err = np.abs(J_num[0, 0] - J_true[0, 0])
im2 = axes[2].imshow(err.T, origin='lower', extent=extent, cmap='magma')
axes[2].set_title("Absolute Error")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# Optional: print max error for all components (excluding edges)
sl = (slice(None), slice(None), slice(1, -1), slice(1, -1))
max_err = np.max(np.abs(J_num[sl] - J_true[sl]))
print(f"Max interior error across all components: {max_err:.4f}")
