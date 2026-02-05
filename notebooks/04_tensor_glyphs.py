import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Add src to path correctly regardless of CWD
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

from tensor_vis.fluids.velocity_field import VelocityField
from tensor_vis.kinematics.jacobian import calculate_jacobian

# --- Setup Data ---
bounds = ((-2, 2), (-2, 2))
shape = (15, 15) # Coarser grid for glpyhs so they don't overlap too much
v_field = VelocityField(shape, bounds)

x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')

# Define Field: v = (x, -y) -> Pure Shear / Extension (Hyperbolic)
# velocity: v_x = x, v_y = -y
# Jacobian: [[1, 0], [0, -1]] -> Constant everywhere!
# Let's do something more interesting:
# v = (sin(x)cos(y), -cos(x)sin(y))
Vx = np.sin(X) * np.cos(Y)
Vy = -np.cos(X) * np.sin(Y)
v_field.set_data(np.stack([Vx, Vy], axis=0))

J_field = calculate_jacobian(v_field)

# --- Visualization ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("Rate of Deformation Field (Symmetric Part of Gradient)\nEllipsoids show principal stretch directions")
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')

# Plot Background Streamlines
ax.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray', density=1.5)

# Calculate Rate of Deformation Tensor D = 0.5 * (J + J.T)
# And visualize as Ellipsoids
# Scale for ellipses
scale = 0.4

for i in range(shape[0]):
    for j in range(shape[1]):
        # Get Jacobian at this point
        J = J_field.data[:, :, i, j]
        
        # Symmetric Part (Rate of Deformation)
        D = 0.5 * (J + J.T)
        
        # Eigen decomposition
        # D is symmetric, so eigs are real
        evals, evecs = np.linalg.eigh(D)
        
        # Sort by eigenvalue strength?
        # evals are principal stretching rates
        # evecs columns are directions
        
        # Visualization Mapping:
        # We want to draw an ellipse.
        # Axis 1 length: 1 + scale * eval1 ?? Or just proportional to eval magnitude?
        # Usually we visualize the deformation of a unit circle.
        # So axes lengths are exp(eval * dt) ~ 1 + eval * dt
        # Let's just map eval directly to length relative to a base circle radius
        
        # Ensure we don't get negative lengths for drawing
        # Let's map radius = base_r * (1 + alpha * eval)
        base_r = 0.1
        alpha = 1.0
        
        r1 = base_r * (1 + alpha * evals[0])
        r2 = base_r * (1 + alpha * evals[1])
        
        # Angle of first eigenvector
        # evecs[:, 0] is [vx, vy]
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        
        # Color by Shear Magnitude or Divergence (Trace)
        # Trace(D) = div(v). For this flow sin(x)cos(y) - cos(x)sin(y) -> div should be near 0?
        # d(sin x cos y)/dx = cos x cos y
        # d(-cos x sin y)/dy = -cos x cos y
        # Sum = 0. Incompressible!
        
        # Let's color by Von Mises (Shear intensity)
        # For 2D: sqrt( (e1-e2)^2 ) ~ |e1 - e2|
        shear_mag = np.abs(evals[0] - evals[1])
        
        # Draw Ellipse
        # Matplotlib Ellipse takes diameter (width, height)
        ell = Ellipse(xy=(X[i, j], Y[i, j]), 
                      width=r1*2, height=r2*2, 
                      angle=angle,
                      edgecolor='black',
                      facecolor=plt.cm.viridis(shear_mag / 2.0), # Normalize?
                      alpha=0.8)
        ax.add_patch(ell)

# Add colorbar for shear
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=2.0))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Shear Magnitude ($|\\lambda_1 - \\lambda_2|$)")

plt.show()
