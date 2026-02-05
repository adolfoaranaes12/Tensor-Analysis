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

# Define Field: v = (sin(y), cos(x)) - Cellular Flow
Vx = np.sin(Y)
Vy = np.cos(X)
v_field.set_data(np.stack([Vx, Vy], axis=0))

J_field = calculate_jacobian(v_field)

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Direct Sum Decomposition: $\\nabla \\mathbf{v} = \\mathbf{D} \\oplus \\mathbf{W}$\nVisualized as Vector Fields", fontsize=16)

# 1. Strain Axis Field (From Symmetric D)
ax1.set_title("Symmetric Subspace $\\mathbf{D}$ (Strain)\nArrows = Direction of Max Stretch (Principal Eigenvector)")
ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')

# 2. Vorticity Field (From Antisymmetric W)
ax2.set_title("Antisymmetric Subspace $\\mathbf{W}$ (Rotation)\nArrows = Vorticity Vector (Axis of Rotation)")
ax2.set_xlim(-2, 2); ax2.set_ylim(-2, 2)
ax2.set_aspect('equal')

# Background streamlines (optional context)
ax1.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray', density=1.0)
ax2.streamplot(X.T, Y.T, Vx.T, Vy.T, color='lightgray', density=1.0)

# Loop and calculate vectors
strain_u = np.zeros_like(X)
strain_v = np.zeros_like(Y)
strain_mag = np.zeros_like(X)

vort_u = np.zeros_like(X) # 2D 'vector', technically z-component
vort_v = np.zeros_like(Y) # 2D plan, vorticity is perpendicular (z)
# But for plotting "arrows", we might visualize the flow loop? 
# OR just the magnitude as color?
# If the user wants "Electric and Magnetic field maps with arrows", usually B is orthogonal in 2D.
# In 2D flow, vorticity is a scalar field (vector in +/- Z).
# To make it "look like arrows" in 2D, we can't really draw Z-arrows easily.
# BUT, we can draw the Shear Vector? Or just color.
# Actually, let's strictly follow "Vector Field" request.
# In 2D, Vorticity is a pseudo-scalar. 
# Maybe we simulate a 3D slice where we *can* draw the arrows? 
# Or we just interpret it as a field of rotation strength.

# Let's stick to Eigenvectors for D.
# For W in 2D, it's just a number. 
# BUT, maybe the user implies the "force" analogy? which is E and B.
# In flow, maybe they mean the Acceleration field?
# Let's stick to Kinematics. 
# We'll plot the D-eigenvector.
# For W, since it's 2D, we'll plot circles (patches) sealed by vorticity?
# Or we assume 3D visualization to allow arrows.

# Let's do 3D like the probe script but for the field? 
# No, "maps with arrows" usually implies 2D slice.

# Decision:
# D: Principal Eigenvector field (lines).
# W: We can't draw Z-arrows in 2D plane easily. 
# We'll use color for W magnitude (Vorticity) and maybe small circular arrows?
# OR we switch to a 3D Quiver plot where W points in Z.

# Let's try 2D Quiver for Strain, and Colormap for Vorticity with superimposed circular glyphs?
# User said "maps with arrows ... like electric and magnetic".
# E and B are both vectors in 3D.
# Let's assume the user is okay with 3D quivers if needed, or 2D projection.
# I will calculate Strain Eigenvector.

for i in range(shape[0]):
    for j in range(shape[1]):
        J = J_field.data[:, :, i, j]
        D = 0.5 * (J + J.T)
        W = 0.5 * (J - J.T)
        
        # Eigen decomp of D
        evals, evecs = np.linalg.eigh(D)
        # evecs[:, k] is eigenvector.
        # Pick max positive eigenvalue (Extension)
        idx_max = np.argmax(evals)
        val = evals[idx_max]
        vec = evecs[:, idx_max]
        
        # If compression everywhere (all negative), pick least negative?
        # Usually looking for principal strain axis.
        
        strain_u[i, j] = vec[0] * np.abs(val) # Scale by magnitude
        strain_v[i, j] = vec[1] * np.abs(val)
        
        # Vorticity (2D)
        # W_10 - W_01
        w_val = W[1, 0] - W[0, 1]
        # In 3D this is the Z component.
        # We can't plot Z-arrow in 2D plot effectively (it's a dot).
        # We'll stick to color for this one or maybe a "rotation vector" if we tilt the plot?
        # Let's assume standard 2D map: Color = Vorticity.
        pass

# Plotting Strain Vectors
q1 = ax1.quiver(X, Y, strain_u, strain_v, color='red', pivot='mid', headaxislength=0, headlength=0) 
# Headless arrows (lines) are better for strain axes since +/- direction is same
ax1.quiverkey(q1, 0.9, 1.05, 1, 'Max Stretch Axis', labelpos='E')


# Plotting Vorticity 
# Since user wants "Arrows", and W is perpendicular, maybe we can't do arrows in 2D.
# I will add a text note and use color.
# Explicit Vorticity Calc for Array
W_field = np.zeros_like(X)
for i in range(shape[0]):
    for j in range(shape[1]):
        J = J_field.data[:, :, i, j]
        W_field[i, j] = J[1, 0] - J[0, 1]

# Visualize Vorticity Magnitude using Color
im2 = ax2.imshow(W_field.T, origin='lower', extent=[-2, 2, -2, 2], cmap='bwr', vmin=-2, vmax=2)
plt.colorbar(im2, ax=ax2, label="Vorticity (Rotation Axis $\\hat{k}$)")

ax2.text(0, 0, "Vorticity in 2D is $\\perp$ to plane.\nColor shows magnitude/direction.", 
         ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

plt.show()
