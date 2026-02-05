import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- Math Helpers ---
def get_circle_points(n=16):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    # Unit circle points
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1) # (N, 2)
    return points

def plot_vector_field(ax, vectors, color, title, label=None):
    # vectors is (N, 2) displacement vectors at the circle points
    points = get_circle_points(len(vectors))
    
    # Plot the "Material Element" (Circle)
    circle = plt.Circle((0, 0), 1.0, color='gray', alpha=0.1)
    ax.add_patch(circle)
    
    # Plot the vectors originating from surface
    ax.quiver(points[:, 0], points[:, 1], 
              vectors[:, 0], vectors[:, 1], 
              color=color, scale=1, scale_units='xy', angles='xy', width=0.015, label=label)
    
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    # Center point
    ax.plot([0], [0], 'k+', markersize=5)

# --- 1. Define a Local Velocity Gradient (Jacobian) ---
# We pick a matrix that has ALL components: Expansion, Rotation, and Shear.
# L = [[1, 2], [3, 1]]
# tr(L) = 2 -> Expansion
# curl = 3 - 2 = 1 -> Rotation
# Symmetric D = [[1, 2.5], [2.5, 1]]
# Off-diagonals != 0 -> Shear

L = np.array([[1.0, 2.0],
              [3.0, 1.0]])

# --- 2. Perform Decomposition ---

# A. Symmetric (Strain Rate) vs Antisymmetric (Rotation)
# D = 0.5 * (L + L.T)
# W = 0.5 * (L - L.T)
D = 0.5 * (L + L.T)
W = 0.5 * (L - L.T)

# B. Volumetric (Spherical) vs Deviatoric (Shape)
# In 2D, Volumetric part is 1/2 * trace(D) * Identity
# This represents purely isotropic expansion/compression
trace_D = np.trace(D)
Volumetric = 0.5 * trace_D * np.eye(2)

# Deviatoric part is what remains of D (Pure Shear / Shape Change)
# D' = D - Vol
Deviatoric = D - Volumetric

# Verify Sum
# L = W + Vol + Dev
L_reconstructed = W + Volumetric + Deviatoric
assert np.allclose(L, L_reconstructed), "Decomposition math failed!"

# --- 3. Compute Vector Fields on a Circle ---
points = get_circle_points(24) # (N, 2)
dr = points.T # (2, N)

# Calculate Delta v for each component
# dv = Matrix @ dr
dv_total = L @ dr        # Total Gradient Field
dv_rot = W @ dr          # Rotation Field
dv_vol = Volumetric @ dr # Expansion Field
dv_dev = Deviatoric @ dr # Shear Field

# --- 4. Visualization ---
fig = plt.figure(figsize=(16, 6))
fig.suptitle(r"Strain Rate Decomposition: $\mathbf{L} = \mathbf{W} + \mathbf{D}_{vol} + \mathbf{D}_{dev}$" + "\nVisualizing the Matrix components as Vector Fields", fontsize=16)

gs = gridspec.GridSpec(1, 4)

# 1. Total Field
ax1 = fig.add_subplot(gs[0])
plot_vector_field(ax1, dv_total.T, 'black', r"1. Total Gradient $\mathbf{L}$" + "\n(The Full Transformation)", "Total")

# 2. Rotation
ax2 = fig.add_subplot(gs[1])
plot_vector_field(ax2, dv_rot.T, 'blue', r"2. Rotation $\mathbf{W}$" + "\n(Antisymmetric Part)", "Spin")

# 3. Volumetric
ax3 = fig.add_subplot(gs[2])
plot_vector_field(ax3, dv_vol.T, 'green', r"3. Volumetric Strain" + "\n(Spherical Part of D)", "Expansion")

# 4. Deviatoric
ax4 = fig.add_subplot(gs[3])
plot_vector_field(ax4, dv_dev.T, 'red', r"4. Deviatoric Strain" + "\n(Shear Part of D)", "Distortion")

# Add 'Plus' and 'Equals' signs visually
fig.text(0.30, 0.5, "=", fontsize=30, ha='center', va='center')
fig.text(0.51, 0.5, "+", fontsize=30, ha='center', va='center')
fig.text(0.73, 0.5, "+", fontsize=30, ha='center', va='center')

plt.tight_layout(rect=[0, 0.05, 1, 0.9])
plt.show()
