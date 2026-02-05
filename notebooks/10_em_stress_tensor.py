import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../src'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- EM Physics Helpers ---
def electric_field_point_charge(q, x0, y0, X, Y):
    """Calculate E field for point charge q at (x0, y0)."""
    k = 1.0 # Coulomb constant
    Rx = X - x0
    Ry = Y - y0
    R2 = Rx**2 + Ry**2 + 1e-9
    R = np.sqrt(R2)
    # E = k * q / R^2 * r_hat = k * q / R^3 * R_vec
    Ex = k * q * Rx / (R**3)
    Ey = k * q * Ry / (R**3)
    return Ex, Ey

def maxwell_stress_tensor_2d(Ex, Ey):
    """
    Calculate Maxwell Stress Tensor (Electrostatic part)
    T_ij = epsilon_0 (E_i E_j - 0.5 delta_ij E^2)
    In 2D (xy plane), assuming Ez=0.
    """
    eps0 = 1.0 
    E2 = Ex**2 + Ey**2
    
    # Txx = eps0 (Ex*Ex - 0.5 * E2)
    Txx = eps0 * (Ex*Ex - 0.5 * E2)
    
    # Txy = eps0 (Ex*Ey)
    Txy = eps0 * (Ex*Ey)
    
    # Tyy = eps0 (Ey*Ey - 0.5 * E2)
    Tyy = eps0 * (Ey*Ey - 0.5 * E2)
    
    # Tyx = Txy (Symmetric)
    
    return Txx, Txy, Tyy

# --- Setup Data ---
bounds = ((-2, 2), (-2, 2))
shape = (15, 15)

x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')

# 1. Create Field (Point Charge at Center)
q = 1.0
Ex, Ey = electric_field_point_charge(q, 0, 0, X, Y)

# 2. Calculate Stress Tensor Field
Txx, Txy, Tyy = maxwell_stress_tensor_2d(Ex, Ey)

# --- Visualization ---
fig, ax = plt.subplots(figsize=(10, 10))
fig.suptitle("Maxwell Stress Tensor (Electrostatics)\nProof of Universality: Tension along Field Lines", fontsize=16)

ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')

# Plot Vector Field Lines (Electric Field)
ax.streamplot(X.T, Y.T, Ex.T, Ey.T, color='lightgray', density=1.5)

# Plot Principal Stresses (Eigenvectors of T)
scale = 0.5

for i in range(shape[0]):
    for j in range(shape[1]):
        # Construct Matrix T
        T = np.array([
            [Txx[i, j], Txy[i, j]],
            [Txy[i, j], Tyy[i, j]]
        ])
        
        # Eigen decomposition
        evals, evecs = np.linalg.eigh(T)
        
        # Sort
        # Evals of Maxwell Stress Tensor:
        # One is +0.5 E^2 (Tension along field)
        # One is -0.5 E^2 (Pressure perpendicular)
        # So sorted: [negative, positive]
        
        # Max Stress (Positive/Tension) -> Should align with E
        val_tension = evals[1]
        vec_tension = evecs[:, 1]
        
        # Min Stress (Negative/Pressure) -> Should be perp to E
        val_pressure = evals[0]
        vec_pressure = evecs[:, 0]
        
        # Normalize length for visual sanity (field drops distance^4 for stress)
        # We'll use log scale or fixed length
        # Let's just normalize to unit length for direction check, 
        # or use E-magnitude scaling (sqrt(val))
        
        # mag_plot = np.sqrt(np.abs(val_tension)) * scale # Proportional to E
        # Use simple fixed size for clarity of direction
        mag_plot = 0.15
        
        cx, cy = X[i, j], Y[i, j]
        
        # Plot Tension Axis (Red) - DIRECTION OF MAX STRESS
        ax.plot([cx - vec_tension[0]*mag_plot, cx + vec_tension[0]*mag_plot],
                [cy - vec_tension[1]*mag_plot, cy + vec_tension[1]*mag_plot],
                color='red', lw=2)
                
        # Plot Pressure Axis (Blue) - DIRECTION OF MIN STRESS
        ax.plot([cx - vec_pressure[0]*mag_plot, cx + vec_pressure[0]*mag_plot],
                [cy - vec_pressure[1]*mag_plot, cy + vec_pressure[1]*mag_plot],
                color='blue', lw=2)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', lw=2, label='Tension Axis (Aligns with E)'),
    Line2D([0], [0], color='blue', lw=2, label='Pressure Axis (Perpendicular)'),
    Line2D([0], [0], color='lightgray', lw=1, label='Electric Field Lines')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.show()
