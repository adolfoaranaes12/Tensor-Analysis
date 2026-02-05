
import numpy as np
from ..core.tensor import TensorField

class ElectromagneticTensor(TensorField):
    """
    Represents the Electromagnetic Field Tensor F_mu_nu (Rank 2).
    Typically 4x4 in relativistic physics.
    """
    def __init__(self, grid_shape, domain_bounds):
        # usually 4D space-time, but for simulation often we have 3D space + time evolution
        # Here we assume a snapshot in 3D space, so the tensor is defined at every spatial point.
        # But the tensor itself is 4x4 indices.
        super().__init__(grid_shape, domain_bounds, rank=2)
        # We need to override dimensions likely if we want 4x4 matrices at each 3D point
        # The base class 'rank' implies the tensor dimension matches the space dimension usually?
        # No, rank is the number of indices.
        # But the size of each index usually matches the dimension unless specified otherwise.
        # In base class: expected_shape = (self.dim,) * self.rank + self.grid_shape
        # Here dim=3 (spatial), but indices are 0..3 (4 dimensions).
        # We need to be careful. The base class might be too restrictive if it assumes tensor dim == spatial dim.
        
    def set_data_from_fields(self, E: np.ndarray, B: np.ndarray):
        """
        Construct F_mu_nu from Electric (E) and Magnetic (B) fields.
        E and B should be shape (3, nx, ny, nz)
        F is (4, 4, nx, ny, nz)
        """
        if E.shape != B.shape:
             raise ValueError("E and B shapes must match")
             
        # grid shape from input fields
        self.grid_shape = E.shape[1:]
        
        # Initialize 4x4 tensor field
        # We handle the 4x4 structure manually since base class might assume 3x3
        f_data = np.zeros((4, 4) + self.grid_shape)
        
        # F_0i = -E_i/c (taking c=1)
        # F_i0 = E_i/c
        # F_ij = -epsilon_ijk B_k
        
        # Components
        Ex, Ey, Ez = E[0], E[1], E[2]
        Bx, By, Bz = B[0], B[1], B[2]
        
        # Row 0 and Col 0 (Time parts with E)
        f_data[0, 1] = -Ex
        f_data[1, 0] = Ex
        f_data[0, 2] = -Ey
        f_data[2, 0] = Ey
        f_data[0, 3] = -Ez
        f_data[3, 0] = Ez
        
        # Spatial parts (Magnetic)
        # F_12 = -Bz
        f_data[1, 2] = -Bz
        f_data[2, 1] = Bz
        
        # F_13 = By
        f_data[1, 3] = By
        f_data[3, 1] = -By
        
        # F_23 = -Bx
        f_data[2, 3] = -Bx
        f_data[3, 2] = Bx
        
        self.data = f_data
        
    def tensor_invariant(self):
        """
        Calculates F_mu_nu * F^mu_nu.
        Proportional to B^2 - E^2.
        Returns Rank 0 scalar field.
        """
        # Simplification: assume Minkowski metric (-1, 1, 1, 1) or (+1, -1, -1, -1)
        # This requires tensor contraction logic.
        pass
