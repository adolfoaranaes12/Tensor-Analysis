
import numpy as np
from ..core.tensor import TensorField

class StressTensor(TensorField):
    """
    Represents the Cauchy Stress Tensor (Rank 2, Symmetric).
    """
    def __init__(self, grid_shape, domain_bounds):
        super().__init__(grid_shape, domain_bounds, rank=2)
        
    def von_mises(self):
        """
        Calculates the Von Mises stress (scalar field).
        Useful for yield criteria.
        """
        if self.data is None:
            return None

        s = self.data

        if self.dim == 3:
            # s shape: (3, 3, nx, ny, nz)
            s11, s12, s13 = s[0, 0], s[0, 1], s[0, 2]
            s22, s23 = s[1, 1], s[1, 2]
            s33 = s[2, 2]

            # sigma_vm = sqrt( 0.5 * [ (s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2
            #                           + 6*(s12^2 + s23^2 + s13^2) ] )
            term1 = (s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2
            term2 = 6 * (s12 ** 2 + s23 ** 2 + s13 ** 2)
            return np.sqrt(0.5 * (term1 + term2))

        if self.dim == 2:
            # Plane stress assumption (s33 = 0).
            # sigma_vm = sqrt(s11^2 - s11*s22 + s22^2 + 3*s12^2)
            s11, s12 = s[0, 0], s[0, 1]
            s22 = s[1, 1]
            return np.sqrt(s11 ** 2 - s11 * s22 + s22 ** 2 + 3 * (s12 ** 2))

        raise ValueError("Von Mises stress is only implemented for 2D or 3D tensors.")
