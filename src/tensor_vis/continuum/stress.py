
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
        # s shape: (3, 3, nx, ny, nz)
        
        # Manual calculation for 3D
        s11, s12, s13 = s[0,0], s[0,1], s[0,2]
        s21, s22, s23 = s[1,0], s[1,1], s[1,2]
        s31, s32, s33 = s[2,0], s[2,1], s[2,2]
        
        # General formula involving second invariant of deviatoric stress
        # sigma_vm = sqrt( 0.5 * [ (s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2 + 6*(s12^2 + s23^2 + s31^2) ] )
        
        term1 = (s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2
        term2 = 6 * (s12**2 + s23**2 + s13**2)
        
        vm = np.sqrt(0.5 * (term1 + term2))
        return vm
