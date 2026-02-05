
import numpy as np
from typing import Optional, Tuple, Callable

class TensorField:
    """
    Base class for representing a tensor field.
    """
    def __init__(self, 
                 grid_shape: Tuple[int, ...], 
                 domain_bounds: Tuple[Tuple[float, float], ...],
                 rank: int = 2):
        """
        Initialize a TensorField.

        Args:
            grid_shape: The shape of the spatial grid (e.g., (100, 100) for 2D).
            domain_bounds: The physical bounds of the domain ((x_min, x_max), (y_min, y_max), ...).
            rank: The rank of the tensor (default 2 for second-rank tensors).
        """
        self.grid_shape = grid_shape
        self.domain_bounds = domain_bounds
        self.rank = rank
        self.dim = len(grid_shape)
        
        # Initialize meshgrid
        coords = [np.linspace(b[0], b[1], s) for b, s in zip(domain_bounds, grid_shape)]
        self.grid = np.meshgrid(*coords, indexing='ij')
        
        # Placeholder for field data: Shape (d, d, nx, ny...) for rank 2
        # For a general rank r tensor in d dimensions, simple shape is (d,)*rank + grid_shape
        self.data: Optional[np.ndarray] = None

    def set_data(self, data: np.ndarray):
        """Set the tensor field data directly."""
        expected_shape = (self.dim,) * self.rank + self.grid_shape
        if data.shape != expected_shape:
            raise ValueError(f"Data shape {data.shape} does not match expected shape {expected_shape}")
        self.data = data

    def from_function(self, func: Callable):
        """
        Initialize the field from a function f(x, y, ...) -> Tensor.
        func should accept coordinates and return a tensor of shape (d, d).
        """
        # specialized for 2D/3D rank-2 for efficiency could be added later
        # This is a slow, generic implementation
        # A vectorized approach would be better in subclasses
        pass

    @property
    def x(self):
        return self.grid[0]
    
    @property
    def y(self):
        return self.grid[1] if self.dim > 1 else None
    
    @property
    def z(self):
        return self.grid[2] if self.dim > 2 else None
