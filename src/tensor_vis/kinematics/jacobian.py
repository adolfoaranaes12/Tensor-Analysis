
import numpy as np
from ..core.tensor import TensorField

def calculate_jacobian(field: TensorField) -> TensorField:
    """
    Calculates the Jacobian matrix of a vector field (Rank 1).
    Result is a Rank 2 TensorField.
    
    J_ij = d f_i / d x_j
    """
    if field.rank != 1:
        raise ValueError("Jacobian is typically defined for vector fields (Rank 1).")
    
    if field.data is None:
        raise ValueError("Field data is empty.")
    
    dim = field.dim
    grid_shape = field.grid_shape
    
    # Initialize Jacobian data: (dim, dim, ...)
    j_shape = (dim, dim) + grid_shape
    j_data = np.zeros(j_shape)
    
    # Calculate gradients
    # For each component i of the vector field, calculate gradient with respect to x_j
    
    # Calculate steps from bounds and shape
    steps = [(b[1] - b[0]) / (s - 1) for b, s in zip(field.domain_bounds, grid_shape)]
    
    for i in range(dim):
        component = field.data[i]
        # np.gradient returns [d/d_axis0, d/d_axis1, ...]
        grads = np.gradient(component, *steps)
        
        for j in range(dim):
            j_data[i, j] = grads[j]
            
    # Create new TensorField for the Jacobian
    jacobian = TensorField(grid_shape, field.domain_bounds, rank=2)
    jacobian.set_data(j_data)
    
    return jacobian
