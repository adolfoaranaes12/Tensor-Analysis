
import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tensor_vis.core.tensor import TensorField
from tensor_vis.fluids.velocity_field import VelocityField
from tensor_vis.kinematics.jacobian import calculate_jacobian

def test_tensor_initialization():
    bounds = ((-1, 1), (-1, 1))
    shape = (10, 10)
    tf = TensorField(shape, bounds, rank=2)
    assert tf.dim == 2
    assert tf.grid[0].shape == shape

def test_jacobian_identity():
    # Field v = (x, y)
    # dv_x/dx = 1, dv_x/dy = 0
    # dv_y/dx = 0, dv_y/dy = 1
    # Jacobian should be identity matrix everywhere
    
    bounds = ((-1, 1), (-1, 1))
    shape = (5, 5) # small grid
    v_field = VelocityField(shape, bounds)
    
    # Manually construction grid
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # rank 1 data: (dim, nx, ny)
    data = np.stack([X, Y], axis=0) 
    v_field.set_data(data)
    
    J = calculate_jacobian(v_field)
    
    # J shape: (2, 2, 5, 5)
    # J[0, 0] should be 1
    assert np.allclose(J.data[0, 0], 1.0)
    assert np.allclose(J.data[1, 1], 1.0)
    assert np.allclose(J.data[0, 1], 0.0)
    assert np.allclose(J.data[1, 0], 0.0)

def test_jacobian_shear():
    # Field v = (y, 0) (Simple Shear)
    # dv_x/dx = 0, dv_x/dy = 1
    # dv_y/dx = 0, dv_y/dy = 0
    
    bounds = ((0, 1), (0, 1))
    shape = (11, 11)
    v_field = VelocityField(shape, bounds)
    
    y = np.linspace(0, 1, 11)
    x = np.linspace(0, 1, 11) # not used directly
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    data = np.stack([Y, np.zeros_like(Y)], axis=0) # v_x = y, v_y = 0
    v_field.set_data(data)
    
    J = calculate_jacobian(v_field)
    
    # J = [[0, 1], [0, 0]]
    assert np.allclose(J.data[0, 1], 1.0)
    assert np.all(J.data[0, 0] == 0.0)
    assert np.all(J.data[1, 0] == 0.0)
    assert np.all(J.data[1, 1] == 0.0)
