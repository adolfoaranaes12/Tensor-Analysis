
import sys
import os
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tensor_vis.core.tensor import TensorField
from tensor_vis.fluids.velocity_field import VelocityField
from tensor_vis.kinematics.jacobian import calculate_jacobian
from tensor_vis.continuum.stress import StressTensor
from tensor_vis.core.interp import bilinear_interpolate

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

def test_von_mises_2d_plane_stress():
    bounds = ((0, 1), (0, 1))
    shape = (3, 3)
    s_field = StressTensor(shape, bounds)

    # Constant 2D stress state
    s11 = 3.0
    s22 = 1.0
    s12 = 2.0
    data = np.zeros((2, 2) + shape)
    data[0, 0] = s11
    data[1, 1] = s22
    data[0, 1] = s12
    data[1, 0] = s12
    s_field.set_data(data)

    expected = np.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * (s12**2))
    vm = s_field.von_mises()
    assert np.allclose(vm, expected)

def test_von_mises_3d():
    bounds = ((0, 1), (0, 1), (0, 1))
    shape = (2, 2, 2)
    s_field = StressTensor(shape, bounds)

    # Constant 3D stress state
    s11, s22, s33 = 1.0, 2.0, 3.0
    s12, s23, s13 = 0.5, -1.0, 0.25
    data = np.zeros((3, 3) + shape)
    data[0, 0] = s11
    data[1, 1] = s22
    data[2, 2] = s33
    data[0, 1] = s12
    data[1, 0] = s12
    data[1, 2] = s23
    data[2, 1] = s23
    data[0, 2] = s13
    data[2, 0] = s13
    s_field.set_data(data)

    term1 = (s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2
    term2 = 6 * (s12 ** 2 + s23 ** 2 + s13 ** 2)
    expected = np.sqrt(0.5 * (term1 + term2))
    vm = s_field.von_mises()
    assert np.allclose(vm, expected)

def test_jacobian_taylor_green_accuracy():
    bounds = ((-np.pi, np.pi), (-np.pi, np.pi))
    shape = (64, 64)
    v_field = VelocityField(shape, bounds)

    x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
    y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Taylor-Green vortex (2D)
    Vx = np.sin(X) * np.cos(Y)
    Vy = -np.cos(X) * np.sin(Y)
    v_field.set_data(np.stack([Vx, Vy], axis=0))

    J = calculate_jacobian(v_field).data

    # Analytic Jacobian
    J_true = np.zeros_like(J)
    J_true[0, 0] = np.cos(X) * np.cos(Y)
    J_true[0, 1] = -np.sin(X) * np.sin(Y)
    J_true[1, 0] = np.sin(X) * np.sin(Y)
    J_true[1, 1] = -np.cos(X) * np.cos(Y)

    # Avoid edge effects from numerical differentiation
    sl = (slice(None), slice(None), slice(1, -1), slice(1, -1))
    max_err = np.max(np.abs(J[sl] - J_true[sl]))
    assert max_err < 5e-2

def test_bilinear_interpolate_linear_field():
    bounds = ((0.0, 1.0), (0.0, 1.0))
    shape = (3, 3)
    x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
    y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Linear field f(x, y) = x + 2y (bilinear interpolation should be exact)
    data = X + 2 * Y
    point = (0.3, 0.7)
    expected = point[0] + 2 * point[1]
    value = bilinear_interpolate(data, bounds, point)
    assert np.allclose(value, expected)
