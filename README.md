# Tensor Analysis & Visualization Project

## Overview
This project is a suite of interactive scripts in `notebooks/` for building intuition about tensor fields in fluid kinematics and physics. The primary focus is the velocity gradient (Jacobian) and its geometric meaning.

Core ideas:
1. Decomposition of the gradient into strain and rotation.
2. Principal directions and eigen-structure.
3. Local linearization and basis mapping.

## Conventions
The code consistently uses:
1. Jacobian definition: `J_ij = ∂v_i / ∂x_j`.
2. Decomposition: `D = 0.5 (J + J^T)` and `W = 0.5 (J - J^T)`.
3. 2D vorticity: `ω_z = ∂v_y/∂x - ∂v_x/∂y`.

Probe notebooks use bilinear interpolation to reduce sampling artifacts.

## Visualizations Guide
1. `01_kinematics_demo.py` — Global flow + local Jacobian probe (particle experiences `J · dr`).
2. `02_split_space_viz.py` — Conceptual 6D view (velocity vs vorticity at a point).
3. `03_tensor_probe.py` — Local deformation probe (`J · dr` as relative velocity).
4. `04_tensor_glyphs.py` — Symmetric part `D` shown as ellipses.
5. `05_kinematics_decomposition.py` — Translation, strain, rotation, combined.
6. `05_1_reconstructed_kinematics.py` — Same decomposition with solid surfaces.
7. `06_field_decomposition.py` — Strain axis field vs vorticity magnitude map.
8. `07_full_field_and_gradient.py` — Velocity field vs gradient glyphs.
9. `08_basis_transformation.py` — Basis vectors mapped by the Jacobian.
10. `09_principal_directions.py` — Max stretch and compression directions.
11. `10_em_stress_tensor.py` — Maxwell stress tensor visualization (electrostatics).
12. `11_convective_acceleration.py` — `a = J · v` and decomposition into `D · v` and `W · v`.
13. `12_taylor_expansion.py` — Local linear reconstruction of a curved field.
14. `13_strain_decomposition.py` — `L = W + D_vol + D_dev` decomposition.
15. `14_jacobian_validation.py` — Numeric vs analytic Jacobian comparison (Taylor–Green).

## Validation
The Jacobian implementation is validated against an analytic Taylor–Green field.
1. Unit test: `tests/test_math.py::test_jacobian_taylor_green_accuracy`.
2. Visual check: `notebooks/14_jacobian_validation.py`.

Finite difference errors are concentrated near boundaries due to one‑sided derivatives. Interior accuracy is high enough for visualization and intuition building.

## How to Run
Activate the environment and run any script:
```bash
./venv/bin/python notebooks/01_kinematics_demo.py
```

## Tests
```bash
./venv/bin/pytest -q
```

## Notes and Limitations
This repo is designed for intuition and visualization, not high‑precision CFD.
1. The kinematics notebooks are reliable for interpreting local strain/rotation and phase‑space intuition.
2. Some advanced modules are partial or placeholder (e.g., EM tensor invariants, velocity divergence helpers).

## Reliability and Scope
If your goal is to build intuition about the kinematics of fluid motion and the geometry of the Jacobian, this repo is a good fit. The visualizations are internally consistent with the standard definitions and validated against an analytic field. Dynamics‑level predictions (e.g., Navier–Stokes evolution or quantitative CFD) are outside scope and intentionally not emphasized here.
