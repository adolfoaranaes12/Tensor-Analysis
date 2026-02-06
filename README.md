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

## Visualizations Guide (Detailed)
1. `01_kinematics_demo.py` — Global flow with a moving probe. Left shows streamlines and the particle path. Right shows a local circle and `J · dr`, i.e., the relative velocity of neighbors (the linearized local action of the field).
2. `02_split_space_viz.py` — Conceptual 6D visualization that stacks velocity and vorticity into one state vector. This is not a physical time evolution, but a geometric picture of “velocity vs local rotation” at a point.
3. `03_tensor_probe.py` — A probe moving on a path while the local `J` is sampled. Green arrows are `J · dr` (relative velocity), so you see rotation‑dominated vs stretch‑dominated regions.
4. `04_tensor_glyphs.py` — Symmetric part `D` is rendered as ellipses. Eigenvectors give axes; eigenvalues give stretching/compression. Color indicates shear magnitude.
5. `05_kinematics_decomposition.py` — Toy 3D object showing translation, strain, rotation, and their combination. This is conceptual, not tied to a specific velocity field.
6. `05_1_reconstructed_kinematics.py` — Same decomposition as above, but with solid surfaces for clearer shape change.
7. `06_field_decomposition.py` — Field‑level decomposition: principal strain directions (from `D`) vs vorticity magnitude (from `W`). This shows where the flow stretches versus spins.
8. `07_full_field_and_gradient.py` — Side‑by‑side comparison: velocity field vs gradient glyphs. The right panel shows how `∇v` changes locally, not where particles move.
9. `08_basis_transformation.py` — Columns of `J` are visualized as transformed basis vectors (`∂v/∂x`, `∂v/∂y`). This makes the Jacobian’s linear map explicit.
10. `09_principal_directions.py` — Eigenvectors of `D` at each point. Red shows max extension direction; blue shows max compression.
11. `10_em_stress_tensor.py` — Maxwell stress tensor example (electrostatics). Red axis aligns with field lines (tension), blue shows perpendicular pressure.
12. `11_convective_acceleration.py` — Shows `a = J · v` and its split into `D · v` (strain contribution) and `W · v` (rotational contribution).
13. `12_taylor_expansion.py` — Compares the true velocity field against the local linear approximation `v0 + J · dr`. Shows where linearization is accurate.
14. `13_strain_decomposition.py` — Matrix‑level decomposition: `L = W + D` and `D = D_vol + D_dev`. Visualizes each part acting on a unit circle.
15. `14_jacobian_validation.py` — Numeric vs analytic Jacobian for Taylor‑Green vortex. Validates the sign conventions and accuracy.
16. `15_strain_only_acceleration.py` — Animation of strain‑only action on velocity: `a_strain = D · v` along a particle path.
17. `16_strain_tensor_field.py` — Symmetric tensor field view: principal directions plus animated `F = D · v` at the probe.

## Suggested Path (Kinematics Focus)
1. `01_kinematics_demo.py`
2. `03_tensor_probe.py`
3. `04_tensor_glyphs.py`
4. `06_field_decomposition.py`
5. `08_basis_transformation.py`
6. `09_principal_directions.py`
7. `11_convective_acceleration.py`
8. `12_taylor_expansion.py`
9. `13_strain_decomposition.py`
10. `15_strain_only_acceleration.py`
11. `16_strain_tensor_field.py`

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

## References
https://gemini.google.com/app/23e6e82426df81b3

https://gemini.google.com/app/291179c21c83738f

https://www.youtube.com/watch?v=YxXyN2ifK8A&list=PL2aHrV9pFqNTEMuDFre16Wx2SwBCNiR7j