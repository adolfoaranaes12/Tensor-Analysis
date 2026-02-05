# Tensor Analysis & Visualization Project

## Overview
This project provides a suite of interactive Python scripts (`notebooks/`) to visualize and understand Second-Rank Tensors in Kinematics (Fluid Dynamics) and Physics (Electromagnetism). 

It focuses on interpreting the **Gradient Tensor** ($\nabla \mathbf{v}$) and the **Stress Tensor** ($\mathbf{\sigma}$) through their geometric properties:
1.  **Decomposition**: Symmetric (Strain) vs Antisymmetric (Rotation).
2.  **Principal Axes**: Directions of maximum stretching/tension.
3.  **Basis Transformation**: How the tensor maps space locally.

## Visualizations Guide

### 1. Kinematics Demo (`01_kinematics_demo.py`)
-   **Concept**: Phase Space & Velocity Field.
-   **Math**: $\mathbf{v}(\mathbf{x})$.
-   **Interpretation**:
    -   **Quiver Plot**: Shows where particles flow.
    -   **Jacobian Matrix**: Shows the raw 4 components ($dv_x/dx$, etc.) which are hard to interpret directly.

### 2. Split Space Viz (`02_split_space_viz.py`)
-   **Concept**: 6D Interaction.
-   **Math**: $\mathbf{v}$ (Velocity) vs $\mathbf{\omega}$ (Vorticity).
-   **Interpretation**:
    -   **Left Cube**: Translation (Linear motion).
    -   **Right Cube**: Rotation (Angular motion).
    -   Shows how one point in flow has both linear and rotational properties.

### 3. Tensor Probe (`03_tensor_probe.py`)
-   **Concept**: Local Deformation (The "Derivative").
-   **Math**: $\delta \mathbf{v} = \nabla \mathbf{v} \cdot \delta \mathbf{r}$.
-   **Interpretation**:
    -   **Green Arrows**: The relative velocity of neighbors around the probe.
    -   If they **spin**: The field has Curl/Vorticity.
    -   If they **stretch**: The field has Divergence/Strain.

### 4. Tensor Glyphs (`04_tensor_glyphs.py`)
-   **Concept**: **Symmetric Tensor Field** (Strain Rate).
-   **Math**: $\mathbf{D} = \frac{1}{2}(\nabla \mathbf{v} + \nabla \mathbf{v}^T)$.
-   **Interpretation**:
    -   **Ellipsoids**: Show how a circular droplet would deform.
    -   **Long Axis**: Direction of Stretching.
    -   **Short Axis**: Direction of Squeezing.
    -   **Color**: Intensity of Shear (Shape change).

### 5. Kinematics Decomposition (`05_kinematics_decomposition.py`)
-   **Concept**: **Direct Sum** ($\mathbf{L} = \mathbf{D} \oplus \mathbf{W}$).
-   **Interpretation**:
    -   **Cyan Sphere**: Base material element.
    -   **Orange**: Effect of **D** alone (Stretches only).
    -   **Green**: Effect of **W** alone (Rotates only).
    -   **Magenta**: Combined effect (Real fluid motion).

### 5.1 Reconstructed Kinematics (`05_1_reconstructed_kinematics.py`)
-   **Concept**: Enhanced Interactive Simulation.
-   **Interpretation**:
    -   **Panel 1 (Cyan)**: Static Initial State (Reference).
    -   **Panel 2 (Orange)**: Strain Component (Solid Ellipsoid).
    -   **Panel 3 (Lime)**: Rotation Component (Spinning Sphere).
    -   **Panel 4 (Magenta)**: Total Motion (Combined).

### 6. Field Decomposition (`06_field_decomposition.py`)
-   **Concept**: Decomposition across the whole map.
-   **Interpretation**:
    -   **Left (Red Arrows)**: The "Strain Field". Shows the axis of principal pull.
    -   **Right (Color)**: The "Vorticity Field". Shows where the flow spins.

### 7. Full Field Comparison (`07_full_field_and_gradient.py`)
-   **Concept**: Vector Field vs Tensor Field.
-   **Interpretation**:
    -   **Left**: Where the flow is going ($\mathbf{v}$).
    -   **Right**: How the flow is deforming ($\nabla \mathbf{v}$).

### 8. Basis Transformation (`08_basis_transformation.py`)
-   **Concept**: Linear Algebra Mapping.
-   **Math**: $\mathbf{J} \cdot \hat{e}_i$.
-   **Interpretation**:
    -   **Dashed Arrows**: Standard Basis (x, y).
    -   **Solid Arrows**: How the Tensor "bends" the basis locally.

### 9. Principal Directions (`09_principal_directions.py`)
-   **Concept**: Eigenvectors (The "Tensor Gradient").
-   **Interpretation**:
    -   **Red Cross**: Axis of Max Extension.
    -   **Blue Cross**: Axis of Max Compression.
    -   This is the tensor equivalent of "Slope" for scalar fields.

### 10. EM Stress Tensor (`10_em_stress_tensor.py`)
-   **Concept**: **Universality**.
-   **Math**: Maxwell Stress Tensor.
-   **Interpretation**:
    -   **Red Axis**: Tension. Notice it aligns **perfectly** with Electric Field lines.
    -   **Blue Axis**: Pressure. Perpendicular to field lines.
    -   *Conclusion*: Electric field lines represent the "tension" axis of the underlying electromagnetic stress.

### 11. Convective Acceleration (`11_convective_acceleration.py`)
-   **Concept**: **Dynamics / Forces** (The "Self-Force").
-   **Math**: $\mathbf{a} = \mathbf{L} \cdot \mathbf{v} = \mathbf{D}\cdot\mathbf{v} + \mathbf{W}\cdot\mathbf{v}$.
-   **Interpretation**:
    -   **Green Arrows (Strain)**: The force that changes your **Speed**. (Like the Electric Force).
    -   **Blue Arrows (Rotation)**: The force that changes your **Direction**. (Like the Magnetic Force, always perpendicular to v).
    -   Combined, they dictate the inertial path of the fluid.

### 12. Taylor Expansion (`12_taylor_expansion.py`)
-   **Concept**: **Local Reconstruction** (The "Linearizer").
-   **Math**: $\mathbf{v}(\mathbf{x}) \approx \mathbf{v}(\mathbf{x}_0) + \mathbf{J} \cdot (\mathbf{x} - \mathbf{x}_0)$.
-   **Visual**: Compares the **True Field** (with curvature) against the **Linear Model** (Straight lines constructed from the Tensor).
-   **Insight**: Shows how the Tensor Field allows you to rebuild the vector field locally, with errors growing as you move further from the center.

### 13. Strain Decomposition (`13_strain_decomposition.py`)
-   **Concept**: **The Anatomy of the Gradient Tensor**.
-   **Math**: $\mathbf{L} = \mathbf{W} + \mathbf{V}_{vol} + \mathbf{V}_{dev}$.
-   **Visual**: Displays the mathematical matrices as **Vector Fields**.
-   **Components**:
    -   **Blue**: Rotation (Spin).
    -   **Green**: Volumetric Strain (Explosion/Implosion).
    -   **Red**: Deviatoric Strain (Pure Shape Distortion).

## How to Run
Activate the environment and run any script:
```bash
./venv/bin/python notebooks/01_kinematics_demo.py
# ... etc
```
