# Hybrid Inverse PINN Framework for Stratified Two-Phase Flow

**Title:** *Reduced-Order Turbulence-Closure Discovery in Smooth-Interface Stratified Gas–Liquid Flow Using a Hybrid Inverse Physics-Informed Neural-Network Framework*

---

## Abstract

Accurate reduced-order prediction of stratified gas–liquid flow in horizontal pipes is strongly limited by the closure of unresolved wall and interfacial momentum-transfer mechanisms. This work develops a **hybrid inverse physics-informed neural-network (PINN) framework** for reduced-order turbulence-closure discovery in smooth, fully developed stratified air–water pipe flow. The closure correlations are derived from eight operating conditions with $Re_L = 3450$–$5310$ and $Re_g = 2444$–$10134$, and are intended for this smooth-interface reduced-flow envelope rather than as universal turbulence laws.

Separate gas and liquid neural networks infer nondimensional axial midplane velocity profiles and the corresponding effective turbulent-viscosity fields. The inverse problem is constrained by the measured pressure-gradient forcing, wall no-slip conditions, phase mean-flow constraints, and dimensional velocity and shear-stress continuity at the gas–liquid interface. The inferred turbulent-viscosity fields are compressed into compact algebraic closures, correlated with Reynolds-number-based variables, and inserted into an independent finite-difference reduced solver for validation — without reuse of PINN outputs.

---

## Overview

The framework bridges the gap between experiments (reliable global quantities), CFD (expensive but model-dependent local fields), and classical one-dimensional two-fluid models (fast but closure-limited). It consists of three stages:

```
CFD data (8 cases, Polimi CFD study)
      │
      ▼
Stage 1 — Hybrid Inverse PINN  [HybridInversePINN.ipynb]
  ├─ Geometry: interface location, hydraulic diameters, Re numbers (from void fraction + superficial velocities)
  ├─ Infer U(y) and ν_t(y) for liquid and gas phases simultaneously
  │   ├─ Adam optimizer — 15,000 iterations (staged loss weighting)
  │   └─ L-BFGS — 500 iterations (residual refinement)
  └─ Export: normalized ν_t profiles per case → CSV
      │
      ▼
Stage 2 — Closure Extraction  [EquationDiscovery.ipynb]
  ├─ Liquid:  ν_{t,L}/ν_{t,L}^I = η²[1 + C_L(η(1−η))²],  C_L = f(Re_L)
  ├─ Gas:     two-branch blended model,  C_1, C_2 = f(Re_g/Re_L)
  └─ LOOCV regression over 6–8 cases per parameter
      │
      ▼
Stage 3 — Independent Validation  [CFDValidation.ipynb]
  ├─ Finite-difference solver: d/dY[(1/Re + ν_t)dU/dY] = dP/dX
  ├─ Discovered closure vs. mixing-length baseline (Van Driest damping)
  └─ RMSE, MAE vs. Polimi CFD reference velocity profiles
```

The PINN is used as a **closure-discovery tool**, not as a final black-box predictor. The discovered closure can be reused in an independent reduced solver that does not call the neural network at all.

---

## Physical Problem

The model considers **horizontal air–water stratified pipe flow** (D = 60 mm). Gravity separates the phases: liquid in the lower layer, gas in the upper layer, both driven by the same measured streamwise pressure gradient.

### Key assumptions

- Steady, fully developed, flat-interface flow along the pipe midplane
- The effective turbulent viscosity ν_t absorbs unresolved turbulent transport, secondary-flow effects, and cross-sectional redistribution; it should be interpreted as a **reduced closure field**, not a universal three-dimensional eddy viscosity
- Interface position is fixed from the measured void fraction
- No phase change, entrainment, or droplet/bubble transport

### Governing equation (each phase)

$$\frac{dP_i}{dX_i} = \frac{d}{dY_i}\!\left[\left(\frac{1}{Re_i} + \nu_{t,i}\right)\frac{dU_i}{dY_i}\right]$$

where $Y_i = y_i / D_{h,i}$ and $U_i = u_i / V_{ave,i}$ are nondimensional coordinates and velocities.

### Interface coupling (dimensional)

| Condition | Expression |
|-----------|-----------|
| Velocity continuity | $U_L V_{ave,L} = U_g V_{ave,g}$ at interface |
| Shear-stress continuity | $\rho_L V_{ave,L}^2\!\left(\tfrac{1}{Re_L}+\nu_{t,L}\right)\tfrac{dU_L}{dY_L} = \rho_g V_{ave,g}^2\!\left(\tfrac{1}{Re_g}+\nu_{t,g}\right)\tfrac{dU_g}{dY_g}$ |

No empirical interfacial friction factor is required; the coupling is enforced through these physical continuity conditions.

---

## Experimental Basis

Operating conditions from the Politecnico di Milano (Polimi) air–water stratified-flow experiments (Carraretto et al. [1]) and the corresponding VOF-based CFD database (Passoni et al. [2]).

### Table 1 — Operating conditions

| Case | D [m] | J_g [m/s] | J_L [m/s] | ε [–] | −dP/dx [Pa/m] |
|:----:|:-----:|:---------:|:---------:|:-----:|:-------------:|
| 1    | 0.060 | 0.551     | 0.040     | 0.705 | 0.486         |
| 2    | 0.060 | 1.308     | 0.040     | 0.743 | 1.958         |
| 3    | 0.060 | 1.538     | 0.040     | 0.753 | 2.650         |
| 4    | 0.060 | 2.306     | 0.040     | 0.772 | 5.010         |
| 5    | 0.060 | 0.551     | 0.060     | 0.623 | 0.670         |
| 6    | 0.060 | 1.308     | 0.060     | 0.698 | 2.631         |
| 7    | 0.060 | 1.538     | 0.060     | 0.708 | 3.418         |
| 8    | 0.060 | 2.306     | 0.060     | 0.728 | 6.105         |

### Table 2 — Derived geometry and Reynolds numbers

| Case | D_{h,L} [mm] | D_{h,g} [mm] | y_I [mm] | V_{ave,L} [m/s] | V_{ave,g} [m/s] | Re_L | Re_g |
|:----:|:------------:|:------------:|:--------:|:---------------:|:---------------:|:----:|:----:|
| 1    | 25.49        | 46.63        | −9.84    | 0.136           | 0.782           | 3450 | 2444 |
| 2    | 23.20        | 48.40        | −11.76   | 0.156           | 1.760           | 3605 | 5714 |
| 3    | 22.58        | 48.86        | −12.27   | 0.162           | 2.042           | 3650 | 6693 |
| 4    | 21.38        | 49.74        | −13.26   | 0.175           | 2.987           | 3745 | 9963 |
| 5    | 30.16        | 42.74        | −5.83    | 0.159           | 0.884           | 4792 | 2535 |
| 6    | 25.90        | 46.31        | −9.49    | 0.199           | 1.874           | 5137 | 5819 |
| 7    | 25.32        | 46.77        | −9.99    | 0.205           | 2.172           | 5192 | 6814 |
| 8    | 24.12        | 47.71        | −11.00   | 0.221           | 3.168           | 5310 | 10134|

---

## Hybrid Inverse PINN Methodology

### Neural-network architecture

Both liquid and gas networks share the same structure:

| Property | Value |
|----------|-------|
| Input | Nondimensional phase coordinate $Y_i$ |
| Hidden layers | 3 × 32 neurons, tanh activation |
| Outputs | $U_i(Y_i)$, $\nu_{t,i}(Y_i)$ |
| Initialization | Xavier (normal) |
| ν_t positivity | softplus transformation on the raw ν_t output |
| Training | Separate liquid and gas networks per case |

### Composite loss function

$$\mathcal{L} = \lambda_{mean}(\mathcal{L}_{mean,L}+\mathcal{L}_{mean,g}) + \lambda_{mom}(\mathcal{L}_{mom,L}+\mathcal{L}_{mom,g}) + \lambda_{data}(\mathcal{L}_{data,L}+\mathcal{L}_{data,g}) + \lambda_{wall,U}(\mathcal{L}_{wall,U,L}+\mathcal{L}_{wall,U,g}) + \lambda_{wall,\nu}(\mathcal{L}_{wall,\nu,L}+\mathcal{L}_{wall,\nu,g}) + \lambda_{int,u}\mathcal{L}_{int,u} + \lambda_{int,\tau}\mathcal{L}_{int,\tau}$$

| Loss term | Physical meaning | Initial weight | Final weight |
|-----------|-----------------|:--------------:|:------------:|
| $\mathcal{L}_{mean}$ | Phase mean-flow consistency (∫U dY = 1) | 1000 | 1000 |
| $\mathcal{L}_{mom}$ | Momentum PDE residual | 0.001 | 1000 |
| $\mathcal{L}_{data}$ | Sparse CFD velocity matching | 1000 | 1000 |
| $\mathcal{L}_{wall,U}$ | No-slip at wall (U = 0) | 1000 | 1000 |
| $\mathcal{L}_{wall,\nu}$ | ν_t = 0 at wall (soft penalty) | 1000 | 1000 |
| $\mathcal{L}_{int,u}$ | Dimensional interface velocity continuity | 1.0 | 1000 |
| $\mathcal{L}_{int,τ}$ | Dimensional interface shear-stress continuity | 0.001 | 1000 |

The momentum and interface losses are ramped up during training (scheduled weighting) to prevent trivial solutions and enforce physical consistency progressively.

### Optimization procedure

| Stage | Optimizer | Iterations | Notes |
|-------|-----------|:----------:|-------|
| 1 | Adam | 15,000 | lr = 1×10⁻³, StepLR (step=1500, γ=0.85), grad clip norm 1.0 |
| 2 | L-BFGS | 500 | lr = 1.0, history size 50, strong-Wolfe line search |

Training is performed **independently for each case**. Liquid and gas networks are optimized simultaneously because the interface losses couple their predictions.

### Collocation points

- 10,000 uniformly distributed collocation points per phase per case
- Boundary and interface points handled separately

---

## Discovered Closure Models

### Liquid phase

The inferred liquid ν_t grows monotonically from zero at the wall to a finite value at the interface:

$$\frac{\nu_{t,L}}{\nu_{t,L}^{I}} = \eta_L^2 \left[1 + C_L \left(\eta_L(1-\eta_L)\right)^2\right]$$

where $\eta_L = d_{wall,L}/\delta_L \in [0,1]$ (0 = wall, 1 = interface) and:

$$C_L = 0.011744\, Re_L - 38.672$$

*(Cases 1 and 5 excluded from the regression — see below.)*

### Gas phase

The inferred gas ν_t grows from the gas wall, reaches an interior peak, then decreases to a finite interfacial value. A two-branch blended form is used:

$$\frac{\nu_{t,g}}{\nu_{t,g}^{P}} = (1-\alpha)\, b_{wall} + \alpha\, b_{interface}$$

**Wall-side branch** ($b_{wall}$, zero at gas wall, unity at peak):
$$b_{wall} = \left(\frac{\eta_g}{\eta_g^P}\right)^2 \left(1 + C_{1,g}\frac{\eta_g(\eta_g^P - \eta_g)}{{\eta_g^P}^2}\right)$$

**Interface-side branch** ($b_{interface}$, unity at peak, $\nu_{t,g}^I/\nu_{t,g}^P$ at interface):
$$b_{interface} = \frac{\nu_{t,g}^I}{\nu_{t,g}^P} + \left(1 - \frac{\nu_{t,g}^I}{\nu_{t,g}^P}\right)\left(\frac{1-\eta_g}{1-\eta_g^P}\right)^2 \left(1 + C_{2,g}\frac{(\eta_g^P-\eta_g)(1-\eta_g)}{(1-\eta_g^P)^2}\right)$$

**Blending function**:
$$\alpha = \frac{\eta_g^2}{\eta_g^2 + {\eta_g^P}^2\!\left(\frac{1-\eta_g}{1-\eta_g^P}\right)^2}$$

**Gas closure correlations** (as functions of $X = \frac{\nu_{t,g}^P - \nu_{t,g}^I}{\nu_{t,g}^P}\cdot\frac{Re_g}{Re_L}$):

| Parameter | Correlation |
|-----------|-------------|
| Interface-to-peak ratio | $\nu_{t,g}^I / \nu_{t,g}^P = 0.207886\,(Re_g/Re_L) - 0.056330$ |
| Peak location | $\eta_g^P = -0.268621\,X + 0.645132$ |
| Wall-side curvature | $C_{1,g} = 2.313850\,X - 3.359563$ |
| Interface-side curvature | $C_{2,g} = 12.055194\,X - 6.164560$ |

*(Cases 1 and 5 excluded from all gas regressions; Case 8 additionally excluded from C₂_g only.)*

### Scope of the closure correlations

The closures are **interpolation laws** within the investigated operating envelope, not universal design correlations. They should not be extrapolated outside:

- $Re_L \approx 3450$–$5310$
- $Re_g \approx 2444$–$10134$
- Smooth or near-flat gas–liquid interface
- Horizontal circular pipe, D = 60 mm, air–water

### Why cases 1 and 5 are excluded

- **Case 1** ($J_g$ = 0.551 m/s, $J_L$ = 0.040 m/s): the CFD over-predicts the experimental pressure gradient, while all other seven cases show systematic CFD under-prediction. This opposite inconsistency alters the inverse problem in a fundamentally different way.
- **Case 5** ($J_g$ = 0.551 m/s, $J_L$ = 0.060 m/s): the CFD void-fraction error is ~10%, the largest among all cases. Since void fraction determines the entire reduced geometry (interface location, hydraulic diameters, Re numbers), this error undermines the geometric foundation of the geometry-aware closure.

---

## Finite-Difference Validation Solver

The discovered closure is inserted into an **independent finite-difference solver** that does not use automatic differentiation or PINN outputs. It solves the same reduced momentum equations over a 400-point interior grid per phase.

**Unknowns solved nonlinearly:**

$$\boldsymbol{\chi} = \left[\phi_L^I,\ \phi_g^I,\ \nu_{t,L}^I,\ \nu_{t,g}^I\right]^T$$

where $\phi_i^I = (1/Re_i + \nu_{t,i}^I)\, dU_i/dY_i\big|_I$ are Neumann interface-flux boundary conditions and $\nu_{t,i}^I$ are the interfacial turbulent-viscosity amplitudes (solved for in log-space to ensure positivity).

The nonlinear system is solved with `scipy.optimize.root` (hybr method), enforcing: liquid mean-flow constraint, gas mean-flow constraint, interface velocity continuity, and interface shear continuity.

**Baseline comparison:** an independent mixing-length model (Van Driest damping, κ = 0.41, A⁺ = 26.0) is run alongside the discovered closure for benchmarking.

---

## Project Structure

```
├── HybridInversePINN.ipynb      # PINN training and ν_t extraction
├── EquationDiscovery.ipynb      # Closure model fitting and parametric correlation
├── CFDValidation.ipynb          # Finite-difference validation and benchmark
└── CFDdata/
    ├── Liquid_case_1.csv        # CFD midplane velocity profiles — liquid phase
    ├── ...
    ├── Liquid_case_8.csv
    ├── Gas_case_1.csv           # CFD midplane velocity profiles — gas phase
    └── Gas_case_8.csv
```

### Generated outputs

| File | Description |
|------|-------------|
| `liquid_dataset_geometry_nut_norm.csv` | Normalized ν_t profiles, liquid (200 pts × 8 cases) |
| `gas_dataset_geometry_nut_norm.csv` | Normalized ν_t profiles, gas (200 pts × 8 cases) |
| `liquid_A_per_case_sqeta_model.csv` | Fitted C_L per case |
| `gas_C1_C2_fit_summary.csv` | Gas curvature coefficients per case |
| `liquid_gas_closure_metrics.csv` | RMSE / MAE / R² for both phases |

Model checkpoints (PyTorch `.pt`) are saved to `StratifiedWeights/` (configurable; default targets Google Drive for Colab use).

---

## Dependencies

```
torch
numpy
scipy
pandas
matplotlib
scikit-learn
```

```bash
pip install torch numpy scipy pandas matplotlib scikit-learn
```

GPU-accelerated training is supported (`torch.device('cuda')`); CPU execution is slower but works.

---

## Usage

Run the three notebooks in order:

1. **`HybridInversePINN.ipynb`** — Set the case index `i_LG` (0–7), then run all cells. Trains the PINN for one operating case and exports normalized ν_t profiles to CSV.
2. **`EquationDiscovery.ipynb`** — Reads the exported CSVs, fits the liquid and gas closure models, computes regression metrics, and exports coefficient tables.
3. **`CFDValidation.ipynb`** — Deploys the discovered closures in the finite-difference solver and compares against CFD reference velocity profiles and the mixing-length baseline.

---

## References

[1] Carraretto, I.M., et al. — Politecnico di Milano air–water horizontal stratified-flow experiments.

[2] Passoni, M., et al. — VOF-based multiphase CFD study of horizontal stratified gas–liquid flow (*Polimi CFD study*).

[3] Taitel, Y., Dukler, A.E. — *AIChE J.*, 22(1), 47–55, 1976.

[4] Shoham, O. — *Mechanistic Modeling of Gas–Liquid Two-Phase Flow in Pipes*, SPE, 2006.

[5] Agrawal, S.S., et al. — *AIChE J.*, 19(3), 501–507, 1973.

[6] Ullmann, A., Brauner, N. — *Multiphase Sci. Technol.*, 18(2), 2006.

[7] Ullmann, A., et al. — *Int. J. Multiphase Flow*, 29(10), 1607–1624, 2003.

[15] Raissi, M., et al. — *J. Comput. Phys.*, 378, 686–707, 2019.

[16] Karniadakis, G.E., et al. — *Nat. Rev. Phys.*, 3(6), 422–440, 2021.

[18] Eivazi, H., et al. — *Phys. Fluids*, 34(6), 065117, 2022.

[52] Parish, E.J., Duraisamy, K. — *J. Comput. Phys.*, 305, 758–774, 2016.
