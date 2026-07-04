# Evaluation Metrics Reference

## Tier A — Common Metrics (All Models)

These metrics apply to **all** model architectures and form the main comparison table.

### 1. State Error (Core Accuracy)

#### MAE (Mean Absolute Error)

$$\text{MAE}_s = \frac{1}{T} \sum_{t=1}^{T} \|\hat{s}_t - s_t\|_1$$

#### RMSE (Root Mean Square Error)

$$\text{RMSE}_s = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \|\hat{s}_t - s_t\|_2^2}$$

**Usage**: Measures trajectory fidelity. Used by Breen et al. (MAE), Zhong et al., HNN benchmarks.

---

### 2. Relative Rollout Error (Chaos-Aware)

#### Per-Step Relative Error

$$\text{RelErr}_s(t) = \frac{\|\hat{s}_t - s_t\|_2}{\|s_t\|_2}$$

#### Time-Averaged Rollout Error

$$\text{RolloutErr}_{\text{avg}} = \frac{1}{T} \sum_{t=1}^{T} \frac{\|\hat{s}_t - s_t\|_2}{\|s_t\|_2}$$

**Usage**: Standard in Zhong et al. Scales naturally as chaos amplifies errors.

---

### 3. Energy Behavior (Model-Agnostic)

#### Absolute Energy Error

$$\text{AbsEnergyErr}(t) = |\hat{E}_t - E_t|$$

#### Energy Drift

$$\text{EnergyDrift} = \max_{1 \leq t \leq T} |\hat{E}_t - \hat{E}_0|$$

**Usage**: Applies to DNN, PINN, HNN, LNN equally. Plot energy vs time and report drift as scalar.

---

### 4. Long-Horizon Rollout Error (Generalization)

Pick horizon $T^* > T_{\text{train}}$:

$$\text{RolloutErr}(T^*) = \mathbb{E}_{\text{IC}} \left[ \frac{\|\hat{s}_{T^*} - s_{T^*}\|_2}{\|s_{T^*}\|_2} \right]$$

**Usage**: Tests extrapolation beyond training horizon. Central to chaos-robust evaluation.

---

### Tier A Summary Table

| Metric                     | DNN | PINN | LNN | HNN |
|----------------------------|-----|------|-----|-----|
| MAE / RMSE (state)         | ✓   | ✓    | ✓   | ✓   |
| Relative rollout error     | ✓   | ✓    | ✓   | ✓   |
| Energy error / drift       | ✓   | ✓    | ✓   | ✓   |
| Long-horizon rollout error | ✓   | ✓    | ✓   | ✓   |

---

## Tier B — Model-Specific Metrics (Diagnostics Only)

These go in **separate subsections**, never in the main comparison table.

### PINN-Specific

#### Physics Residual (ODE Residual)

$$R_{\text{ODE}}(t) = \|\ddot{q}_t^{(\text{pred})} - f(q_t, \dot{q}_t)\|_2$$

$$\text{ODERes} = \frac{1}{T} \sum_{t=1}^{T} R_{\text{ODE}}(t)$$

Measures equation compliance. Only meaningful for PINNs.

---

### LNN-Specific

#### Euler-Lagrange Residual

$$\mathcal{E}_i = \frac{d}{dt}\left(\frac{\partial \hat{L}}{\partial \dot{q}_i}\right) - \frac{\partial \hat{L}}{\partial q_i}$$

$$\text{ELRes} = \frac{1}{T} \sum_{t=1}^{T} \|\mathcal{E}(q_t, \dot{q}_t, \hat{a}_t)\|_2$$

#### Acceleration MSE

$$\text{MSE}_a = \frac{1}{T} \sum_{t=1}^{T} \|\hat{a}_t - a_t\|_2^2$$

Measures whether learned Lagrangian produces correct forces.

---

### HNN-Specific

#### Hamiltonian Vector-Field Residual

$$R_H(t) = \|J \nabla \hat{H}(s_t) - \dot{s}_t\|_2$$

$$\text{HRes} = \frac{1}{T} \sum_{t=1}^{T} R_H(t)$$

#### Hamiltonian MSE

$$\text{MSE}_H = \frac{1}{T} \sum_{t=1}^{T} (\hat{H}_t - H_t)^2$$

Tests whether learned Hamiltonian generates the motion.

---

## Implementation Notes

### Energy Computation (Three-Body Problem)

$$E = \underbrace{\frac{1}{2} \sum_{i=1}^{3} m_i \|\mathbf{v}_i\|^2}_{\text{Kinetic}} - \underbrace{G \sum_{i<j} \frac{m_i m_j}{\|\mathbf{r}_i - \mathbf{r}_j\|}}_{\text{Potential}}$$

```python
def compute_energy(positions, velocities, G=1.0, masses=[1,1,1]):
    # Kinetic: KE = 0.5 * Σ m_i * |v_i|²
    # Potential: PE = -G * Σ_{i<j} m_i * m_j / |r_i - r_j|
    return KE + PE
```

### State Vector Convention
- Positions: $[x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3]$ (9 components)
- Velocities: $[v_{x1}, v_{y1}, v_{z1}, v_{x2}, v_{y2}, v_{z2}, v_{x3}, v_{y3}, v_{z3}]$ (9 components)
- Full state: 18 components (but $y=0$ for planar motion)

### Reporting Guidelines
1. **Main Results Table**: Only Tier A metrics
2. **Energy & Rollout Plots**: All models included
3. **Model-Specific Diagnostics**: Separate subsection with disclaimer:
   > "These diagnostics apply only to models with an explicit physical structure."
