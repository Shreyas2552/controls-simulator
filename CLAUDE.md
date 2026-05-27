# CLAUDE.md — Control Systems Simulator

AI developer reference. Read this before touching any file.

---

## Quick Start

```bash
# Install deps
pip install streamlit numpy scipy plotly

# Run locally (port 8501)
streamlit run app.py

# Run on alternate port
streamlit run app.py --server.port 8502
```

Deployed: `Shreyas2552/controls-simulator` → Streamlit Cloud auto-deploys on push to `master`.

---

## Repo Structure

```
app.py                          Landing page — navigation cards only
pages/
  1_Control_Simulator.py        PID tuning, Bode, root locus, Nyquist, stability, filters, theory
  2_LQR_LQG.py                  LQR optimal control + Kalman filter + LQG full loop
  3_MPC.py                      Model Predictive Control — finite-horizon, constrained
  4_Advanced_Control.py         Smith predictor, system ID, gain scheduling, interview Q&A
  5_Control_Strategies.py       Cascade, SIMC, feedforward, SMC/relay nonlinear simulations
modules/
  plants.py                     TF plant catalogue — get_plant_tf()
  pid_controller.py             PID TF + Tustin discretization
  analysis.py                   Step/ramp/bode/locus/stability/metrics
  filters.py                    Analog + digital filter design
  lqr_lqg.py                    State-space plants, LQR/Kalman Riccati, LQG simulation
  mpc.py                        MPC prediction matrices, unconstrained solver, simulation
  advanced_control.py           Cascade, SIMC/IMC, feedforward, SMC, relay simulation
  __init__.py
requirements.txt
CHANGELOG.md
```

---

## Architecture Rules

1. **No UI code in `modules/`** — all Streamlit calls live in `pages/` or `app.py`
2. **No finance code here** — see `Shreyas2552/portfolio-analyzer`
3. **`st.set_page_config()` must be first** in every page file before any `st.*` call
4. **NumPy 2.x**: use `.item()` not `float()` on arrays — `float(arr)` raises TypeError in NumPy 2.0+
5. **Shared colour palette**: import `C` dict and `PLOT_LAYOUT` pattern from the page-level constants (replicate in each page — no shared state between pages in Streamlit)

---

## Adding a New Plant (to PID simulator)

1. Open `modules/plants.py`
2. Add entry to `PLANT_MODELS` dict:

```python
"My Plant": {
    "tf_display":        "G(s) = ...",                   # LaTeX string shown in UI
    "physical_context":  "What physical system this is",  # caption under TF
    "params": {
        "K": {
            "label":   "DC Gain (K)",
            "default": 1.0,
            "min":     0.1,
            "max":     10.0,
            "step":    0.1,
        },
        # ... more params
    },
},
```

3. Add `if plant_name == "My Plant":` branch to `get_plant_tf()` returning `(num, den)` lists
4. No changes needed in the page — the sidebar loop reads `PLANT_MODELS` automatically

### Padé dead-time pattern

```python
if L < 1e-6:
    return [float(K)], [float(tau), 1.0]          # no delay
pade_n = np.array([-L/2, 1.0])
pade_d = np.array([ L/2, 1.0])
num = float(K) * pade_n
den = np.polymul(pade_d, [float(tau), 1.0])
return num.tolist(), den.tolist()
```

---

## Adding a New State-Space Plant (to LQR/LQG/MPC)

1. Open `modules/lqr_lqg.py`
2. Add entry to `SS_PLANTS` dict — required keys:
   - `desc`, `physical`, `n_states`, `n_inputs`, `n_outputs`
   - `state_names`, `input_names`, `output_names`, `ref_state`
   - `params` (same structure as plants.py)
   - `default_Q` (list, length = n_states), `default_R` (float)
3. Add branch to `get_ss()` returning `(A, B, C, D)` numpy arrays
4. The pages read `SS_PLANTS` automatically — no page changes needed

---

## Adding a New Page

1. Create `pages/N_Name.py` (Streamlit orders by prefix number)
2. First 3 lines must be:

```python
import streamlit as st
st.set_page_config(page_title="...", page_icon="...", layout="wide",
                   initial_sidebar_state="expanded")
```

3. Add a navigation card in `app.py`:

```python
with colN:
    st.markdown("### 🔧 Title")
    st.markdown("- Feature 1\n- Feature 2")
    if st.button("Open Page →", use_container_width=True, type="primary"):
        st.switch_page("pages/N_Name.py")
```

---

## Module APIs

### `modules/plants.py`
```python
PLANT_MODELS: Dict                                    # catalogue of TF plants
get_plant_tf(plant_name, params) → (num, den)         # highest-power-first lists
```

### `modules/pid_controller.py`
```python
get_pid_tf(Kp, Ki, Kd, N) → (num, den)               # continuous PID TF
discretize_tf(num, den, Ts) → (num_d, den_d)          # Tustin (bilinear)
```

### `modules/analysis.py`
```python
build_ol_cl(pn, pd, cn, cd) → (ol_num, ol_den, cl_num, cl_den)
step_response(num, den, t_end, n, discrete, Ts) → (t, y)
ramp_response(num, den, t_end, n) → (t, y)
control_signal_step(pn, pd, cn, cd, t_end, n) → (t, u)
bode_data(num, den, n, discrete, Ts) → (omega, mag_dB, phase_deg)
stability_margins(ol_num, ol_den, discrete, Ts) → {gm_db, pm_deg, wgc, wpc}
root_locus_data(ol_num, ol_den, n_gains) → (locus, ol_poles, ol_zeros)
cl_pole_analysis(cl_den) → List[Dict]                 # per-pole stability info
performance_metrics(t, y, ref) → Dict                 # OS%, rise/settle times
```

### `modules/filters.py`
```python
FILTER_TYPES, FILTER_FAMILIES                         # UI option lists
design_analog(ftype, family, order, wc, wc2, ripple) → (b, a)
design_digital(ftype, family, order, wn, wn2, ripple, fs) → (b, a)
filter_bode(b, a, analog, n) → (omega, mag_dB, phase_deg)
filter_step(b, a, analog, t_end, n) → (t, y)
```

### `modules/lqr_lqg.py`
```python
SS_PLANTS: Dict
get_ss(plant_name, params) → (A, B, C, D)
controllability_rank(A, B) → int
observability_rank(A, C) → int
lqr_design(A, B, Q, R) → (K, P, cl_eigs)
kalman_design(A, C, Qn, Rn) → (L, Pe)
compute_nbar(A, B, C_row, K) → float
simulate_lqr(A, B, C, K, Nbar, ref, t_end, ref_state) → (t, x, y, u)
simulate_lqg(A, B, C, C_obs, K, L, Nbar, ref, t_end, q_std, r_std, ref_state, seed) → (t, x_true, x_est, y_meas, u)
```

### `modules/mpc.py`
```python
discretize_ss(A, B, dt) → (Ad, Bd)                   # ZOH via matrix exponential
build_prediction_matrices(Ad, Bd, C, Np, Nc) → (Phi, Psi)
                                                      # Phi: (Np*ny × n), Psi: (Np*ny × Nc*nu)
solve_mpc_step(Phi, Psi, Q_bar, R_bar, x_k, ref_vec, u_min, u_max) → (U_star, constrained_bool)
simulate_mpc(Ad, Bd, C, Np, Nc, Q_y, R_u, ref, t_end, dt, ref_state,
             u_min, u_max, q_noise_std, seed)
    → (t, x_hist, y_hist, u_hist, constraint_hist, snapshots)
# snapshots: dict {time_idx: {t_pred, y_pred, u_pred}} for visualizing prediction window
```

### `modules/advanced_control.py`
```python
# Cascade control
simulate_cascade(K1, tau1, K2, tau2, Kp_out, Ki_out, Kp_in, Ki_in,
                 disturbance, dist_time, ref, t_end, dt)
    → (t, y1_cascade, y1_single_loop, u_cascade)

# SIMC / IMC tuning (return Kp, Ki, Kd)
simc_pid_fopdt(K, tau, theta, lambda_c) → (Kp, Ki, Kd=0)
simc_pid_2nd_order(K, tau1, tau2, theta, lambda_c) → (Kp, Ki, Kd)
zn_pid_fopdt(K, tau, theta) → (Kp, Ki, Kd)
simulate_pid_fopdt(K, tau, theta, Kp, Ki, Kd, ref, t_end, dt) → (t, y, u)

# Feedforward control
simulate_feedforward(K_G, tau_G, K_Gd, tau_Gd, ff_gain_mismatch,
                     Kp, Ki, dist_amp, dist_time, ref, t_end, dt)
    → (t, y_feedback_only, y_ff_perfect, y_ff_mismatched)

# Nonlinear — mass-spring-damper (m·ẍ + b·ẋ + k·x = u)
simulate_smc(m, k, b, c_smc, K_smc, phi, ref, t_end, dt) → (t, x, xdot, u, s_hist)
simulate_relay(m, k, b, u_max, ref, t_end, dt) → (t, x, u)
```

---

## UI Patterns

### Standard colour dict (replicate in each page):
```python
C = dict(
    ref="#2196F3", output="#4CAF50", error="#FF5722", control="#9C27B0",
    locus="#607D8B", ol_pole="#F44336", ol_zero="#2196F3", cl_pole="#FF9800",
    stable="#4CAF50", unstable="#F44336", neutral="#90A4AE",
    gm_line="#FF9800", pm_line="#E91E63", grid="rgba(255,255,255,0.07)",
)
```

### Standard plot layout:
```python
PLOT_LAYOUT = dict(
    template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font=dict(family="Inter, sans-serif", size=12, color="#e0e0e0"),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444"),
    margin=dict(l=60, r=30, t=40, b=50),
)
```

### Theory box CSS class:
```html
<div class="theory-box">  <!-- blue left border, dark bg -->
```
Defined via `st.markdown(""" <style> .theory-box {...} </style> """, unsafe_allow_html=True)`.
Each page defines its own CSS block at the top.

---

## Common Gotchas

### NumPy 2.x scalar extraction
```python
# WRONG in NumPy 2.0+:
val = float(arr)         # TypeError if arr has shape (1,)
# CORRECT:
val = arr.item()         # always works
val = float(arr.flat[0]) # alternative
```

### Streamlit page ordering
Files in `pages/` are sorted by filename. Use numeric prefix:
`1_Control_Simulator.py`, `2_LQR_LQG.py`, `3_MPC.py`, `4_Advanced_Control.py`

### Riccati solver failure
`scipy.linalg.solve_continuous_are` fails if the system is not stabilizable/detectable.
Always wrap in `try/except` and show `st.error()` with the message.

### Padé approximation limitation
First-order Padé for dead-time adds a RHP zero at s=2/L.
This causes non-minimum-phase behaviour (initial undershoot in step response).
It's a good approximation for L << τ; explain this limitation in the theory tab.

### MPC prediction matrix
`Psi` uses "hold last input" convention: for rows i >= Nc-1 (last column), contributions
accumulate because u_{Nc-1} is held constant for steps Nc-1, Nc, Nc+1, ... (not zero-padded).
This gives better MPC performance and is the industry-standard formulation.

### Discrete MPC stability
Unconstrained MPC → LQR as Np → ∞ (under same Q/R weights mapped to discrete cost).
With active constraints, stability is not guaranteed unless a terminal constraint is added.
The simulator uses the receding-horizon principle without terminal constraint.

---

## Test Checklist (after any change)

- [ ] `streamlit run app.py` loads without error
- [ ] Landing page shows all page cards
- [ ] PID page: all 7 tabs render; Nyquist tab shows -1 point
- [ ] LQR/LQG page: all 7 tabs render, Inverted Pendulum stabilises
- [ ] MPC page: all 5 tabs render, simulation runs, prediction window shows
- [ ] Advanced page: all 4 tabs render, Smith predictor comparison shows improvement
- [ ] Control Strategies page: all 4 tabs render; cascade beats single-loop; SMC converges to ref
- [ ] No `float(array)` calls (use `.item()`)
- [ ] No `import` at top of page file before `st.set_page_config()`

---

## What's Planned / Feature Gaps

| Feature | Priority | Where |
|---------|----------|-------|
| Nyquist plot | ✅ Done | PID page Tab 7 |
| Dead-time (Padé) plant | ✅ Done | plants.py |
| Flexible/Resonant plant | ✅ Done | plants.py |
| MPC (constrained) | ✅ Done | pages/3_MPC.py |
| Smith Predictor | ✅ Done | pages/4_Advanced_Control.py |
| System Identification | ✅ Done | pages/4_Advanced_Control.py |
| Interview Q&A | ✅ Done | pages/4_Advanced_Control.py |
| H-infinity control | 🔜 Next | new page |
| MIMO / RGA analysis | 🔜 Next | new page |
| Cascade Control | ✅ Done | pages/5_Control_Strategies.py |
| IMC / SIMC Tuning | ✅ Done | pages/5_Control_Strategies.py |
| Feedforward Control | ✅ Done | pages/5_Control_Strategies.py |
| Nonlinear (SMC + relay) | ✅ Done | pages/5_Control_Strategies.py |
| H-infinity control | 🔜 Next | new page |
| MIMO / RGA analysis | 🔜 Next | new page |
| Gain scheduling simulation | 🔜 Future | new page |
| Iterative Learning Control | 🔜 Future | new page |

---

## Scope Boundary

**Only control systems content here.** Finance/portfolio code lives in `Shreyas2552/portfolio-analyzer`.
These two repos were briefly merged (commit `79747e6`, 2026-04-30) and immediately reverted.
Do not merge them again.
