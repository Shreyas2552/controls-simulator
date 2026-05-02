# Controls Simulator — Changelog

> **Scope:** This repo (`controls-simulator`) contains **only** the Control Systems Simulator.
> The Portfolio Analyzer (stocks/ETFs/fundamentals) lives in its own repo:
> [`Shreyas2552/portfolio-analyzer`](https://github.com/Shreyas2552/portfolio-analyzer).
> Do **not** add finance/stock-data code here.

---

## v1.0 — 2026-05-02 — Initial standalone release

Extracted from `portfolio-analyzer` repo where it was briefly added as a multi-page app
(commit `79747e6`, 2026-04-30) then reverted (commit `12473eb`) the same day due to scope
and dependency coupling. Now lives here as its own standalone deployment.

### What's included

**`app.py`** — Landing page with navigation cards to both simulators.

**`pages/1_Control_Simulator.py`** — Interactive PID tuning:
- 6 plant models: First Order, Second Order, DC Motor, Integrating, Unstable, Third Order
- Live step response with P / I / D / filter gain sliders
- Bode plot (magnitude + phase) with gain & phase margin callouts
- Root locus with K-sweep and branch assignment
- Stability analysis (pole classification, ζ, ωn, overshoot, settling time)
- Filter design (Butterworth, Chebyshev, Bessel — analog and digital)
- Control theory reference guide tab

**`pages/2_LQR_LQG.py`** — Optimal control:
- 4 state-space plants: Mass-Spring-Damper, DC Motor, Inverted Pendulum, Double Integrator
- LQR state-feedback with Q / R weight sliders and eigenvalue plot
- Kalman filter observer with process & measurement noise tuning
- Full LQG loop simulation comparing LQR-only vs LQG
- Controllability / observability rank checks

**`modules/`** — Shared computation layer (no UI code):
- `plants.py` — transfer-function plant catalogue + `get_plant_tf()`
- `pid_controller.py` — continuous PID with derivative filter; Tustin discretisation
- `analysis.py` — step/ramp/control responses, Bode, root locus, stability margins, metrics
- `filters.py` — analog + digital filter design wrapping scipy
- `lqr_lqg.py` — state-space plants, LQR/Kalman Riccati solvers, LQG Euler simulation

**`requirements.txt`**: streamlit, numpy, scipy, plotly — no finance dependencies.

### Known gotchas preserved from original development

**NumPy 2.x — use `.item()` not `float()` on arrays**
`float(arr)` raises `TypeError` for 1-D arrays in NumPy 2.0+.
Use `(-K @ x).item()` to extract scalars. Already fixed in `lqr_lqg.py` at
`simulate_lqr` (lines ~231, ~237) and `simulate_lqg` (line ~272).

**`st.set_page_config()` must be first**
Each page file calls `st.set_page_config()` before any other `st.*` call.
Do not add `st.write()` or imports that trigger Streamlit above it.
