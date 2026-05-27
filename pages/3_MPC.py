"""
Model Predictive Control (MPC) Simulator
=========================================
Finite-horizon, receding-horizon optimal control with optional input constraints.
Compares MPC against LQR for each state-space plant.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.linalg import eigvals

from modules.lqr_lqg import (
    SS_PLANTS, get_ss,
    lqr_design, compute_nbar,
    simulate_lqr,
    controllability_rank,
)
from modules.mpc import (
    discretize_ss,
    build_prediction_matrices,
    build_cost_matrices,
    simulate_mpc,
)

st.set_page_config(
    page_title="MPC Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px; border-radius: 6px 6px 0 0; font-weight: 500;
    }
    div[data-testid="metric-container"] {
        background: #1e2130; border-radius: 8px; padding: 10px 14px;
    }
    .theory-box {
        background: #1a1f2e;
        border-left: 4px solid #9C27B0;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
    .insight-box {
        background: #0d1b2a;
        border-left: 4px solid #FF9800;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

C = dict(
    mpc_out="#9C27B0", lqr_out="#4CAF50", ref="#90A4AE",
    pred="#FF9800", u_mpc="#00BCD4", u_lqr="#9C27B0",
    constrained="#F44336", grid="rgba(255,255,255,0.07)",
    stable="#4CAF50", unstable="#F44336",
)
PLOT = dict(
    template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font=dict(family="Inter, sans-serif", size=12, color="#e0e0e0"),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444"),
    margin=dict(l=60, r=30, t=40, b=50),
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 MPC Simulator")
    st.markdown("---")

    st.markdown("### Plant Model")
    plant_name = st.selectbox("Plant", list(SS_PLANTS.keys()), label_visibility="collapsed")
    info = SS_PLANTS[plant_name]
    st.caption(f"**{info['desc']}**")
    st.caption(f"📌 {info['physical']}")

    plant_params = {}
    for pname, pm in info["params"].items():
        plant_params[pname] = st.slider(
            pm["label"], float(pm["min"]), float(pm["max"]),
            float(pm["default"]), float(pm["step"]),
        )
    st.markdown("---")

    st.markdown("### Discretisation")
    dt = st.select_slider(
        "Sample time dt (s)",
        options=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        value=0.05,
        help="Smaller → more accurate but longer build time for large Np",
    )
    st.markdown("---")

    st.markdown("### MPC Horizons")
    st.caption("Prediction Np = how far ahead MPC looks; Control Nc = free moves (Nc ≤ Np)")
    Np = st.slider("Prediction Horizon Np", 2, 40, 15)
    Nc = st.slider("Control Horizon Nc", 1, Np, min(5, Np))
    st.markdown("---")

    st.markdown("### Cost Weights")
    st.caption("Q penalises output error; R penalises control effort")
    n_states = info["n_states"]
    snames   = info["state_names"]
    def_Q    = info["default_Q"]

    q_out = st.number_input("Output weight Q", 0.1, 10000.0, 100.0, 10.0)
    r_in  = st.number_input("Input weight R",  0.001, 1000.0, float(info["default_R"]), 0.01,
                             format="%.3f")
    st.markdown("---")

    st.markdown("### Input Constraints")
    use_constraints = st.toggle("Enable input limits", value=False)
    u_min_v = u_max_v = None
    if use_constraints:
        u_min_v = st.number_input("u_min", -1000.0, 0.0, -10.0, 0.5)
        u_max_v = st.number_input("u_max",  0.0, 1000.0,  10.0, 0.5)
    st.markdown("---")

    st.markdown("### Process Noise")
    add_noise  = st.toggle("Add process noise", value=False)
    q_noise_std = st.select_slider(
        "Process noise std σw",
        options=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1],
        value=0.01,
    ) if add_noise else 0.0
    st.markdown("---")

    st.markdown("### Simulation")
    ref_val = st.number_input("Setpoint / Reference", -10.0, 10.0, 1.0, 0.1)
    t_end   = st.slider("Duration (s)", 2.0, 60.0, 15.0, 0.5)

# ── Compute ─────────────────────────────────────────────────────────────────
A, B, C_out, D = get_ss(plant_name, plant_params)
n = A.shape[0]
ny = C_out.shape[0]
nu = B.shape[1]
ref_state = info["ref_state"]

Ad, Bd = discretize_ss(A, B, dt)
ol_eigs = eigvals(A)
ol_stable = all(e.real < 0 for e in ol_eigs)
ctrl_rank = controllability_rank(A, B)

Q_y  = np.eye(ny) * q_out
R_u  = np.eye(nu) * r_in

mpc_ok  = False
lqr_ok  = False
mpc_err = lqr_err = None

# MPC simulation
try:
    t_mpc, x_mpc, y_mpc, u_mpc, const_hist, snapshots = simulate_mpc(
        Ad, Bd, C_out, Np, Nc, Q_y, R_u,
        ref_val, t_end, dt,
        ref_state=ref_state,
        u_min=u_min_v, u_max=u_max_v,
        q_noise_std=q_noise_std,
    )
    mpc_ok = True
except Exception as e:
    mpc_err = str(e)

# LQR simulation (for comparison — continuous, no noise)
Q_lqr = np.diag([float(def_Q[i]) for i in range(n)])
try:
    K_lqr, _, cl_eigs = lqr_design(A, B, Q_lqr, np.array([[r_in]]))
    Nbar = compute_nbar(A, B, C_out[0:1, :], K_lqr)
    t_lqr, x_lqr, y_lqr, u_lqr_h = simulate_lqr(
        A, B, C_out, K_lqr, Nbar, ref_val, t_end, ref_state=ref_state)
    lqr_ok = True
except Exception as e:
    lqr_err = str(e)

# ── Header ───────────────────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("## 🤖 Model Predictive Control (MPC) Simulator")
    st.markdown(
        f"**Plant:** {info['desc']}  |  **Np={Np}**, **Nc={Nc}**  |  "
        f"**dt={dt}s**  |  **Constraints:** {'ON' if use_constraints else 'OFF'}"
    )
with h2:
    badge = "🟢 **Stable OL**" if ol_stable else "🔴 **Unstable OL**"
    st.markdown(f"### {badge}")
    st.caption(f"Ctrl rank: {ctrl_rank}/{n}")

st.markdown("---")

tabs = st.tabs([
    "📈 MPC Response",
    "🔭 Prediction Window",
    "📊 Horizon Analysis",
    "⚔️ MPC vs LQR",
    "📚 Theory",
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — MPC Response
# ─────────────────────────────────────────────────────────────────────────
with tabs[0]:
    if not mpc_ok:
        st.error(f"MPC simulation failed: {mpc_err}")
    else:
        # Performance metrics
        y_prim = y_mpc[:, 0]
        try:
            i10 = np.where(y_prim >= 0.1 * ref_val)[0][0] if ref_val != 0 else 0
            i90 = np.where(y_prim >= 0.9 * ref_val)[0][0] if ref_val != 0 else 0
            tr  = float(t_mpc[i90] - t_mpc[i10])
        except Exception:
            tr  = None
        try:
            outside = np.where(np.abs(y_prim - ref_val) > 0.02 * abs(ref_val))[0]
            ts = float(t_mpc[outside[-1]]) if len(outside) else 0.0
        except Exception:
            ts = None
        os_pct = max(0.0, (y_prim.max() - ref_val) / (abs(ref_val) + 1e-15) * 100) if ref_val != 0 else 0.0
        ss_err = float(ref_val - y_prim[-1])
        n_const = int(const_hist.sum())

        mc = st.columns(5)
        mc[0].metric("Overshoot (%)",   f"{os_pct:.1f}")
        mc[1].metric("Rise Time (s)",   f"{tr:.3f}" if tr is not None else "N/A")
        mc[2].metric("Settle (2%) s",   f"{ts:.3f}" if ts is not None else "N/A")
        mc[3].metric("SS Error",        f"{ss_err:.4f}")
        mc[4].metric("Clipped steps",   str(n_const),
                     delta=("⚠️ active" if n_const > 0 else "✅ none"),
                     delta_color="inverse" if n_const > 0 else "off")

        fig_r = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=("Output y(t) vs Reference", "All States", "Control Input u(t)"),
        )

        fig_r.add_trace(go.Scatter(x=t_mpc, y=np.full(len(t_mpc), ref_val),
                                   name="Reference", line=dict(color=C["ref"], dash="dash", width=1.5)),
                        row=1, col=1)
        fig_r.add_trace(go.Scatter(x=t_mpc, y=y_prim, name="MPC output",
                                   line=dict(color=C["mpc_out"], width=2.5)), row=1, col=1)

        palette = ["#9C27B0", "#2196F3", "#FF9800", "#E91E63"]
        for si in range(min(n, 4)):
            fig_r.add_trace(go.Scatter(x=t_mpc, y=x_mpc[:, si],
                                       name=snames[si].split()[0],
                                       line=dict(color=palette[si % 4], width=1.5)),
                            row=2, col=1)

        fig_r.add_trace(go.Scatter(x=t_mpc, y=u_mpc, name="u(t) — MPC",
                                   line=dict(color=C["u_mpc"], width=2)), row=3, col=1)

        if use_constraints and u_min_v is not None:
            fig_r.add_hline(y=u_min_v, row=3, col=1,
                            line=dict(color=C["constrained"], dash="dot", width=1.5))
        if use_constraints and u_max_v is not None:
            fig_r.add_hline(y=u_max_v, row=3, col=1,
                            line=dict(color=C["constrained"], dash="dot", width=1.5))

        # Shade constrained steps
        if n_const > 0:
            for k, c in enumerate(const_hist):
                if c and k + 1 < len(t_mpc):
                    fig_r.add_vrect(x0=t_mpc[k], x1=t_mpc[k + 1], fillcolor="rgba(244,67,54,0.15)",
                                    line_width=0, row=3, col=1)

        for r in [1, 2, 3]:
            fig_r.update_xaxes(gridcolor=C["grid"], row=r, col=1)
            fig_r.update_yaxes(gridcolor=C["grid"], row=r, col=1)
        fig_r.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig_r.update_layout(height=680, **PLOT)
        st.plotly_chart(fig_r, use_container_width=True)

        if n_const > 0:
            st.warning(
                f"⚠️ Input constraint active at {n_const} steps ({n_const*dt:.2f}s). "
                "Red shading shows constraint-active periods. Note: unconstrained MPC "
                "is solved analytically; constraints are applied by clipping (suboptimal). "
                "Full constrained MPC would require QP solve at each step."
            )

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — Prediction Window
# ─────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown(
        "MPC computes an **optimal trajectory over Np future steps** at every time instant, "
        "but applies **only the first control move**. The prediction restarts next step — "
        "this is the **receding horizon** principle. Below: 4 snapshots of what MPC was predicting."
    )
    if not mpc_ok:
        st.error(f"MPC simulation failed: {mpc_err}")
    elif not snapshots:
        st.info("No snapshots available — increase simulation duration.")
    else:
        snap_keys = sorted(snapshots.keys())
        n_snaps   = len(snap_keys)
        cols_snap = st.columns(min(n_snaps, 4))

        for idx, k in enumerate(snap_keys[:4]):
            snap = snapshots[k]
            with cols_snap[idx]:
                fig_s = go.Figure()
                # Already-simulated trajectory up to k
                fig_s.add_trace(go.Scatter(
                    x=t_mpc[:k + 1], y=y_mpc[:k + 1, 0],
                    name="Past", line=dict(color=C["mpc_out"], width=2),
                ))
                # Prediction window
                fig_s.add_trace(go.Scatter(
                    x=snap["t_pred"], y=snap["y_pred"],
                    name="Prediction", line=dict(color=C["pred"], dash="dash", width=1.5),
                ))
                # Reference
                t_full = np.concatenate([t_mpc[:k + 1], snap["t_pred"]])
                fig_s.add_trace(go.Scatter(
                    x=t_full, y=np.full(len(t_full), ref_val),
                    name="Ref", line=dict(color=C["ref"], dash="dot", width=1),
                ))
                fig_s.update_layout(
                    title=f"t = {t_mpc[k]:.1f}s",
                    xaxis_title="Time (s)", yaxis_title="Output",
                    xaxis=dict(gridcolor=C["grid"]),
                    yaxis=dict(gridcolor=C["grid"]),
                    height=300, showlegend=False, **PLOT,
                )
                st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("""<div class="insight-box">

**Key insight:** At each snapshot, MPC sees the orange dashed prediction and adjusts u to drive it
toward the reference. If the prediction were perfect and the model exact, all orange lines would hit the
reference exactly at the end of the horizon. In practice, model mismatch and noise cause them to
miss — and the horizon recedes to correct.
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — Horizon Analysis (Np sweep)
# ─────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown(
        "How does the **prediction horizon Np** affect closed-loop performance? "
        "Longer horizon → smoother, more anticipatory response (but more computation). "
        "Short horizon → myopic, faster but can be aggressive or oscillatory."
    )
    Np_values = st.multiselect(
        "Select Np values to compare",
        options=[2, 5, 10, 15, 20, 30, 40],
        default=[3, 8, 20],
    )
    if not Np_values:
        st.info("Select at least one Np value above.")
    else:
        fig_h = go.Figure()
        palette_h = ["#9C27B0", "#2196F3", "#4CAF50", "#FF9800", "#F44336", "#00BCD4", "#E91E63"]

        summary_rows = []
        for i_h, Np_h in enumerate(sorted(Np_values)):
            Nc_h = min(3, Np_h)
            try:
                t_h, _, y_h, u_h, _, _ = simulate_mpc(
                    Ad, Bd, C_out, Np_h, Nc_h, Q_y, R_u,
                    ref_val, t_end, dt, ref_state=ref_state,
                )
                y_hv = y_h[:, 0]
                fig_h.add_trace(go.Scatter(
                    x=t_h, y=y_hv, name=f"Np={Np_h}",
                    line=dict(color=palette_h[i_h % len(palette_h)], width=2),
                ))
                os_h = max(0.0, (y_hv.max() - ref_val) / (abs(ref_val) + 1e-15) * 100) if ref_val != 0 else 0.0
                outside_h = np.where(np.abs(y_hv - ref_val) > 0.02 * abs(ref_val))[0]
                ts_h = float(t_h[outside_h[-1]]) if len(outside_h) else 0.0
                summary_rows.append({"Np": Np_h, "Nc": Nc_h,
                                     "Overshoot (%)": round(os_h, 1),
                                     "Settling (2%) s": round(ts_h, 3)})
            except Exception as ex:
                st.warning(f"Np={Np_h} failed: {ex}")

        fig_h.add_trace(go.Scatter(x=t_mpc, y=np.full(len(t_mpc), ref_val),
                                   name="Reference", line=dict(color=C["ref"], dash="dash", width=1.5)))
        fig_h.update_layout(xaxis_title="Time (s)", yaxis_title="Output",
                             xaxis=dict(gridcolor=C["grid"]), yaxis=dict(gridcolor=C["grid"]),
                             height=420, **PLOT)
        st.plotly_chart(fig_h, use_container_width=True)

        if summary_rows:
            st.markdown("#### Performance Summary")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — MPC vs LQR
# ─────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown(
        "**LQR** solves an **infinite-horizon** unconstrained problem analytically (one-time Riccati solve).  \n"
        "**MPC** solves a **finite-horizon** problem at every time step — allows constraints, "
        "handles nonlinear plants, and can be re-tuned online."
    )
    if not mpc_ok and not lqr_ok:
        st.error("Both MPC and LQR failed.")
    else:
        fig_cmp = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=("Output: MPC vs LQR", "Control input"))

        fig_cmp.add_trace(go.Scatter(x=t_mpc, y=np.full(len(t_mpc), ref_val),
                                     name="Reference", line=dict(color=C["ref"], dash="dash", width=1.5)),
                          row=1, col=1)
        if mpc_ok:
            fig_cmp.add_trace(go.Scatter(x=t_mpc, y=y_mpc[:, 0], name="MPC",
                                         line=dict(color=C["mpc_out"], width=2.5)), row=1, col=1)
            fig_cmp.add_trace(go.Scatter(x=t_mpc, y=u_mpc, name="u — MPC",
                                         line=dict(color=C["u_mpc"], width=2)), row=2, col=1)
        if lqr_ok:
            fig_cmp.add_trace(go.Scatter(x=t_lqr, y=x_lqr[:, ref_state], name="LQR",
                                         line=dict(color=C["lqr_out"], width=2.5, dash="dash")), row=1, col=1)
            fig_cmp.add_trace(go.Scatter(x=t_lqr, y=u_lqr_h, name="u — LQR",
                                         line=dict(color=C["u_lqr"], width=2, dash="dash")), row=2, col=1)

        for r in [1, 2]:
            fig_cmp.update_xaxes(gridcolor=C["grid"], row=r, col=1)
            fig_cmp.update_yaxes(gridcolor=C["grid"], row=r, col=1)
        fig_cmp.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig_cmp.update_layout(height=520, **PLOT)
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Comparison table
        rows_cmp = []
        if mpc_ok:
            y_pm = y_mpc[:, 0]
            os_m = max(0.0, (y_pm.max() - ref_val) / (abs(ref_val) + 1e-15) * 100) if ref_val != 0 else 0.0
            out_m = np.where(np.abs(y_pm - ref_val) > 0.02 * abs(ref_val))[0]
            ts_m  = float(t_mpc[out_m[-1]]) if len(out_m) else 0.0
            rows_cmp.append({"Controller": f"MPC (Np={Np}, Nc={Nc})",
                             "Overshoot (%)": round(os_m, 1),
                             "Settle (2%) s": round(ts_m, 3),
                             "Max |u|": round(float(np.max(np.abs(u_mpc))), 3),
                             "Constraints": "ON" if use_constraints else "OFF"})
        if lqr_ok:
            y_pl = x_lqr[:, ref_state]
            os_l = max(0.0, (y_pl.max() - ref_val) / (abs(ref_val) + 1e-15) * 100) if ref_val != 0 else 0.0
            out_l = np.where(np.abs(y_pl - ref_val) > 0.02 * abs(ref_val))[0]
            ts_l  = float(t_lqr[out_l[-1]]) if len(out_l) else 0.0
            rows_cmp.append({"Controller": "LQR (infinite horizon)",
                             "Overshoot (%)": round(os_l, 1),
                             "Settle (2%) s": round(ts_l, 3),
                             "Max |u|": round(float(np.max(np.abs(u_lqr_h))), 3),
                             "Constraints": "N/A (unconstrained)"})

        if rows_cmp:
            st.dataframe(pd.DataFrame(rows_cmp), use_container_width=True, hide_index=True)

        st.markdown("""<div class="insight-box">

**When does MPC beat LQR?**
- When hard **input/state constraints** must be respected (actuator limits, safety bounds)
- When the plant is **nonlinear** (use nonlinear MPC or re-linearise at each step)
- When **setpoint changes** are known in advance (preview MPC)
- When **model uncertainty** is high and you need to re-optimise online

**When LQR wins:**
- Unconstrained linear systems → LQR is exactly optimal and needs no online computation
- Embedded / real-time systems where QP solve is too expensive
- Requires only state feedback (full-state available or from Kalman filter)
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 5 — Theory
# ─────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("## Model Predictive Control — Theory Guide")
    t1, t2 = st.columns(2)

    with t1:
        st.markdown("### What is MPC?")
        st.markdown("""<div class="theory-box">

**MPC is an optimisation-based control strategy that:**

1. Uses a **model** to predict future plant behaviour over Np steps
2. Solves an **optimisation** (minimise predicted error + control effort)
3. Applies **only the first control move**
4. Repeats at the next sample (receding / rolling horizon)

**Discrete-time state-space model:**
$$x_{k+1} = A_d x_k + B_d u_k$$
$$y_k = C x_k$$

**Prediction:**
$$Y = \\Psi U + \\Phi x_k$$

where $Y = [y_{k+1}; \\ldots; y_{k+N_p}]$,  $U = [u_k; \\ldots; u_{k+N_c-1}]$
</div>""", unsafe_allow_html=True)

        st.markdown("### Cost Function")
        st.markdown("""<div class="theory-box">

$$J = \\underbrace{(Y - Y_{ref})^\\top \\bar{Q} (Y - Y_{ref})}_{\\text{tracking error}} + \\underbrace{U^\\top \\bar{R} U}_{\\text{control effort}}$$

$\\bar{Q} = \\text{blkdiag}(Q, Q, \\ldots, Q)$ — $N_p$ blocks
$\\bar{R} = \\text{blkdiag}(R, R, \\ldots, R)$ — $N_c$ blocks

**Unconstrained analytical solution:**
$$U^* = (\\Psi^\\top \\bar{Q} \\Psi + \\bar{R})^{-1} \\Psi^\\top \\bar{Q} (Y_{ref} - \\Phi x_k)$$

Only $u_k = U^*[0]$ is applied. The matrix is inverted **once offline** for constant models.
</div>""", unsafe_allow_html=True)

        st.markdown("### Prediction Horizon Np vs Control Horizon Nc")
        st.markdown("""<div class="theory-box">

| Parameter | Meaning | Effect of increasing |
|-----------|---------|---------------------|
| **Np** | Steps ahead predicted | Smoother, anticipates future; slower to compute |
| **Nc** | Free control moves | More degrees of freedom; faster response |

**Rule of thumb:** Nc ≈ Np/3 to Np/5. Setting Nc=1 gives "move blocking" (cheapest).

**As Np → ∞:** Unconstrained MPC → **LQR** with the same Q/R (under appropriate mapping).
The finite horizon is the key difference that enables constraints.
</div>""", unsafe_allow_html=True)

    with t2:
        st.markdown("### Constraints")
        st.markdown("""<div class="theory-box">

**The killer advantage of MPC over LQR** — hard constraints can be embedded in the optimisation:

- **Input constraints:** $u_{min} \\leq u_k \\leq u_{max}$ (actuator limits)
- **State constraints:** $x_{min} \\leq x_k \\leq x_{max}$ (safety envelopes)
- **Rate constraints:** $|\\Delta u_k| \\leq \\Delta u_{max}$ (slew-rate limits)

These turn the QP into a **constrained quadratic program** (requires solver like OSQP, ECOS, or Clarabel).

This simulator uses **analytical unconstrained + clipping** for illustration — a full implementation would use cvxpy or do-mpc.
</div>""", unsafe_allow_html=True)

        st.markdown("### Stability and Robustness")
        st.markdown("""<div class="theory-box">

**Stability guarantees:**
- Without constraints: guaranteed stable (= LQR) if Np is large enough
- With constraints: need **terminal constraint** or **terminal cost** (Lyapunov condition)
- **Offset-free tracking**: add integrator or use **incremental MPC** formulation (Δu as decision variable)

**Robustness (compared to LQR):**
- MPC can have **poor robustness margins** if model mismatch is large
- **Tube MPC** and **robust MPC** add constraint tightening for guaranteed robust stability
- LQR has guaranteed gain margin [0.5, ∞) and phase margin ≥ 60° — MPC has none in general

**Computational cost:**
- Unconstrained: O(n_states · Np · Nc) — offline inversion possible
- Constrained QP: active-set or interior-point, O(Nc²) iterations typical
</div>""", unsafe_allow_html=True)

        st.markdown("### MPC in Industry")
        st.markdown("""<div class="theory-box">

| Application | Why MPC? |
|-------------|----------|
| Refinery MIMO control | Handles 100s of inputs/outputs, constraints |
| Building HVAC | Preview of weather forecast (known disturbances) |
| Autonomous vehicles | Path + speed constraints, obstacle avoidance |
| Battery management | SoC/SoH constraints, thermal limits |
| Chemical reactors | Safety constraints, nonlinear kinetics |
| Rocket landing | Fuel + structural + landing zone constraints |

**Commercial tools:** Aspen DMC, Honeywell RMPCT, ABB 800xA MPC, do-mpc (open source), CVXPY
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Interview tip:** MPC is the most asked-about advanced control method in senior/principal control engineer interviews. Know the cost function, prediction matrices, and why constraints make it better than LQR.")
