"""
Control Strategies — interactive simulation of advanced loop architectures.

Tabs
----
  1  Cascade Control   — nested PI loops vs single-loop PI with disturbance
  2  IMC / SIMC        — lambda-based tuning vs Ziegler-Nichols comparison
  3  Feedforward       — FB-only vs perfect FF vs mismatched FF
  4  Nonlinear Control — Sliding-mode (SMC) and relay on mass-spring-damper
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from modules.advanced_control import (
    simulate_cascade,
    simc_pid_fopdt, simc_pid_2nd_order, zn_pid_fopdt,
    simulate_pid_fopdt,
    simulate_feedforward,
    simulate_smc, simulate_relay,
)

st.set_page_config(
    page_title="Control Strategies",
    page_icon="🔗",
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
        border-left: 4px solid #00BCD4;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

PLOT = dict(
    template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font=dict(family="Inter, sans-serif", size=12, color="#e0e0e0"),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444"),
    margin=dict(l=60, r=30, t=40, b=50),
)
C = dict(
    casc="#4CAF50", single="#F44336", ref="#90A4AE",
    simc="#4CAF50", zn="#FF9800", manual="#2196F3",
    fb="#F44336", ff_perfect="#4CAF50", ff_mismatch="#FF9800",
    smc="#4CAF50", relay="#2196F3", slide="#FF9800",
    grid="rgba(255,255,255,0.07)",
)

st.markdown("## 🔗 Control Strategies")
st.markdown("*Interactive simulations: cascade control, IMC/SIMC tuning, feedforward, and nonlinear control*")
st.markdown("---")

tabs = st.tabs([
    "🔗 Cascade Control",
    "🎛️ IMC / SIMC Tuning",
    "➡️ Feedforward Control",
    "🌀 Nonlinear Control",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Cascade Control
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown(
        "**Cascade control** uses two nested feedback loops — a fast *inner loop* that tightly "
        "controls an intermediate variable (e.g. flow, heater power), and a slow *outer loop* "
        "that controls the main process variable. Disturbances entering between the loops are "
        "rejected by the inner loop **before** they affect the output. "
        "Adjust the disturbance slider and watch how cascade beats a single loop."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("#### Outer Plant G₁(s) = K₁/(τ₁s+1)")
        K1    = st.number_input("K₁ (outer gain)",    0.1, 10.0, 2.0, 0.1, key="cc_K1")
        tau1  = st.number_input("τ₁ — slow (s)",      1.0, 60.0, 10.0, 1.0, key="cc_tau1")

        st.markdown("#### Inner Plant G₂(s) = K₂/(τ₂s+1)")
        K2    = st.number_input("K₂ (inner gain)",    0.1, 10.0, 1.0, 0.1, key="cc_K2")
        tau2  = st.number_input("τ₂ — fast (s)",      0.1, 10.0, 1.0, 0.5, key="cc_tau2")

        st.markdown("#### Controller Tuning")
        Kp_out = st.number_input("Outer Kp", 0.01, 20.0, 0.5, 0.05, key="cc_Kpout")
        Ki_out = st.number_input("Outer Ki", 0.0,  5.0,  0.05, 0.01, key="cc_Kiout")
        Kp_in  = st.number_input("Inner Kp", 0.1,  50.0, 2.0, 0.1, key="cc_Kpin")
        Ki_in  = st.number_input("Inner Ki", 0.0,  20.0, 0.5, 0.1, key="cc_Kiin")

        st.markdown("#### Simulation")
        ref_cc   = st.number_input("Setpoint", -5.0, 5.0, 1.0, 0.1, key="cc_ref")
        dist_amp = st.slider("Disturbance amplitude", -5.0, 5.0, 1.0, 0.1)
        dist_t   = st.slider("Disturbance start time (s)", 1.0, 30.0, 15.0, 1.0)
        t_end_cc = st.slider("Duration (s)", 10.0, 120.0, 60.0, 5.0)

    with c2:
        try:
            t_cc, y_casc, y_single, u_casc = simulate_cascade(
                K1, tau1, K2, tau2,
                Kp_out, Ki_out, Kp_in, Ki_in,
                disturbance=dist_amp, dist_time=dist_t,
                ref=ref_cc, t_end=t_end_cc, dt=0.05,
            )

            fig_cc = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                subplot_titles=("Output: Cascade vs Single-loop PI",
                                "Cascade inner loop input u(t)"),
            )
            ref_line = np.full(len(t_cc), ref_cc)
            fig_cc.add_trace(go.Scatter(x=t_cc, y=ref_line, name="Setpoint",
                                         line=dict(color=C["ref"], dash="dash", width=1.5)),
                             row=1, col=1)
            fig_cc.add_trace(go.Scatter(x=t_cc, y=y_casc, name="Cascade (inner + outer)",
                                         line=dict(color=C["casc"], width=2.5)), row=1, col=1)
            fig_cc.add_trace(go.Scatter(x=t_cc, y=y_single, name="Single-loop PI",
                                         line=dict(color=C["single"], width=2, dash="dash")),
                             row=1, col=1)

            # Disturbance annotation
            fig_cc.add_vline(x=dist_t, line_dash="dot",
                              line_color="rgba(255,200,0,0.5)",
                              annotation_text="disturbance", row=1, col=1)

            fig_cc.add_trace(go.Scatter(x=t_cc, y=u_casc, name="u (cascade)",
                                         line=dict(color="#2196F3", width=1.5)), row=2, col=1)

            for r in [1, 2]:
                fig_cc.update_xaxes(gridcolor=C["grid"], row=r, col=1)
                fig_cc.update_yaxes(gridcolor=C["grid"], row=r, col=1)
            fig_cc.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig_cc.update_layout(height=520, **PLOT)
            st.plotly_chart(fig_cc, use_container_width=True)

            # Metrics
            m1, m2, m3 = st.columns(3)
            ss_casc  = float(y_casc[-1])
            ss_single = float(y_single[-1])
            # Disturbance rejection: peak deviation after dist_t
            idx_d = int(dist_t / 0.05)
            dev_casc   = float(np.max(np.abs(y_casc[idx_d:]   - ref_cc)))
            dev_single = float(np.max(np.abs(y_single[idx_d:] - ref_cc)))
            m1.metric("Cascade SS error", f"{abs(ss_casc - ref_cc):.4f}")
            m2.metric("Single-loop SS error", f"{abs(ss_single - ref_cc):.4f}")
            m3.metric("Peak disturbance rejection",
                       f"{100*(1 - dev_casc/max(dev_single, 1e-9)):.0f}% better" if dev_single > 1e-4 else "N/A")

        except Exception as e:
            st.error(f"Cascade simulation error: {e}")

    st.markdown("---")
    st.markdown("""<div class="theory-box">

**Rule for cascade tuning:** The inner loop must be at least **3–5× faster** than the outer loop.
Tune inner loop first (with outer loop open), then close outer loop.

**When cascade helps:** Disturbance enters between inner and outer plant (e.g. supply voltage to heater),
or inner process dynamics need tight regulation (e.g. flow in a temperature reactor).

**When it doesn't help:** If you can't measure an intermediate variable, or if the inner loop
can't be made significantly faster than the outer loop — a single PI with feedforward is simpler.

**Real example:** Boiler drum level — inner loop controls feedwater flow rate (fast valve,
time constant ~2s); outer loop controls drum level (water inventory, time constant ~30s).
Disturbance: steam demand changes → inner loop cancels flow mismatch in seconds.
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — IMC / SIMC Tuning
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown(
        "**IMC (Internal Model Control)** designs a controller from the desired closed-loop "
        "response shape. **SIMC** (Skogestad's Simplified IMC) converts that into standard "
        "PID parameters with a single tuning knob λ (the desired closed-loop time constant). "
        "Compare against Ziegler-Nichols on the same plant."
    )

    imc_c1, imc_c2 = st.columns([1, 2])

    with imc_c1:
        st.markdown("#### Plant Model")
        plant_order = st.radio("Plant type", ["1st order (FOPDT)", "2nd order + dead-time"],
                               horizontal=True)

        K_imc   = st.number_input("DC Gain K",        0.1, 20.0, 1.0, 0.1, key="imc_K")
        tau1_imc = st.number_input("τ₁ — dominant (s)", 0.5, 50.0, 5.0, 0.5, key="imc_tau1")
        if plant_order.startswith("2nd"):
            tau2_imc = st.number_input("τ₂ — secondary (s)", 0.1, 20.0, 1.0, 0.1, key="imc_tau2")
        else:
            tau2_imc = 0.0
        theta_imc = st.number_input("Dead time θ (s)", 0.0, 20.0, 1.0, 0.1, key="imc_theta")

        st.markdown("#### Tuning")
        lambda_c = st.slider("λ — IMC/SIMC filter constant (s)",
                              0.1, 20.0, 2.0, 0.1,
                              help="Larger λ → slower, more robust. Rule: λ ≥ θ")

        st.markdown("#### Simulation")
        ref_imc  = st.number_input("Setpoint", -5.0, 5.0, 1.0, 0.1, key="imc_ref")
        t_imc    = st.slider("Duration (s)", 5.0, 200.0, 50.0, 5.0, key="imc_dur")

    with imc_c2:
        try:
            if plant_order.startswith("1st"):
                Kp_s, Ki_s, Kd_s = simc_pid_fopdt(K_imc, tau1_imc, theta_imc, lambda_c)
                Kp_zn, Ki_zn, Kd_zn = zn_pid_fopdt(K_imc, tau1_imc, theta_imc)
                tau2_imc = 0.0
            else:
                Kp_s, Ki_s, Kd_s   = simc_pid_2nd_order(K_imc, tau1_imc, tau2_imc, theta_imc, lambda_c)
                Kp_zn, Ki_zn, Kd_zn = zn_pid_fopdt(K_imc, tau1_imc, theta_imc)

            dt_imc = 0.02
            t_v, y_simc, _ = simulate_pid_fopdt(K_imc, tau1_imc + tau2_imc * 0.5,
                                                  theta_imc, Kp_s, Ki_s, Kd_s,
                                                  ref_imc, t_imc, dt_imc)
            _,   y_zn,   _ = simulate_pid_fopdt(K_imc, tau1_imc + tau2_imc * 0.5,
                                                  theta_imc, Kp_zn, Ki_zn, Kd_zn,
                                                  ref_imc, t_imc, dt_imc)

            fig_imc = go.Figure()
            fig_imc.add_trace(go.Scatter(x=t_v, y=np.full(len(t_v), ref_imc), name="Setpoint",
                                          line=dict(color=C["ref"], dash="dash", width=1.5)))
            fig_imc.add_trace(go.Scatter(x=t_v, y=y_simc, name=f"SIMC (λ={lambda_c:.1f})",
                                          line=dict(color=C["simc"], width=2.5)))
            fig_imc.add_trace(go.Scatter(x=t_v, y=y_zn, name="Ziegler-Nichols",
                                          line=dict(color=C["zn"], width=2, dash="dash")))

            fig_imc.update_layout(
                xaxis_title="Time (s)", yaxis_title="Output",
                xaxis=dict(gridcolor=C["grid"]), yaxis=dict(gridcolor=C["grid"]),
                height=380, **PLOT,
            )
            st.plotly_chart(fig_imc, use_container_width=True)

            # Computed gains table
            m1, m2, m3 = st.columns(3)
            m1.metric("SIMC Kp", f"{Kp_s:.3f}")
            m2.metric("SIMC Ki", f"{Ki_s:.3f}")
            m3.metric("SIMC Kd", f"{Kd_s:.3f}")

            m4, m5, m6 = st.columns(3)
            m4.metric("Z-N Kp", f"{Kp_zn:.3f}")
            m5.metric("Z-N Ki", f"{Ki_zn:.3f}")
            m6.metric("Z-N Kd", f"{Kd_zn:.3f}")

            # Overshoot / settling
            os_simc = max(0.0, (float(np.max(y_simc)) - ref_imc) / (ref_imc + 1e-9) * 100)
            os_zn   = max(0.0, (float(np.max(y_zn))   - ref_imc) / (ref_imc + 1e-9) * 100)
            st.caption(f"Overshoot — SIMC: **{os_simc:.1f}%** | Z-N: **{os_zn:.1f}%**  "
                       f"(SIMC overshoot ≤ 5% by design when λ ≥ θ)")

        except Exception as e:
            st.error(f"IMC/SIMC simulation error: {e}")

    st.markdown("---")
    st.markdown("""<div class="theory-box">

**SIMC Tuning Rules (Skogestad 2003) — the fastest formula worth memorising:**

For FOPDT  G(s) = K·e^{-θs}/(τs+1):

| Parameter | Formula |
|-----------|---------|
| Kp | τ / (K·(λ+θ)) |
| Ti (= Kp/Ki) | min(τ, 4·(λ+θ)) |
| Td (= Kd/Kp) | 0 (PI for FOPDT) |

**λ selection guide:**
- Start at λ = θ → fast, ~5% overshoot
- Increase λ for more robustness or noisier plants
- λ = τ/2 → conservative, barely any overshoot

**Why SIMC beats Ziegler-Nichols:** Z-N gives ~25% overshoot and requires driving the system
to the edge of instability. SIMC uses a plant model (from a step test) and a single λ parameter —
predictable, safe, and usually better performing.

**Interview answer:** "I use SIMC as my default tuning rule because it's based on a simple
plant model, has one intuitive knob, and gives consistent performance across plants."
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Feedforward Control
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown(
        "**Feedforward control** measures a disturbance *before* it reaches the plant and "
        "pre-computes a compensating input: u_ff = −(Gd/G)·d. This cancels the disturbance "
        "before feedback can even react. The key limitation: it requires an accurate plant model. "
        "Adjust the **FF model mismatch** to see what happens when the model is wrong — "
        "feedback still cleans up the residual."
    )

    ff_c1, ff_c2 = st.columns([1, 2])
    with ff_c1:
        st.markdown("#### Process G(s) = K_G/(τ_G s+1)")
        K_G   = st.number_input("Process gain K_G",      0.1, 10.0, 1.0, 0.1, key="ff_KG")
        tau_G = st.number_input("Process τ_G (s)",       0.5, 30.0, 5.0, 0.5, key="ff_tauG")

        st.markdown("#### Disturbance Path G_d(s) = K_Gd/(τ_Gd s+1)")
        K_Gd   = st.number_input("Disturbance gain K_Gd",  0.1, 10.0, 0.8, 0.1, key="ff_KGd")
        tau_Gd = st.number_input("Disturbance τ_Gd (s)",   0.1, 10.0, 1.5, 0.1, key="ff_tauGd")

        st.markdown("#### Feedback PI")
        Kp_ff = st.number_input("Kp", 0.01, 20.0, 0.4, 0.05, key="ff_Kp")
        Ki_ff = st.number_input("Ki", 0.0,  5.0,  0.08, 0.01, key="ff_Ki")

        st.markdown("#### Disturbance & Mismatch")
        dist_amp_ff = st.slider("Disturbance amplitude", -5.0, 5.0, 1.0, 0.1, key="ff_dist")
        dist_t_ff   = st.slider("Disturbance start (s)", 1.0, 30.0, 10.0, 1.0, key="ff_dtim")
        mismatch    = st.slider("FF model mismatch (×perfect)",
                                 0.0, 2.0, 1.0, 0.05,
                                 help="1.0=perfect cancellation, 0=FF off, 1.5=50% over-compensation")

        ref_ff  = st.number_input("Setpoint", -5.0, 5.0, 0.0, 0.1, key="ff_ref")
        t_ff    = st.slider("Duration (s)", 5.0, 100.0, 40.0, 5.0, key="ff_dur")

    with ff_c2:
        try:
            t_v, y_fb, y_ff_p, y_ff_m = simulate_feedforward(
                K_G, tau_G, K_Gd, tau_Gd,
                ff_gain_mismatch=mismatch,
                Kp=Kp_ff, Ki=Ki_ff,
                dist_amp=dist_amp_ff, dist_time=dist_t_ff,
                ref=ref_ff, t_end=t_ff, dt=0.02,
            )

            fig_ff = go.Figure()
            fig_ff.add_trace(go.Scatter(x=t_v, y=np.full(len(t_v), ref_ff), name="Setpoint",
                                         line=dict(color=C["ref"], dash="dash", width=1.5)))
            fig_ff.add_trace(go.Scatter(x=t_v, y=y_fb, name="Feedback only (PI)",
                                         line=dict(color=C["fb"], width=2, dash="dash")))
            fig_ff.add_trace(go.Scatter(x=t_v, y=y_ff_p, name="Perfect FF + PI",
                                         line=dict(color=C["ff_perfect"], width=2.5)))
            if abs(mismatch - 1.0) > 0.05:
                fig_ff.add_trace(go.Scatter(x=t_v, y=y_ff_m,
                                             name=f"Mismatched FF (×{mismatch:.2f}) + PI",
                                             line=dict(color=C["ff_mismatch"], width=2,
                                                       dash="dot")))

            fig_ff.add_vline(x=dist_t_ff, line_dash="dot",
                              line_color="rgba(255,200,0,0.5)",
                              annotation_text="disturbance")

            fig_ff.update_layout(
                xaxis_title="Time (s)", yaxis_title="Output",
                xaxis=dict(gridcolor=C["grid"]), yaxis=dict(gridcolor=C["grid"]),
                height=420, **PLOT,
            )
            st.plotly_chart(fig_ff, use_container_width=True)

            # Peak deviation after disturbance
            idx_d = int(dist_t_ff / 0.02)
            dev_fb  = float(np.max(np.abs(y_fb[idx_d:]  - ref_ff)))
            dev_ffp = float(np.max(np.abs(y_ff_p[idx_d:] - ref_ff)))
            dev_ffm = float(np.max(np.abs(y_ff_m[idx_d:] - ref_ff)))

            ma, mb, mc = st.columns(3)
            ma.metric("Peak dev — FB only",      f"{dev_fb:.4f}")
            mb.metric("Peak dev — Perfect FF",   f"{dev_ffp:.4f}")
            mc.metric("Peak dev — Mismatch FF",  f"{dev_ffm:.4f}")

        except Exception as e:
            st.error(f"Feedforward simulation error: {e}")

    st.markdown("---")
    st.markdown("""<div class="theory-box">

**Feedforward formula:**

$$u_{ff}(s) = -\\frac{G_d(s)}{G(s)} \\cdot d(s)$$

Perfect cancellation when model is exact: the disturbance never appears in the output.

**Combined feedback + feedforward:**
$$u = \\underbrace{C(s) \\cdot e}_{\\text{FB}} - \\underbrace{\\frac{G_d}{G} \\cdot d}_{\\text{FF}}$$

Feedback handles model errors and unmeasured disturbances. Feedforward handles the known, measurable disturbance before it propagates.

**When feedforward is essential:**
- *Process industries:* feed-flow disturbances in reactors, steam pressure changes in heat exchangers
- *Motion control:* gravity compensation, known friction model
- *Power electronics:* grid voltage feedforward in inverters

**Interview tip:** "Feedforward improves disturbance rejection speed; it does not change closed-loop stability (no feedback path). Its benefit is limited by model accuracy."
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Nonlinear Control
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown(
        "**Nonlinear control** goes beyond PID when the plant is strongly nonlinear or when "
        "you need guaranteed convergence. Here we simulate two methods on a "
        "**mass-spring-damper** (m·ẍ + b·ẋ + k·x = u) — the simplest nonlinear-capable benchmark: "
        "**Sliding Mode Control (SMC)** and **Relay (bang-bang) control**. "
        "Watch the phase portrait to see how states converge to the sliding surface."
    )

    nl_c1, nl_c2 = st.columns([1, 2])
    with nl_c1:
        st.markdown("#### Plant — Mass-Spring-Damper")
        m_nl = st.number_input("Mass m (kg)",       0.1, 20.0, 1.0, 0.1)
        k_nl = st.number_input("Spring k (N/m)",    0.0, 50.0, 2.0, 0.5)
        b_nl = st.number_input("Damping b (N·s/m)", 0.0, 20.0, 0.5, 0.1)

        st.markdown("#### SMC Parameters")
        c_smc  = st.slider("Sliding surface slope c  (ė + c·e)", 0.1, 10.0, 2.0, 0.1,
                           help="Larger c → faster convergence along surface")
        K_smc  = st.slider("Switching gain K",  0.1, 50.0, 5.0, 0.5,
                           help="Must exceed disturbance bound for guaranteed convergence")
        phi_smc = st.slider("Boundary layer φ  (chattering softener)",
                             0.0, 5.0, 0.5, 0.05,
                             help="φ=0 → hard sign (chattering); φ>0 → sat() (smooth)")

        st.markdown("#### Relay Parameters")
        st.caption("Relay applies bias k·ref to hold setpoint, plus ±u_amp corrective switching.")
        u_amp_relay = st.slider("Relay corrective amplitude u_amp", 0.1, 20.0, 2.0, 0.1)

        st.markdown("#### Simulation")
        ref_nl  = st.number_input("Reference position (m)", -5.0, 5.0, 1.0, 0.1, key="nl_ref")
        t_nl    = st.slider("Duration (s)", 2.0, 30.0, 10.0, 1.0)

    with nl_c2:
        try:
            dt_nl = 0.005
            t_v, x_smc, xd_smc, u_smc, s_smc = simulate_smc(
                m_nl, k_nl, b_nl, c_smc, K_smc, phi_smc,
                ref_nl, t_nl, dt_nl,
            )
            t_v, x_rel, u_rel = simulate_relay(
                m_nl, k_nl, b_nl, u_amp_relay, ref_nl, t_nl, dt_nl,
            )

            # ── Time-domain plot ─────────────────────────────────────────────
            fig_nl = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Position x(t)", "Control input u(t)",
                                "Phase portrait (x vs ẋ)", "Sliding surface s(t)"),
                vertical_spacing=0.12, horizontal_spacing=0.1,
            )

            fig_nl.add_trace(go.Scatter(x=t_v, y=np.full(len(t_v), ref_nl),
                                         name="Reference",
                                         line=dict(color=C["ref"], dash="dash", width=1.5)),
                             row=1, col=1)
            fig_nl.add_trace(go.Scatter(x=t_v, y=x_smc, name="SMC",
                                         line=dict(color=C["smc"], width=2.5)), row=1, col=1)
            fig_nl.add_trace(go.Scatter(x=t_v, y=x_rel, name="Relay",
                                         line=dict(color=C["relay"], width=2, dash="dash")),
                             row=1, col=1)

            fig_nl.add_trace(go.Scatter(x=t_v, y=u_smc, name="u — SMC",
                                         line=dict(color=C["smc"], width=1.5),
                                         showlegend=False), row=1, col=2)
            fig_nl.add_trace(go.Scatter(x=t_v, y=u_rel, name="u — Relay",
                                         line=dict(color=C["relay"], width=1.5, dash="dash"),
                                         showlegend=False), row=1, col=2)

            # Phase portrait
            fig_nl.add_trace(go.Scatter(x=x_smc, y=xd_smc, name="SMC phase",
                                         mode="lines",
                                         line=dict(color=C["smc"], width=1.5),
                                         showlegend=False), row=2, col=1)

            # Sliding surface
            fig_nl.add_trace(go.Scatter(x=t_v, y=s_smc, name="s(t) — SMC",
                                         line=dict(color=C["slide"], width=2),
                                         showlegend=False), row=2, col=2)
            fig_nl.add_hline(y=0, line_dash="dot",
                              line_color="rgba(255,255,255,0.3)", row=2, col=2)
            if phi_smc > 0:
                fig_nl.add_hrect(y0=-phi_smc, y1=phi_smc,
                                  fillcolor="rgba(255,152,0,0.08)",
                                  line_width=0, row=2, col=2)

            for r, co in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                fig_nl.update_xaxes(gridcolor=C["grid"], row=r, col=co)
                fig_nl.update_yaxes(gridcolor=C["grid"], row=r, col=co)

            fig_nl.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig_nl.update_xaxes(title_text="Time (s)", row=1, col=2)
            fig_nl.update_xaxes(title_text="Position x (m)", row=2, col=1)
            fig_nl.update_xaxes(title_text="Time (s)", row=2, col=2)
            fig_nl.update_yaxes(title_text="ẋ (m/s)", row=2, col=1)
            fig_nl.update_yaxes(title_text="s = ė + c·e", row=2, col=2)

            fig_nl.update_layout(height=600, **PLOT)
            st.plotly_chart(fig_nl, use_container_width=True)

            ma, mb, mc = st.columns(3)
            ma.metric("SMC final position", f"{float(x_smc[-1]):.4f} m")
            mb.metric("Relay final position", f"{float(x_rel[-1]):.4f} m")
            mc.metric("SMC final surface |s|", f"{abs(float(s_smc[-1])):.5f}")

        except Exception as e:
            st.error(f"Nonlinear simulation error: {e}")

    st.markdown("---")

    nc1, nc2 = st.columns(2)
    with nc1:
        st.markdown("""<div class="theory-box">

**Sliding Mode Control (SMC)**

1. Define sliding surface: **s = ė + c·e** (first-order surface, e = ref − x)
2. Reaching phase: drive state to s = 0 using: u = u_eq − K·sign(s)
3. Sliding phase: state stays on surface; dynamics governed by s = 0 → **ė = −c·e** → exponential decay

**Chattering problem:** sign(s) switches infinitely fast near s=0 → high-frequency control signal.

**Fix — boundary layer:** Replace sign(s) with sat(s/φ):
- Inside |s| < φ: linear feedback (smooth)
- Outside: still discontinuous switching

**Why SMC is powerful:** Robust to matched disturbances (same channel as input) — if K > |disturbance|, convergence is **guaranteed** regardless of plant uncertainty.

**Applications:** UAV attitude control, power converters, automotive ABS.
</div>""", unsafe_allow_html=True)

    with nc2:
        st.markdown("""<div class="theory-box">

**Relay (Bang-Bang) Control**

Simplest possible nonlinear control: u = +u_max if e > 0, else −u_max.

**Advantages:**
- No tuning needed except u_max
- Guaranteed finite-time switching
- Natural for on/off actuators (solenoids, digital valves)

**Disadvantages:**
- Persistent chattering / limit cycling around setpoint (no dead band)
- High wear on actuators from rapid switching
- No formal optimality

**Relay test (Åström-Hägglund):** Intentionally drive a relay oscillation in closed loop → measure amplitude A and period T_u → compute ultimate gain Ku = 4u_max/(π·A) and Ziegler-Nichols from Tu.

**Nonlinear control survey:**

| Method | Key idea |
|--------|---------|
| Feedback linearisation | Cancel nonlinearities via coordinate transform |
| **Sliding Mode (SMC)** | Drive to sliding surface; robust to matched disturbances |
| Backstepping | Recursive Lyapunov-based stabilisation |
| Gain scheduling | LTI controllers interpolated vs operating point |
| NMPC | MPC with nonlinear prediction model |
| Adaptive control | On-line parameter estimation + controller update |
</div>""", unsafe_allow_html=True)
