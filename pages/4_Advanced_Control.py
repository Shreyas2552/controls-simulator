"""
Advanced Control Topics & Interview Preparation
================================================
Smith Predictor (dead-time compensation)
System Identification (FOPDT fit from step response)
Interview Q&A (entry → principal level)
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from modules.mpc import simulate_smith_predictor, identify_fopdt

st.set_page_config(
    page_title="Advanced Control & Interview Prep",
    page_icon="🎓",
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
    .entry-q   { border-left: 4px solid #4CAF50; }
    .mid-q     { border-left: 4px solid #FF9800; }
    .senior-q  { border-left: 4px solid #F44336; }
    .princ-q   { border-left: 4px solid #9C27B0; }
    .q-box {
        background: #12181f;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

PLOT = dict(
    template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font=dict(family="Inter, sans-serif", size=12, color="#e0e0e0"),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#444"),
    margin=dict(l=60, r=30, t=40, b=50),
)
C = dict(sp="#4CAF50", std="#F44336", ref="#90A4AE", u_sp="#2196F3", u_std="#FF9800",
         data="#2196F3", fit="#FF9800", grid="rgba(255,255,255,0.07)")

st.markdown("## 🎓 Advanced Control Topics & Interview Prep")
st.markdown("---")

tabs = st.tabs([
    "⏱️ Smith Predictor",
    "🔬 System Identification",
    "🎤 Interview Q&A",
    "📚 Extended Theory",
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Smith Predictor
# ─────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown(
        "The **Smith Predictor** (1957) removes transport delay from the control loop, "
        "letting a standard PID work as if there were no delay. Critical in process industries "
        "(pipelines, distillation columns, pH reactors) where L/τ > 0.3."
    )

    sp_c1, sp_c2 = st.columns([1, 2])

    with sp_c1:
        st.markdown("#### Plant Parameters — G(s) = K·e^{−Ls}/(τs+1)")
        K_sp   = st.number_input("DC Gain K",        0.1, 10.0, 1.0,  0.1)
        tau_sp = st.number_input("Time constant τ (s)", 0.5, 20.0, 3.0,  0.5)
        L_sp   = st.number_input("Dead time L (s)",  0.0, 10.0, 1.5,  0.1)

        st.markdown("#### PID Tuning (same for both)")
        Kp_sp = st.number_input("Kp", 0.0, 50.0, 1.2,  0.1, key="kp_sp")
        Ki_sp = st.number_input("Ki", 0.0, 50.0, 0.3,  0.05, key="ki_sp")
        Kd_sp = st.number_input("Kd", 0.0, 20.0, 0.1,  0.05, key="kd_sp")

        st.markdown("#### Simulation")
        ref_sp = st.number_input("Reference", -5.0, 5.0, 1.0, 0.1, key="ref_sp")
        t_sp   = st.slider("Duration (s)", 5.0, 100.0, 30.0, 1.0)
        dt_sp  = 0.02

    with sp_c2:
        try:
            t_v, y_sp_v, y_std_v, u_sp_v, u_std_v = simulate_smith_predictor(
                K_sp, tau_sp, L_sp, dt_sp,
                Kp_sp, Ki_sp, Kd_sp, N_filt=20.0,
                ref=ref_sp, t_end=t_sp,
            )

            fig_sp = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                subplot_titles=("Output: Smith Predictor vs Standard PID",
                                "Control Signal u(t)"),
            )

            fig_sp.add_trace(go.Scatter(x=t_v, y=np.full(len(t_v), ref_sp),
                                        name="Reference",
                                        line=dict(color=C["ref"], dash="dash", width=1.5)),
                             row=1, col=1)
            fig_sp.add_trace(go.Scatter(x=t_v, y=y_sp_v, name="Smith Predictor + PID",
                                        line=dict(color=C["sp"], width=2.5)), row=1, col=1)
            fig_sp.add_trace(go.Scatter(x=t_v, y=y_std_v, name="Standard PID (no SP)",
                                        line=dict(color=C["std"], width=2, dash="dash")),
                             row=1, col=1)

            fig_sp.add_trace(go.Scatter(x=t_v, y=u_sp_v, name="u — Smith",
                                        line=dict(color=C["u_sp"], width=1.5)), row=2, col=1)
            fig_sp.add_trace(go.Scatter(x=t_v, y=u_std_v, name="u — Standard",
                                        line=dict(color=C["u_std"], width=1.5, dash="dash")),
                             row=2, col=1)

            for r in [1, 2]:
                fig_sp.update_xaxes(gridcolor=C["grid"], row=r, col=1)
                fig_sp.update_yaxes(gridcolor=C["grid"], row=r, col=1)
            fig_sp.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig_sp.update_layout(height=520, **PLOT)
            st.plotly_chart(fig_sp, use_container_width=True)

            # Metrics
            m1, m2, m3 = st.columns(3)
            ss_sp  = float(y_sp_v[-1])
            ss_std = float(y_std_v[-1])
            m1.metric("SS value — Smith", f"{ss_sp:.3f}")
            m2.metric("SS value — Standard", f"{ss_std:.3f}")
            m3.metric("L/τ ratio", f"{L_sp/tau_sp:.2f}",
                      delta="high delay" if L_sp / tau_sp > 0.3 else "low delay",
                      delta_color="inverse" if L_sp / tau_sp > 0.3 else "off")

        except Exception as e:
            st.error(f"Smith predictor simulation failed: {e}")

    st.markdown("---")
    st.markdown("#### How the Smith Predictor Works")
    st.markdown("""<div class="theory-box">

**Problem:** The delay e^{-Ls} adds -180° of phase at high frequencies, severely limiting achievable bandwidth.

**Smith Predictor architecture:**

$$u → \\underbrace{G(s)e^{-Ls}}_{\\text{true plant}} → y$$

The controller sees a **modified feedback signal:**
$$e_{SP} = r - y - \\underbrace{(G_m(s)e^{-Ls} - G_m(s))}_{\\text{predictor}} \\cdot u$$
$$= r - G_m(s)e^{-Ls} \\cdot u - (y - G_m(s)e^{-Ls} \\cdot u)$$

Effective closed-loop (when model is exact):
$$T(s) = \\frac{C(s)G(s)}{1 + C(s)G(s)} \\cdot e^{-Ls}$$

The delay **only appears as a multiplicative output delay** — not in the denominator!
So the controller C(s) is designed for G(s) **without** delay. Much easier to tune.

**Limitations:**
- Model must be accurate — large model mismatch degrades performance badly
- Unstable plants need modified predictor (Watanabe-Ito predictor)
- For large L/τ ratios (>1.0), consider IMC (Internal Model Control) instead
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — System Identification
# ─────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown(
        "**System identification** = estimating a plant model from measured input-output data. "
        "Here we fit a **First-Order Plus Dead-Time (FOPDT)** model G(s) = K·e^{-Ls}/(τs+1) "
        "to a step response — the most common starting point for process control."
    )

    sid_c1, sid_c2 = st.columns([1, 2])

    with sid_c1:
        st.markdown("#### Step Response Data")
        data_source = st.radio("Data source", ["Generate synthetic", "Manual entry"])

        if data_source == "Generate synthetic":
            K_true   = st.number_input("True K",   0.1, 10.0, 2.0, 0.1, key="K_true")
            tau_true = st.number_input("True τ (s)", 0.1, 20.0, 5.0, 0.1, key="tau_true")
            L_true   = st.number_input("True L (s)", 0.0, 10.0, 1.0, 0.1, key="L_true")
            noise_std = st.select_slider(
                "Measurement noise σ",
                options=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2],
                value=0.05,
            )
            t_id_end = max(5 * tau_true + 3 * L_true, 10.0)
            t_id     = np.linspace(0, t_id_end, 500)
            y_true_id = K_true * (1 - np.exp(-np.maximum(t_id - L_true, 0) / tau_true))
            y_true_id[t_id < L_true] = 0.0
            rng = np.random.default_rng(0)
            y_id = y_true_id + rng.normal(0, noise_std, len(t_id))

        else:
            st.caption("Enter comma-separated time values, then output values")
            t_raw = st.text_area("Time points (s)", "0,1,2,3,4,5,6,7,8,9,10")
            y_raw = st.text_area("Output values",   "0,0,0.2,0.5,0.7,0.85,0.92,0.96,0.98,0.99,1.0")
            try:
                t_id = np.array([float(x) for x in t_raw.split(",")])
                y_id = np.array([float(x) for x in y_raw.split(",")])
                K_true = tau_true = L_true = None
            except Exception:
                st.error("Parse error — check comma-separated format")
                t_id = y_id = None

    with sid_c2:
        if t_id is not None and y_id is not None:
            try:
                K_id, tau_id, L_id, y_fit = identify_fopdt(t_id, y_id)

                fig_sid = go.Figure()
                if K_true is not None:
                    fig_sid.add_trace(go.Scatter(x=t_id, y=y_true_id,
                                                 name="True (noiseless)",
                                                 line=dict(color="#90A4AE", dash="dot", width=1.5)))
                fig_sid.add_trace(go.Scatter(x=t_id, y=y_id, name="Measured data",
                                             mode="lines",
                                             line=dict(color=C["data"], width=1.5)))
                fig_sid.add_trace(go.Scatter(x=t_id, y=y_fit, name="FOPDT fit",
                                             line=dict(color=C["fit"], width=2.5, dash="dash")))

                fig_sid.update_layout(
                    xaxis_title="Time (s)", yaxis_title="Output",
                    xaxis=dict(gridcolor=C["grid"]),
                    yaxis=dict(gridcolor=C["grid"]),
                    height=380, **PLOT,
                )
                st.plotly_chart(fig_sid, use_container_width=True)

                rmse = float(np.sqrt(np.mean((y_id - y_fit) ** 2)))
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("K (identified)", f"{K_id:.4f}",
                          delta=f"true={K_true:.2f}" if K_true else None)
                m2.metric("τ (identified, s)", f"{tau_id:.4f}",
                          delta=f"true={tau_true:.2f}" if tau_true else None)
                m3.metric("L (identified, s)", f"{L_id:.4f}",
                          delta=f"true={L_true:.2f}" if L_true else None)
                m4.metric("Fit RMSE", f"{rmse:.5f}")

                st.markdown("#### Auto-Tuned PID (Cohen-Coon)")
                if tau_id > 0 and K_id > 0:
                    # Cohen-Coon rules for FOPDT
                    r = L_id / tau_id if tau_id > 1e-9 else 0.1
                    if r > 0:
                        Kp_cc = (1.35 / K_id) * (tau_id / L_id) * (1 + 0.185 * r)
                        Ti_cc = L_id * (2.5 - 2 * r) / (1 - 0.39 * r) if (1 - 0.39 * r) > 1e-9 else tau_id
                        Td_cc = 0.37 * L_id / (1 - 0.81 * r) if (1 - 0.81 * r) > 1e-9 else 0.1 * L_id
                        Ki_cc = Kp_cc / Ti_cc
                        st.markdown(
                            f"**Cohen-Coon PID:** Kp = `{Kp_cc:.3f}`, Ki = `{Ki_cc:.3f}`, Kd = `{Td_cc*Kp_cc:.3f}`  \n"
                            f"*(L/τ ratio = {r:.3f} — {'use Smith Predictor' if r > 0.3 else 'direct PID acceptable'})*"
                        )

            except Exception as e:
                st.error(f"Identification failed: {e}")

    st.markdown("---")
    st.markdown("""<div class="theory-box">

**FOPDT Model — Industry Standard for Process Control**

$$G(s) = \\frac{K e^{-Ls}}{\\tau s + 1}$$

| Parameter | Physical meaning | How to estimate from step response |
|-----------|-----------------|-----------------------------------|
| K | Process gain (y_ss / u_step) | Final value ÷ step amplitude |
| τ | Time constant | Time to reach 63.2% of final value (after L) |
| L | Dead time | Delay before any output change |

**Identification methods:**
- **Graphical (2-point):** mark 28.3% and 63.2% of y_ss on the curve → τ and L
- **Nonlinear least squares:** this simulator uses scipy's curve_fit (recommended)
- **Relay feedback (Åström-Hägglund):** auto-oscillation to find Ku and Tu in closed loop
- **Subspace methods (N4SID):** identify full state-space model from MIMO data

**After identification:** use Cohen-Coon, IMC, or SIMC rules to compute initial PID gains,
then fine-tune on plant. Always validate model on a **different** dataset than used for fitting.
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — Interview Q&A
# ─────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown(
        "Control engineering interview questions organized by experience level. "
        "Expand each question to see the concise interview-ready answer."
    )

    level = st.radio(
        "Experience level",
        ["🟢 Entry (0–2 yrs)", "🟡 Mid (2–6 yrs)", "🔴 Senior (6–12 yrs)", "🟣 Principal (12+ yrs)"],
        horizontal=True,
    )

    # ── Entry Level ─────────────────────────────────────────────────────────
    if "Entry" in level:
        st.markdown("### Entry Level — Fundamentals")

        questions = [
            ("What is the difference between open-loop and closed-loop control?",
             """**Open-loop:** No feedback. Output depends only on input (e.g., microwave timer).
Simple but cannot reject disturbances or handle model error.

**Closed-loop (feedback):** Measures output, compares to setpoint, corrects error.
Handles disturbances, model mismatch, and parameter drift.

*Interview tip:* Give a physical example — "A thermostat is closed-loop; a fixed heater timer is open-loop." """),

            ("Explain P, I, D and what each term does.",
             """**P (Proportional):** u = Kp·e  — acts on current error. Fast response, but leaves steady-state error (offset) for most plants.

**I (Integral):** u = Ki·∫e dt  — eliminates steady-state error by accumulating past errors. Risks integral windup, slows response.

**D (Derivative):** u = Kd·ė  — anticipates future error from its rate of change. Damps overshoot but amplifies noise.

*Rule of thumb:* P gets you close, I eliminates offset, D smooths the response."""),

            ("What is a transfer function?",
             """A transfer function G(s) = Y(s)/U(s) describes a **linear time-invariant (LTI)** system's input-output relationship in the Laplace domain, assuming zero initial conditions.

The **poles** (roots of denominator) determine stability and transient response.
The **zeros** (roots of numerator) shape how inputs are transmitted.

*Physical example:* A first-order thermal system G(s) = 1/(τs+1) — temperature reaches 63% of final value in one time constant τ."""),

            ("What is phase margin and why does it matter?",
             """Phase margin PM = 180° + ∠L(jω_gc) at the gain crossover frequency ω_gc.

**Why it matters:** PM ≈ 60° means the system can tolerate up to 60° of additional phase lag before going unstable. It's the most practical robustness measure.

**Rules of thumb:**
- PM > 45°: well-damped, good robustness
- PM ≈ 30°: borderline, may be lightly damped
- PM < 0°: unstable closed loop

*Approximate link to damping:* ζ ≈ PM / 100 (rough rule for second-order systems)."""),

            ("What is steady-state error? What determines it?",
             """Steady-state error = final value of error after transients die out = lim_{t→∞} e(t).

Determined by **system type** (number of integrators in open loop):
- Type 0: constant SS error for step input (offset) → need integral action
- Type 1 (1 integrator): zero SS error for step, constant error for ramp
- Type 2: zero SS error for step and ramp, constant error for parabolic

*Interview tip:* "Adding integral action (Ki > 0) makes the open loop a Type 1 system, guaranteeing zero steady-state error for step setpoints." """),

            ("How do you tune a PID using Ziegler-Nichols?",
             """**Ultimate gain method:**
1. Set Ki = Kd = 0
2. Increase Kp until sustained oscillation → record K_u (ultimate gain) and T_u (period)
3. Apply Z-N formulas: Kp = 0.6 K_u, Ti = T_u/2, Td = T_u/8
4. Ki = Kp/Ti, Kd = Kp·Td

**Result:** ~25% overshoot typical. Often detune by ×0.7 for less aggressiveness.

**Limitation:** Requires bringing system to edge of instability — not safe on all plants. Better alternatives: SIMC, IMC-based tuning, or model-based design."""),
        ]

        for q, a in questions:
            with st.expander(f"Q: {q}"):
                st.markdown(f'<div class="q-box entry-q">{a}</div>', unsafe_allow_html=True)

    # ── Mid Level ────────────────────────────────────────────────────────────
    elif "Mid" in level:
        st.markdown("### Mid Level — State-Space & Frequency Domain")

        questions = [
            ("What is the state-space representation and why use it over TFs?",
             """State-space: ẋ = Ax + Bu,  y = Cx + Du

**Advantages over TF:**
- Handles **MIMO** systems (multiple inputs/outputs)
- Natural framework for **LQR, Kalman filter, observer design**
- Easier for **nonlinear** extension (just replace A with f(x))
- Captures internal dynamics (controllable/observable subspaces)
- Numerically better conditioned than high-order TF polynomial arithmetic

**When to use TF:** SISO systems, classical loop-shaping, Bode/root-locus design."""),

            ("Explain controllability and observability.",
             """**Controllability (Kalman, 1960):** Can every state be driven to any desired value in finite time using admissible inputs?

Controllability matrix: C = [B, AB, A²B, ..., A^{n-1}B]
System is controllable iff rank(C) = n (full rank).

**Observability:** Can the initial state be uniquely determined from input-output observations?

Observability matrix: O = [C; CA; CA²; ...; CA^{n-1}]
System is observable iff rank(O) = n.

**Physical meaning:**
- Uncontrollable mode: control signal has no path to that state → can't be stabilised by feedback
- Unobservable mode: state affects no output → can't build an observer for it

*Interview tip:* "PBH test: (A, B) is controllable iff no left eigenvector of A is orthogonal to B." """),

            ("What is LQR and how do you tune Q and R?",
             """**LQR (Linear Quadratic Regulator):** optimal full-state feedback that minimises

J = ∫₀^∞ (x'Qx + u'Ru) dt

Solved via Algebraic Riccati Equation → gain K = R⁻¹B'P.

**Tuning Q and R:**
- **Bryson's rule:** Q_ii = 1/x_i,max², R_jj = 1/u_j,max²
  (Normalise by maximum acceptable state/input values)
- Large Q_ii → state i driven to zero aggressively (fast, high control effort)
- Large R → gentle control (slow, small inputs)
- Q/R ratio determines speed of response

**Practical approach:** Start with Bryson, then adjust Q for states you care most about. Use pole locus (Tab 6 in LQR page) to visualise pole movement vs Q/R ratio."""),

            ("What is the Separation Principle (LQG)?",
             """The Separation Principle states that for LQG (LQR + Kalman filter):

**You can design LQR (choose K) and Kalman filter (choose L) independently, then combine — the combined system is stable and optimal.**

Why? The augmented system has block-triangular structure:
eigenvalues = eig(A-BK) ∪ eig(A-LC) — fully decoupled!

**Practical implication:** First design LQR to meet performance specs, then design Kalman to meet estimation specs. No interaction.

**Limitation:** LQG can have poor robustness (no guaranteed gain/phase margins unlike pure LQR). Check robustness separately via Bode."""),

            ("What is anti-windup and when is it needed?",
             """**Windup problem:** When actuator saturates (e.g., valve at 100%), the integrator keeps accumulating error → when actuator comes out of saturation, large overshoot.

**Anti-windup strategies:**
1. **Clamping:** Stop integrating when actuator saturates
2. **Back-calculation:** u_actual − u_pid → fed back to slow integrator (standard method)
3. **Observer-based:** Model actuator saturation and compensate

*When needed:* Any PID with integral action on a plant with actuator limits (always in practice).
*Interview tip:* "I always add anti-windup in production — without it, startup or large setpoint changes can cause large overshoots." """),

            ("Explain the difference between Bode and Nyquist stability.",
             """Both analyse the open-loop L(s) = C(s)G(s).

**Bode:** Reads gain margin (at phase crossover) and phase margin (at gain crossover).
Simple, visual, gives direct tuning guidance. Limitation: can mislead for non-minimum-phase (NMP) systems with multiple gain crossovers.

**Nyquist:** Plots L(jω) in complex plane. The full **Nyquist criterion** counts clockwise encirclements of −1 to determine CL stability — works for any system including unstable open-loop.

*When Nyquist is essential:* Unstable open-loop plants (NMP, time delay), conditional stability, systems with RHP zeros."""),
        ]

        for q, a in questions:
            with st.expander(f"Q: {q}"):
                st.markdown(f'<div class="q-box mid-q">{a}</div>', unsafe_allow_html=True)

    # ── Senior Level ─────────────────────────────────────────────────────────
    elif "Senior" in level:
        st.markdown("### Senior Level — Advanced Design")

        questions = [
            ("Compare MPC, LQR, and H∞. When do you choose each?",
             """| | LQR | MPC | H∞ |
|---|---|---|---|
| Horizon | Infinite | Finite, receding | Infinite |
| Constraints | None | Hard constraints | None (implicit via weights) |
| MIMO | Yes | Yes | Yes |
| Online compute | None | QP at each step | None |
| Robustness | Good (GM, PM) | Problem-specific | Guaranteed (worst-case) |
| Nonlinear | Linearise | Re-linearise | Linearise |

**Choose LQR:** Unconstrained linear MIMO, full-state available, embedded/real-time
**Choose MPC:** Constraint handling critical, known future setpoints, MIMO with input/output constraints
**Choose H∞:** Robust stability under modelled uncertainty, worst-case disturbance rejection, loop-shaping for MIMO"""),

            ("What is dead-time and how do you handle it in control design?",
             """Dead time (transport delay) = time before plant responds to input, e^{-Ls}.

**Effect:** Adds −Lω degrees of phase lag → drastically reduces phase margin.
Rule: bandwidth ω_c < 0.5/L for stability.

**Handling strategies:**
1. **Detune PID** — reduce bandwidth (loss of performance)
2. **Smith Predictor** — removes delay from loop, best for simple SISO
3. **IMC (Internal Model Control)** — systematic design for delay-dominant plants
4. **MPC** — naturally handles delay by including it in prediction model
5. **Padé approximation** — approximate e^{-Ls} ≈ (1-Ls/2)/(1+Ls/2) for analysis

*Interview depth:* Be ready to derive the Smith Predictor block diagram and explain why it works (delay only on output, not in characteristic equation)."""),

            ("How do you analyse and design for a flexible structure (lightly damped modes)?",
             """**Challenge:** Flexible modes have resonances (peaks in Bode) that can cause instability if the controller excites them. Typical in aerospace, robotics, large antennas.

**Identification:** Experimental frequency response shows sharp peaks. Use ERA/OKID for state-space model.

**Design strategies:**
1. **Roll off before resonance:** Limit bandwidth below ω_n. Simple, conservative.
2. **Notch filter:** Add G_notch(s) = (s²+2ζ_z·ω_n·s+ω_n²)/(s²+2ζ_p·ω_n·s+ω_n²) — zeros cancel resonance. Sensitive to frequency uncertainty.
3. **Active damping (PPF/IRC):** Feedback from rate sensor to add damping to flexible mode.
4. **H∞ with multiplicative uncertainty:** Model uncertainty covers modal frequency variation, synthesise robust controller.

*Key insight:* Collocated sensors (near actuator) give guaranteed stability margin — use whenever possible."""),

            ("Explain gain scheduling and when you would use it.",
             """**Gain scheduling:** The controller gains K(θ) depend on a scheduling variable θ (e.g., speed, altitude, temperature) that indicates the operating condition.

**When to use:** Plant dynamics change significantly with operating point (nonlinear plant with known structure) — aircraft (speed/altitude), chemical reactors (conversion), automotive (gear/speed).

**Design procedure:**
1. Linearise plant at N operating points θ_1, ..., θ_N
2. Design LTI controller at each point
3. Interpolate gains vs θ (lookup table, polynomial, fuzzy)
4. Validate for transient between operating points

**Limitations:**
- No stability guarantee during transitions (unless LPV framework used)
- Scheduling variable must capture all relevant dynamics changes
- Grid must be dense enough — interpolation errors

*Modern alternative:* Linear Parameter-Varying (LPV) control gives formal stability proofs."""),

            ("What is system type and how does it relate to steady-state error?",
             """System type = number of open-loop poles at s=0 (integrators in open-loop TF L(s)).

| System Type | Step input e_ss | Ramp input e_ss | Parabolic e_ss |
|-------------|----------------|----------------|----------------|
| 0 | K_p = lim L(s) as s→0 | ∞ | ∞ |
| 1 | 0 | K_v = lim s·L(s) | ∞ |
| 2 | 0 | 0 | K_a = lim s²·L(s) |

**Practical:** A Type 1 system (one open-loop integrator) tracks steps with zero SS error. An I-only or PI controller makes the loop Type 1 even if the plant is Type 0.

*Interview trap:* "Does adding a plant integrator help?" Yes — DC motor (G=K_m/s·(τs+1)) is already Type 1; adding PID makes it Type 2 (zero error for ramps). But too many integrators destabilise (Routh-Hurwitz)."""),
        ]

        for q, a in questions:
            with st.expander(f"Q: {q}"):
                st.markdown(f'<div class="q-box senior-q">{a}</div>', unsafe_allow_html=True)

    # ── Principal Level ───────────────────────────────────────────────────────
    else:
        st.markdown("### Principal Level — Robust & Advanced Methods")

        questions = [
            ("Explain H∞ control design and the small gain theorem.",
             r"""**H∞ norm:** ||G||_∞ = sup_ω σ_max(G(jω)) — peak singular value of frequency response.

**H∞ problem:** Find controller C(s) to minimise ||T_zw||_∞ where T_zw is the closed-loop TF from disturbances w to performance outputs z.

**Mixed-sensitivity H∞ (S/KS/T):**
Find C to minimise ||[W_1·S; W_2·KS; W_3·T]||_∞ < γ
- W_1 shapes S (sensitivity) → disturbance rejection, SS error
- W_2 shapes KS (control effort) → actuator limits
- W_3 shapes T (complementary sensitivity) → noise rejection, robustness

**Small Gain Theorem:** Feedback loop is stable if ||Δ||·||G||_∞ < 1 for all perturbations Δ.
This is the foundation of robust stability analysis.

**Practical:** Use MATLAB's hinfsyn or Python's python-control. Start from loop-shaping weights, iterate."""),

            ("What is μ-synthesis and when does it outperform H∞?",
             r"""**Structured Singular Value μ:** Generalises H∞ to **structured** uncertainty Δ = blkdiag(δ_1·I, ..., δ_k·I).

μ(M) = 1/min{σ_max(Δ): det(I - MΔ) = 0}

**Why μ > H∞:** H∞ assumes unstructured Δ (worst case over ALL perturbations of same norm).
If uncertainty has known structure (e.g., ±10% in gain, ±20% in time constant, independently), μ is less conservative.

**DK-iteration (μ-synthesis):**
1. Solve H∞ problem (K step)
2. Scale uncertainty via D matrix to minimise μ upper bound (D step)
3. Repeat until convergence

**Use when:** MIMO system with multiple independent uncertainties, tight performance required, H∞ too conservative.

**Industrial use:** Flight control (multiple uncertain aerodynamic parameters), chemical reactors with model uncertainty."""),

            ("How do you approach control design for a MIMO system with strong coupling?",
             r"""**Step 1: Analyse plant at operating point**
- Compute RGA (Relative Gain Array): Λ = G ⊙ (G^{-1})^T (element-wise product)
- RGA diagonal ≈ 1 → low coupling, pair input i with output i
- RGA off-diagonal large → heavy coupling, reconsider pairing

**Step 2: Choose architecture**
- **Decentralised:** N×N PID loops (if RGA ≈ I at crossover)
- **Decoupler:** Pre-filter D(s) = G^{-1}(s)·Gd(s) to diagonalise — sensitive to model error
- **LQR/LQG:** Full MIMO state-feedback — handles coupling natively
- **H∞/MPC:** For constrained or robust MIMO

**Step 3: Validate coupling rejection**
- Closed-loop RGA should be close to identity at operating frequencies
- Check condition number of G(jω) — high condition number → ill-conditioned plant, hard to control

**Interview depth:** Be ready to compute and interpret a 2×2 RGA by hand."""),

            ("Explain model-based fault detection and isolation (FDI) in control systems.",
             """**Principle:** Use plant model to predict output; compare to actual output.
Residual r(t) = y(t) − ŷ(t) should be ≈ 0 in healthy operation.

**Parity equations:** r = Py + Qu where P selects direction orthogonal to fault-free subspace.

**Kalman-based FDI:** Innovation sequence ε = y − C·x̂. Under healthy conditions, ε is zero-mean white. Faults cause systematic bias or changed statistics.

**Design requirements:**
- **Sensitivity:** Faults must produce detectable residuals (design detection direction)
- **Robustness:** Disturbances/model error should not trigger false alarms (parity residuals must be orthogonal to disturbance directions)

**Methods:**
- CUSUM / GLR statistical tests on residuals
- Structured residuals (FDI filters) that are sensitive to specific faults
- Deep learning anomaly detection on residual time series

*Principal-level expectation:* Can specify detection threshold, FDI filter design, and integration with fault-tolerant control (FTC) to reconfigure around failed actuators/sensors."""),

            ("How do you validate a control system before deployment on a production asset?",
             """**Validation layers (V-model):**

1. **Unit testing:** Each module (plant model, observer, controller) tested in isolation
2. **Software-in-the-loop (SIL):** Full control algorithm simulated with plant model
3. **Hardware-in-the-loop (HIL):** Controller hardware, real-time plant simulation — tests timing, I/O, saturation
4. **Processor-in-the-loop (PIL):** Target processor in the loop — catches fixed-point/quantisation issues
5. **Factory Acceptance Test (FAT):** Control system tested against physical plant in controlled environment
6. **Site Acceptance Test (SAT):** Commissioned on production asset with limited operating envelope
7. **Production validation:** Phased rollout, monitoring, KPI tracking

**Key controls-specific checks:**
- Step tests at multiple operating points to validate model coverage
- Disturbance injection (bump tests) to measure rejection performance
- Actuator saturation / anti-windup validation
- Failure mode testing: sensor dropout, actuator loss, communication fault
- Gain/phase margin verification at deployment vs design

*Interview tip:* Always mention **functional safety (IEC 61508 / ISO 26262)** standards if applicable to the domain."""),
        ]

        for q, a in questions:
            with st.expander(f"Q: {q}"):
                st.markdown(f'<div class="q-box princ-q">{a}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — Extended Theory
# ─────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## Extended Control Theory Reference")

    t1, t2 = st.columns(2)

    with t1:
        st.markdown("### Cascade Control")
        st.markdown("""<div class="theory-box">

**When:** Plant can be split into fast inner loop (G₂) and slow outer loop (G₁).

**Architecture:**
- Outer controller C₁(s) generates setpoint for inner loop
- Inner controller C₂(s) controls G₂ at much higher bandwidth
- Inner loop rejects fast disturbances before they propagate to outer loop

**Tuning rule:** Inner loop bandwidth ≥ 3× outer loop bandwidth.

**Example:** Temperature control — inner loop controls heater power (fast), outer loop controls product temperature (slow). Inner loop rejects supply voltage disturbances.

**When NOT to use:** If inner measurement is noisy or inner loop can't be tuned faster. Single loop with feedforward may be better.
</div>""", unsafe_allow_html=True)

        st.markdown("### IMC (Internal Model Control)")
        st.markdown("""<div class="theory-box">

**Idea:** Design directly in terms of the desired closed-loop response.

For plant G(s), choose IMC filter Q(s):
$$C(s) = \\frac{Q(s)}{1 - Q(s)\\tilde{G}(s)}$$

where Q(s) = G_m⁻¹(s) · f(s) and f(s) = 1/(λs+1)^n is a low-pass filter.

**λ** (filter time constant) = sole tuning parameter:
- Small λ → fast, aggressive (less robust)
- Large λ → slow, gentle (more robust)

**IMC-PID tuning:** IMC can be algebraically converted to PID parameters — gives:
Kp = τ/(K·λ),  Ti = τ,  Td = θ/2  (for FOPDT plant)

This is the **SIMC rule** (Skogestad's simplified IMC) — arguably the best single-parameter PID tuning method.

**Limitation:** Perfect model → perfect control. Model errors cause offset. Add integral for offset-free tracking.
</div>""", unsafe_allow_html=True)

        st.markdown("### Discrete Control Gotchas")
        st.markdown("""<div class="theory-box">

**Sampling requirements:**
- Shannon: f_s > 2 × f_max (anti-aliasing)
- Control: f_s ≥ 10–20 × bandwidth (rule of thumb)

**Discretisation methods:**
| Method | Formula | Preserves |
|--------|---------|-----------|
| Euler (forward) | s = (z−1)/T | N/A — can destabilise |
| Euler (backward) | s = (z−1)/(Tz) | Stability |
| Tustin (bilinear) | s = 2(z−1)/(T(z+1)) | Frequency shape |
| ZOH | matrix exponential | DC gain + step response |

**Quantisation:** Fixed-point implementations need ≥12 bits for control gains; validate with PIL.

**Computational delay:** One sample delay in digital controller adds −ω·T phase lag — account for in margin calculations. Use Pade approximation or predict ahead by one step.
</div>""", unsafe_allow_html=True)

    with t2:
        st.markdown("### Feedforward Control")
        st.markdown("""<div class="theory-box">

**Concept:** Measure disturbance d directly; compute compensating input:
$$u_{ff}(s) = -G_d(s)/G(s) \\cdot d(s)$$

Cancels disturbance **before** it affects output (feedback can only react after).

**Combined feedback + feedforward:**
$$u = C(s)·e + G_d(s)/G(s)·d$$

**When to use:**
- Measurable, repeatable disturbances (e.g., load torque in motor drive, flow rate disturbance in reactor)
- Preview information known in advance (MPC naturally incorporates this)

**Limitation:** Requires accurate model of G and G_d. Model error → incomplete cancellation (but feedback handles the residual).
</div>""", unsafe_allow_html=True)

        st.markdown("### Nonlinear Control — Quick Survey")
        st.markdown("""<div class="theory-box">

| Method | Key idea | Example application |
|--------|---------|---------------------|
| **Feedback linearisation** | Exactly cancel nonlinearity via coordinate change | Robot arm computed torque |
| **Sliding mode (SMC)** | Drive state to sliding surface, maintain it | UAV attitude, power electronics |
| **Backstepping** | Recursively stabilise subsystems | Marine vessel control |
| **Gain scheduling** | LTI control at each operating point | Aircraft, engines |
| **Nonlinear MPC (NMPC)** | MPC with nonlinear model in QP | Autonomous driving, bioprocess |
| **Adaptive control** | On-line parameter estimation + update | Unknown plant, changing conditions |

**Lyapunov stability:** Nonlinear stability analysis. Find V(x) > 0 with dV/dt < 0 → globally asymptotically stable. LQR uses V = x'Px (quadratic Lyapunov function) for linear systems.
</div>""", unsafe_allow_html=True)

        st.markdown("### Process Control Hierarchy")
        st.markdown("""<div class="theory-box">

Industrial control systems operate in a **hierarchy:**

```
L4  Business planning  (ERP)      — hours to days
L3  Production scheduling (MES)   — minutes to hours
L2  Supervisory control (SCADA)   — seconds to minutes
L1  Regulatory control (DCS/PLC)  — 10ms to seconds   ← PID lives here
L0  Sensors + actuators (I/O)     — 1–10ms
```

**PID lives at Level 1.** MPC and optimisation typically at L2–L3.

**Key standards:**
- IEC 61511 — Functional Safety (process industry)
- IEC 62443 — Industrial Cybersecurity
- ISA-88 — Batch control
- ISA-95 — Enterprise–control integration
</div>""", unsafe_allow_html=True)
