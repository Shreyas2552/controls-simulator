"""
Control Systems Simulator
=========================
Interactive tools for control theory analysis and design.

Pages
-----
  1_Control_Simulator  — PID tuning, 8 plant models, Bode, Nyquist,
                         root locus, stability analysis, filter design
  2_LQR_LQG            — LQR optimal control, Kalman filter, full
                         LQG loop across 4 state-space plants
  3_MPC                — Model Predictive Control, finite-horizon
                         constrained optimal control vs LQR
  4_Advanced_Control   — Smith predictor, system identification,
                         gain scheduling, interview Q&A (entry→principal)
"""

import streamlit as st

st.set_page_config(
    page_title="Control Systems Simulator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("## ⚙️ Control Systems Simulator")
st.markdown(
    '<p style="color:#8b949e; margin-top:-12px; font-size:0.95rem;">'
    "Interactive PID · LQR/LQG · MPC · Smith Predictor · System ID · Interview Prep"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📐 PID Control Simulator")
    st.markdown(
        """
        - 8 plant models (incl. dead-time Padé, flexible/resonant mode)
        - Live step response · Bode · **Nyquist diagram** · root locus
        - Gain & phase margin, stability analysis, filter design
        - Control theory quick-reference guide
        """
    )
    if st.button("Open PID Simulator →", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Control_Simulator.py")

with col2:
    st.markdown("### 🎯 LQR / LQG Optimal Control")
    st.markdown(
        """
        - LQR state-feedback with Q / R weight tuning and pole locus
        - Kalman filter observer with process & measurement noise
        - Full LQG loop — Separation Principle demonstrated
        - Eigenvalue plots and cost comparison across 4 state-space plants
        """
    )
    if st.button("Open LQR/LQG →", use_container_width=True, type="primary"):
        st.switch_page("pages/2_LQR_LQG.py")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🤖 Model Predictive Control (MPC)")
    st.markdown(
        """
        - Finite-horizon receding-horizon optimal control
        - Input constraints (u_min / u_max) with violation highlighting
        - Prediction window snapshots at 4 time instants
        - MPC vs LQR comparison · Horizon sweep (Np) analysis
        - Full MPC theory: cost function, Riccati, stability notes
        """
    )
    if st.button("Open MPC →", use_container_width=True, type="primary"):
        st.switch_page("pages/3_MPC.py")

with col4:
    st.markdown("### 🎓 Advanced Topics & Interview Prep")
    st.markdown(
        """
        - **Smith Predictor** — dead-time compensation vs standard PID
        - **System Identification** — FOPDT fit + Cohen-Coon auto-tuning
        - **Interview Q&A** — entry → principal level, expandable answers
        - Extended theory: cascade, IMC, feedforward, nonlinear survey
        """
    )
    if st.button("Open Advanced Topics →", use_container_width=True, type="primary"):
        st.switch_page("pages/4_Advanced_Control.py")

col5, col6 = st.columns(2)

with col5:
    st.markdown("### 🔗 Control Strategies")
    st.markdown(
        """
        - **Cascade control** — nested PI loops with disturbance rejection demo
        - **IMC / SIMC tuning** — λ-based auto-tuning vs Ziegler-Nichols comparison
        - **Feedforward control** — perfect & mismatched FF vs feedback-only
        - **Nonlinear control** — Sliding Mode (SMC) + relay on mass-spring-damper
        """
    )
    if st.button("Open Control Strategies →", use_container_width=True, type="primary"):
        st.switch_page("pages/5_Control_Strategies.py")

with col6:
    st.markdown("### 📖 Coming Soon")
    st.markdown(
        """
        - H∞ robust control with loop-shaping weights
        - MIMO / RGA analysis for 2×2 plants
        - Gain scheduling interactive simulation
        - Iterative Learning Control (ILC)
        """
    )
    st.info("More pages in development — contributions welcome on GitHub.")

st.markdown("---")
st.caption("Built with Streamlit · NumPy · SciPy · Plotly")
