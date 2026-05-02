"""
Control Systems Simulator
=========================
Interactive tools for control theory analysis and design.

Pages
-----
  1_Control_Simulator  — PID tuning, 6 plant models, Bode plot,
                         root locus, stability analysis, filter design
  2_LQR_LQG            — LQR optimal control, Kalman filter, full
                         LQG loop across 4 state-space plants
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
    "Interactive PID tuning · LQR/LQG optimal control · Bode &amp; root locus · Kalman filter"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📐 PID Control Simulator")
    st.markdown(
        """
        - 6 plant models (first-order, second-order, integrator, unstable, dead-time, flexible)
        - Interactive P / I / D gain sliders with live step response
        - Bode plot, root locus, gain & phase margin
        - Stability analysis and filter design
        """
    )
    if st.button("Open PID Simulator →", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Control_Simulator.py")

with col2:
    st.markdown("### 🎯 LQR / LQG Optimal Control")
    st.markdown(
        """
        - LQR state-feedback design with Q / R weight tuning
        - Kalman filter observer with process & measurement noise
        - Full LQG loop simulation across 4 state-space plants
        - Eigenvalue plots and cost comparison
        """
    )
    if st.button("Open LQR/LQG →", use_container_width=True, type="primary"):
        st.switch_page("pages/2_LQR_LQG.py")

st.markdown("---")
st.caption("Built with Streamlit · NumPy · SciPy · Plotly")
