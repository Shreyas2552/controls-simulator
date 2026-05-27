"""
Advanced Control Strategies — computation module.

Functions
---------
simulate_cascade        Cascade two-loop PI control vs single-loop PI
simc_pid_fopdt          SIMC/IMC tuning for FOPDT plant
simc_pid_2nd_order      SIMC tuning for 2nd-order + dead-time plant
zn_pid_fopdt            Ziegler-Nichols tuning for FOPDT plant
simulate_pid_fopdt      FOPDT plant under PID, Euler-discrete
simulate_feedforward    Feedback-only vs perfect FF vs mismatched FF
simulate_smc            Sliding-mode control on mass-spring-damper
simulate_relay          Bang-bang (relay) control on mass-spring-damper
"""

import numpy as np
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fopdt_step(y: float, u_delayed: float, a: float, b: float) -> float:
    """One Euler step of exact ZOH FOPDT model."""
    return a * y + b * u_delayed


def _pid_update(e: float, e_prev: float, integ: float, dt: float,
                Kp: float, Ki: float, Kd: float, N: float = 20.0,
                d_prev: float = 0.0) -> Tuple[float, float, float]:
    """Discrete PID with derivative filter. Returns (u, new_integ, new_d_prev)."""
    integ_new = integ + e * dt
    d_filt = N * (e - e_prev) / (1.0 + N * dt)
    u = Kp * e + Ki * integ_new + Kd * d_filt
    return u, integ_new, d_filt


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cascade control
# ─────────────────────────────────────────────────────────────────────────────

def simulate_cascade(
    K1: float, tau1: float,          # outer plant G1 = K1/(tau1·s+1)
    K2: float, tau2: float,          # inner plant G2 = K2/(tau2·s+1)
    Kp_out: float, Ki_out: float,    # outer PI
    Kp_in:  float, Ki_in:  float,    # inner PI
    disturbance: float,              # step disturbance injected between G2 and G1
    dist_time: float,                # time at which disturbance is applied
    ref: float, t_end: float, dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cascade control: r → C_outer → y2_ref → C_inner → G2 → [+d] → G1 → y1.

    Returns (t, y1_casc, y1_single, u_casc).
    y1_single: same outer PI, inner loop replaced by direct actuator (no inner loop).
    """
    n = int(t_end / dt)
    t = np.arange(n) * dt

    a1 = np.exp(-dt / tau1) if tau1 > 1e-9 else 0.0
    b1 = K1 * (1.0 - a1)
    a2 = np.exp(-dt / tau2) if tau2 > 1e-9 else 0.0
    b2 = K2 * (1.0 - a2)

    # ── Cascade ──────────────────────────────────────────────────────────────
    y1_c = np.zeros(n)
    y2_c = np.zeros(n)
    u_c  = np.zeros(n)
    int_out = 0.0
    int_in  = 0.0
    e_out_prev = 0.0

    # ── Single-loop (equivalent gain approximation: C_out alone) ─────────────
    # Equivalent plant: G_eq = G1*G2 in series, same outer PI
    K_eq   = K1 * K2
    tau_eq = tau1 + tau2
    a_eq   = np.exp(-dt / tau_eq) if tau_eq > 1e-9 else 0.0
    b_eq   = K_eq * (1.0 - a_eq)
    y1_s   = np.zeros(n)
    u_s    = np.zeros(n)
    int_sl = 0.0
    e_sl_prev = 0.0

    for k in range(n - 1):
        d_k = disturbance if t[k] >= dist_time else 0.0

        # ── Cascade ─────────────────────────────────────────────────────────
        e_out = ref - y1_c[k]
        int_out += e_out * dt
        y2_ref = Kp_out * e_out + Ki_out * int_out

        e_in = y2_ref - y2_c[k]
        int_in += e_in * dt
        u_c[k] = Kp_in * e_in + Ki_in * int_in

        y2_next = a2 * y2_c[k] + b2 * u_c[k]
        y1_c[k + 1] = a1 * y1_c[k] + b1 * (y2_next + d_k)
        y2_c[k + 1] = y2_next
        e_out_prev = e_out

        # ── Single loop ──────────────────────────────────────────────────────
        e_sl = ref - y1_s[k]
        int_sl += e_sl * dt
        u_s[k] = Kp_out * e_sl + Ki_out * int_sl
        y1_s[k + 1] = a_eq * y1_s[k] + b_eq * (u_s[k] + d_k / K_eq)
        e_sl_prev = e_sl

    u_c[-1] = u_c[-2]
    return t, y1_c, y1_s, u_c


# ─────────────────────────────────────────────────────────────────────────────
# 2. SIMC / Ziegler-Nichols tuning
# ─────────────────────────────────────────────────────────────────────────────

def simc_pid_fopdt(K: float, tau: float, theta: float,
                   lambda_c: float) -> Tuple[float, float, float]:
    """
    SIMC (Skogestad IMC) PI tuning for FOPDT G(s) = K·e^{-θs}/(τs+1).

    Returns (Kp, Ki, Kd=0).
    Kp  = τ / (K·(λ + θ))
    Ti  = min(τ, 4·(λ + θ))
    """
    denom = K * (lambda_c + theta)
    Kp = tau / denom if denom > 1e-9 else 1.0
    Ti = min(tau, 4.0 * (lambda_c + theta))
    Ki = Kp / Ti if Ti > 1e-9 else 0.0
    return Kp, Ki, 0.0


def simc_pid_2nd_order(K: float, tau1: float, tau2: float, theta: float,
                        lambda_c: float) -> Tuple[float, float, float]:
    """
    SIMC PID tuning for 2nd-order + dead-time:  G = K/((τ1s+1)(τ2s+1))·e^{-θs}.

    Kp = τ1 / (K·(λ + θ')),  τ' = θ + τ2/2 (half-rule),  Ti = τ1,  Td = τ2
    Returns (Kp, Ki, Kd).
    """
    theta_eff = theta + tau2 / 2.0
    tau1_eff  = tau1
    denom = K * (lambda_c + theta_eff)
    Kp = tau1_eff / denom if denom > 1e-9 else 1.0
    Ti = tau1_eff
    Td = tau2 / 2.0
    Ki = Kp / Ti if Ti > 1e-9 else 0.0
    Kd = Kp * Td
    return Kp, Ki, Kd


def zn_pid_fopdt(K: float, tau: float, theta: float) -> Tuple[float, float, float]:
    """
    Ziegler-Nichols (process-reaction-curve) PID tuning for FOPDT.
    Returns (Kp, Ki, Kd).
    """
    if theta < 1e-9:
        theta = 0.01 * tau
    Kp = 1.2 * tau / (K * theta)
    Ti = 2.0 * theta
    Td = 0.5 * theta
    Ki = Kp / Ti if Ti > 1e-9 else 0.0
    Kd = Kp * Td
    return Kp, Ki, Kd


def simulate_pid_fopdt(K: float, tau: float, theta: float,
                        Kp: float, Ki: float, Kd: float,
                        ref: float, t_end: float, dt: float,
                        label: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Closed-loop step response for FOPDT plant under PID.
    Returns (t, y, u).
    """
    n       = int(t_end / dt)
    n_delay = max(0, int(round(theta / dt)))
    t       = np.arange(n) * dt

    a = np.exp(-dt / tau) if tau > 1e-9 else 0.0
    b = K * (1.0 - a)

    y    = np.zeros(n)
    u    = np.zeros(n)
    intg = 0.0
    e_prev = 0.0
    d_prev = 0.0

    for k in range(n - 1):
        u_del = u[max(0, k - n_delay)]
        e     = ref - y[k]
        u_k, intg, d_prev = _pid_update(e, e_prev, intg, dt, Kp, Ki, Kd,
                                         d_prev=d_prev)
        u[k]     = u_k
        y[k + 1] = a * y[k] + b * u_del
        e_prev   = e

    u[-1] = u[-2]
    return t, y, u


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feedforward control
# ─────────────────────────────────────────────────────────────────────────────

def simulate_feedforward(
    K_G: float, tau_G: float,      # process G(s) = K_G/(tau_G·s+1)
    K_Gd: float, tau_Gd: float,   # disturbance path Gd(s) = K_Gd/(tau_Gd·s+1)
    ff_gain_mismatch: float,       # actual FF gain = (K_Gd/K_G)*ff_gain_mismatch (1.0=perfect)
    Kp: float, Ki: float,         # feedback PI gains
    dist_amp: float,              # disturbance step amplitude
    dist_time: float,             # disturbance step time
    ref: float, t_end: float, dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate three scenarios:
      1. Feedback-only (PI)
      2. Perfect feedforward (ff_gain_mismatch = 1.0)
      3. Mismatched feedforward (ff_gain_mismatch != 1.0)

    All use same feedback PI.
    Returns (t, y_fb, y_ff_perfect, y_ff_mismatch).
    """
    n   = int(t_end / dt)
    t   = np.arange(n) * dt
    a_G  = np.exp(-dt / tau_G)  if tau_G  > 1e-9 else 0.0
    b_G  = K_G  * (1.0 - a_G)
    a_Gd = np.exp(-dt / tau_Gd) if tau_Gd > 1e-9 else 0.0
    b_Gd = K_Gd * (1.0 - a_Gd)

    ff_gain_perfect  = K_Gd / K_G if abs(K_G) > 1e-9 else 1.0
    ff_gain_actual   = ff_gain_perfect * ff_gain_mismatch

    def _run(use_ff: bool, ff_g: float):
        # Separate process output (y_G, driven by u) from disturbance output (y_d, driven by d).
        # Mixing them into a single state and applying a_G to both causes incorrect dynamics
        # for the disturbance path when tau_G != tau_Gd.
        y_G  = np.zeros(n)
        y_d  = np.zeros(n)
        u    = np.zeros(n)
        intg = 0.0

        for k in range(n - 1):
            d_k = dist_amp if t[k] >= dist_time else 0.0
            y_k = y_G[k] + y_d[k]      # total measured output

            e      = ref - y_k
            intg  += e * dt
            u_fb   = Kp * e + Ki * intg
            u_ff   = ff_g * d_k if use_ff else 0.0
            u[k]   = u_fb - u_ff       # FF subtracts to pre-cancel disturbance effect

            y_G[k + 1] = a_G  * y_G[k] + b_G  * u[k]
            y_d[k + 1] = a_Gd * y_d[k] + b_Gd * d_k

        return y_G + y_d

    y_fb         = _run(use_ff=False, ff_g=0.0)
    y_ff_perfect = _run(use_ff=True,  ff_g=ff_gain_perfect)
    y_ff_mismatch = _run(use_ff=True, ff_g=ff_gain_actual)
    return t, y_fb, y_ff_perfect, y_ff_mismatch


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sliding Mode Control (SMC) — mass-spring-damper
# ─────────────────────────────────────────────────────────────────────────────

def simulate_smc(
    m: float, k: float, b: float,  # mass, spring, damper
    c_smc: float,                  # sliding surface slope s = ė + c·e
    K_smc: float,                  # switching gain
    phi: float,                    # boundary layer width (saturation instead of sign)
    ref: float, t_end: float, dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    SMC on m·ẍ + b·ẋ + k·x = u.
    Sliding surface: s = ė + c·e,  control: u = m·(ref_ddot + c·ė - b/m·ẋ - k/m·x) - K·sat(s/φ)
    Returns (t, x, xdot, u, s_hist).
    """
    n = int(t_end / dt)
    t = np.arange(n) * dt

    x    = np.zeros(n)
    xdot = np.zeros(n)
    u    = np.zeros(n)
    s_h  = np.zeros(n)

    def sat(v: float, bnd: float) -> float:
        if bnd < 1e-9:
            return float(np.sign(v))
        return float(np.clip(v / bnd, -1.0, 1.0))

    for k in range(n - 1):
        e  = ref - x[k]
        de = -xdot[k]           # ė = ref_dot(=0) - ẋ
        s  = de + c_smc * e
        s_h[k] = s

        # Equivalent control: cancels plant dynamics so ṡ = 0 on the surface.
        # Derived from m·ẍ + b·ẋ + k·x = u with ṡ = 0 → ẍ = c·de:
        #   u_eq = m·c·de + b·ẋ + k·x   (note: +b and +k, not minus)
        u_eq = m * c_smc * de + b * xdot[k] + k * x[k]
        # Switching: u_sw must share sign with s so that s·ṡ = s·(−u_sw/m) < 0
        # → u_sw = +K·sat(s/φ)   (same sign as s → ṡ negative for s>0, positive for s<0)
        u_sw = K_smc * sat(s, phi)
        u[k] = u_eq + u_sw

        # Euler integration
        xddot = (u[k] - b * xdot[k] - k * x[k]) / m
        xdot[k + 1] = xdot[k] + xddot * dt
        x[k + 1]    = x[k]    + xdot[k] * dt

    s_h[-1] = s_h[-2]
    u[-1]   = u[-2]
    return t, x, xdot, u, s_h


def simulate_relay(
    m: float, k: float, b: float,  # mass, spring, damper
    u_amp: float,                   # relay corrective amplitude (above equilibrium bias)
    ref: float, t_end: float, dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Biased bang-bang (relay) control on spring-mass-damper.

    u = k·ref + u_amp·sign(e)
    The bias k·ref provides the steady-state force to hold the spring at ref.
    u_amp is the corrective switching amplitude around that equilibrium.

    Returns (t, x, u).
    """
    n = int(t_end / dt)
    t = np.arange(n) * dt
    x    = np.zeros(n)
    xdot = np.zeros(n)
    u    = np.zeros(n)
    u_bias = k * ref      # equilibrium force to hold spring at setpoint

    for ki in range(n - 1):
        e     = ref - x[ki]
        u_sw  = u_amp * float(np.sign(e)) if abs(e) > 1e-9 else 0.0
        u[ki] = u_bias + u_sw
        xddot = (u[ki] - b * xdot[ki] - k * x[ki]) / m
        xdot[ki + 1] = xdot[ki] + xddot * dt
        x[ki + 1]    = x[ki]    + xdot[ki] * dt

    u[-1] = u[-2]
    return t, x, u
