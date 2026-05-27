"""
Model Predictive Control (MPC) — finite-horizon optimal control.

Discrete-time formulation
    x_{k+1} = Ad x_k + Bd u_k        (zero-order hold discretization)
    y_k     = C x_k

Prediction (Np steps, Nc control moves):
    Y = Psi · U + Phi · x_k
    Psi: forced response matrix (Np·ny × Nc·nu)
    Phi: free  response matrix  (Np·ny × n)

Last control input u_{Nc-1} is held constant for steps Nc, Nc+1, ..., Np-1
(standard "input blocking / ZOH extension" formulation).

Unconstrained cost:
    J = (Y - Yref)' Q̄ (Y - Yref) + U' R̄ U

Unconstrained solution:
    U* = (Psi'Q̄Psi + R̄)^{-1} Psi'Q̄ (Yref - Phi·x_k)

Receding horizon: only U*[0] is applied; U* is recomputed each step.
"""
import numpy as np
from scipy.linalg import expm
from typing import Dict, Optional, Tuple


# ── Discretisation ────────────────────────────────────────────────────────────

def discretize_ss(A: np.ndarray, B: np.ndarray, dt: float):
    """
    Zero-order-hold discretization via matrix exponential.
    Returns (Ad, Bd) such that x_{k+1} = Ad x_k + Bd u_k.
    """
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


# ── Prediction matrices ───────────────────────────────────────────────────────

def build_prediction_matrices(Ad: np.ndarray, Bd: np.ndarray,
                               C: np.ndarray, Np: int, Nc: int):
    """
    Build Phi (free-response) and Psi (forced-response) matrices.

    For i-th prediction step (y_{k+i+1}, i=0..Np-1) and j-th control input (u_j, j=0..Nc-1):

        Phi[i, :]    = C · Ad^{i+1}
        Psi[i, j]    = C · Ad^{i-j} · Bd            if j < Nc-1  and i >= j
        Psi[i, Nc-1] = C · Σ_{q=0}^{i-j} Ad^q · Bd if j = Nc-1  and i >= j
                        (hold last input: accumulates contributions)
    """
    n  = Ad.shape[0]
    ny = C.shape[0]
    nu = Bd.shape[1]

    Phi = np.zeros((Np * ny, n))
    Psi = np.zeros((Np * ny, Nc * nu))

    Ad_pow = [np.eye(n)]
    for _ in range(Np):
        Ad_pow.append(Ad_pow[-1] @ Ad)

    for i in range(Np):
        Phi[i * ny:(i + 1) * ny, :] = C @ Ad_pow[i + 1]

        for j in range(Nc):
            if i < j:
                continue
            if j < Nc - 1:
                Psi[i * ny:(i + 1) * ny, j * nu:(j + 1) * nu] = C @ Ad_pow[i - j] @ Bd
            else:
                # Last control move — accumulate held contributions
                acc = np.zeros((ny, nu))
                for q in range(i - j + 1):
                    acc += C @ Ad_pow[q] @ Bd
                Psi[i * ny:(i + 1) * ny, j * nu:(j + 1) * nu] = acc

    return Phi, Psi


# ── Cost matrices ─────────────────────────────────────────────────────────────

def build_cost_matrices(Q_y: np.ndarray, R_u: np.ndarray, Np: int, Nc: int):
    """Block-diagonal cost matrices Q̄ (Np·ny × Np·ny) and R̄ (Nc·nu × Nc·nu)."""
    return np.kron(np.eye(Np), Q_y), np.kron(np.eye(Nc), R_u)


# ── One-step MPC solve ────────────────────────────────────────────────────────

def solve_mpc_step(Phi: np.ndarray, Psi: np.ndarray,
                   Q_bar: np.ndarray, R_bar: np.ndarray,
                   x_k: np.ndarray, ref_vec: np.ndarray,
                   u_min: Optional[float] = None,
                   u_max: Optional[float] = None) -> Tuple[np.ndarray, bool]:
    """
    Solve unconstrained MPC at one time step; clip to [u_min, u_max] if given.

    Returns (U_star, constrained_flag).
    U_star has shape (Nc·nu,); first element is the control to apply.
    constrained_flag is True if clipping changed U_star.
    """
    rhs = ref_vec - Phi @ x_k
    H   = Psi.T @ Q_bar @ Psi + R_bar
    try:
        U_star = np.linalg.solve(H, Psi.T @ Q_bar @ rhs)
    except np.linalg.LinAlgError:
        U_star = np.zeros(Psi.shape[1])

    if u_min is None and u_max is None:
        return U_star, False

    U_clipped   = np.clip(U_star,
                          -np.inf if u_min is None else u_min,
                           np.inf if u_max is None else u_max)
    constrained = not np.allclose(U_star, U_clipped)
    return U_clipped, constrained


# ── Closed-loop simulation ────────────────────────────────────────────────────

def simulate_mpc(Ad: np.ndarray, Bd: np.ndarray, C: np.ndarray,
                 Np: int, Nc: int,
                 Q_y: np.ndarray, R_u: np.ndarray,
                 ref: float, t_end: float, dt: float,
                 ref_state: int = 0,
                 u_min: Optional[float] = None,
                 u_max: Optional[float] = None,
                 q_noise_std: float = 0.0,
                 seed: int = 42):
    """
    Run closed-loop MPC simulation (receding-horizon).

    Returns
    -------
    t              : (N,) time vector
    x_hist         : (N, n) state history
    y_hist         : (N, ny) output history
    u_hist         : (N,) control history
    constraint_hist: (N,) bool — True when u was clipped
    snapshots      : dict {k: {'t_pred', 'y_pred', 'u_pred'}} — prediction at 4 moments
    """
    n  = Ad.shape[0]
    ny = C.shape[0]
    nu = Bd.shape[1]

    t = np.arange(0.0, t_end, dt)
    N = len(t)

    Phi, Psi   = build_prediction_matrices(Ad, Bd, C, Np, Nc)
    Q_bar, R_bar = build_cost_matrices(Q_y, R_u, Np, Nc)
    ref_vec    = np.tile(np.array([ref] * ny), Np)

    x_hist          = np.zeros((N, n))
    y_hist          = np.zeros((N, ny))
    u_hist          = np.zeros(N)
    constraint_hist = np.zeros(N, dtype=bool)

    snap_indices = {N // 6, N // 3, N // 2, 2 * N // 3}
    snapshots: Dict = {}

    rng = np.random.default_rng(seed) if q_noise_std > 0.0 else None

    for k in range(N - 1):
        x_k = x_hist[k]

        U_star, const = solve_mpc_step(Phi, Psi, Q_bar, R_bar, x_k, ref_vec,
                                        u_min, u_max)
        u_k = float(U_star[0])
        u_hist[k]          = u_k
        constraint_hist[k] = const

        if k in snap_indices:
            Y_pred = (Psi @ U_star + Phi @ x_k).reshape(Np, ny)
            snapshots[k] = {
                "t_pred": t[k] + np.arange(1, Np + 1) * dt,
                "y_pred": Y_pred[:, 0],
                "u_pred": U_star.reshape(Nc, nu)[:, 0],
            }

        w               = rng.normal(0.0, q_noise_std, n) if rng else np.zeros(n)
        x_hist[k + 1]   = Ad @ x_k + (Bd @ np.array([[u_k]])).flatten() + w
        y_hist[k]       = (C @ x_k).flatten()

    y_hist[-1] = (C @ x_hist[-1]).flatten()
    u_hist[-1] = u_hist[-2]
    return t, x_hist, y_hist, u_hist, constraint_hist, snapshots


# ── Smith Predictor simulation (FOPDT plant, continuous Euler) ────────────────

def simulate_smith_predictor(K: float, tau: float, L: float, dt: float,
                              Kp: float, Ki: float, Kd: float, N_filt: float,
                              ref: float, t_end: float):
    """
    Discrete Euler simulation of PID with Smith Predictor on a FOPDT plant.
    G(s) = K/(τs+1) with dead-time L.

    Discrete model (exact ZOH):  y[k+1] = a·y[k] + b·u_delayed[k]
    where a = exp(-dt/τ),  b = K·(1 - a)

    Also simulates standard PID (no SP) for comparison.

    Returns (t, y_sp, y_std, u_sp, u_std).
    """
    n_steps   = int(t_end / dt)
    n_delay   = max(0, int(round(L / dt)))
    t         = np.arange(n_steps) * dt
    a         = np.exp(-dt / tau) if tau > 1e-9 else 0.0
    b         = K * (1.0 - a)

    # ── Smith Predictor ──────────────────────────────────────────────────────
    y_sp      = np.zeros(n_steps)   # true plant output
    ym_sp     = np.zeros(n_steps)   # undelayed model output
    u_sp      = np.zeros(n_steps)
    integ_sp  = 0.0
    prev_e_sp = 0.0

    # ── Standard PID ─────────────────────────────────────────────────────────
    y_std     = np.zeros(n_steps)
    u_std     = np.zeros(n_steps)
    integ_std = 0.0
    prev_e_std = 0.0

    for k in range(n_steps - 1):
        u_del_sp  = u_sp[max(0, k - n_delay)]   # delayed input to true plant
        u_del_std = u_std[max(0, k - n_delay)]

        # ── Smith Predictor error ────────────────────────────────────────
        ym_del = ym_sp[max(0, k - n_delay)]
        e_sp   = ref - (y_sp[k] + ym_del - ym_sp[k])

        integ_sp    += e_sp * dt
        d_sp         = N_filt * (e_sp - prev_e_sp) / (1.0 + N_filt * dt)
        u_sp[k]      = Kp * e_sp + Ki * integ_sp + Kd * d_sp
        prev_e_sp    = e_sp

        y_sp[k + 1]  = a * y_sp[k]  + b * u_del_sp
        ym_sp[k + 1] = a * ym_sp[k] + b * u_sp[k]

        # ── Standard PID error ───────────────────────────────────────────
        e_std        = ref - y_std[k]
        integ_std   += e_std * dt
        d_std        = N_filt * (e_std - prev_e_std) / (1.0 + N_filt * dt)
        u_std[k]     = Kp * e_std + Ki * integ_std + Kd * d_std
        prev_e_std   = e_std

        y_std[k + 1] = a * y_std[k] + b * u_del_std

    return t, y_sp, y_std, u_sp, u_std


# ── System identification: FOPDT fit ─────────────────────────────────────────

def fopdt_model(t: np.ndarray, K: float, tau: float, L: float) -> np.ndarray:
    """First-order plus dead-time step response (amplitude = 1)."""
    y = np.zeros_like(t, dtype=float)
    mask = t >= L
    if tau > 1e-9:
        y[mask] = K * (1.0 - np.exp(-(t[mask] - L) / tau))
    else:
        y[mask] = K
    return y


def identify_fopdt(t: np.ndarray, y: np.ndarray):
    """
    Fit FOPDT model K/(τs+1)·e^{-Ls} to step-response data.

    Returns (K_id, tau_id, L_id, y_fit) or raises RuntimeError on failure.
    """
    from scipy.optimize import curve_fit

    y_ss   = float(y[-1])
    K_init = y_ss if abs(y_ss) > 1e-6 else 1.0

    # Estimate L as time when y first exceeds 5% of y_ss
    mask5 = y >= 0.05 * abs(y_ss)
    L_init = float(t[mask5][0]) if mask5.any() else 0.0

    # Estimate τ from 63% rise time minus dead time
    mask63 = y >= 0.632 * abs(y_ss)
    t63    = float(t[mask63][0]) if mask63.any() else float(t[-1]) / 3
    tau_init = max(0.05, t63 - L_init)

    p0     = [K_init, tau_init, L_init]
    bounds = ([0.0, 1e-3, 0.0], [abs(K_init) * 5 + 1, t[-1], t[-1] / 2])

    popt, _ = curve_fit(fopdt_model, t, y, p0=p0, bounds=bounds,
                        maxfev=10000)
    K_id, tau_id, L_id = popt
    y_fit = fopdt_model(t, K_id, tau_id, L_id)
    return K_id, tau_id, L_id, y_fit
