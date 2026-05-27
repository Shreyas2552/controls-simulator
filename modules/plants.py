import numpy as np
from typing import Dict, Tuple, List

PLANT_MODELS: Dict = {
    "First Order": {
        "tf_display": "G(s) = K / (τs + 1)",
        "physical_context": "Thermal systems, RC circuits, first-order chemical reactors, simple tank level",
        "params": {
            "K":   {"label": "DC Gain (K)",           "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
            "tau": {"label": "Time Constant τ (s)",    "default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1},
        },
    },
    "Second Order": {
        "tf_display": "G(s) = K·ωₙ² / (s² + 2ζωₙs + ωₙ²)",
        "physical_context": "Mass-spring-damper, RLC circuit, servo motor, electromechanical systems",
        "params": {
            "K":    {"label": "DC Gain (K)",              "default": 1.0,  "min": 0.1,  "max": 10.0, "step": 0.1},
            "wn":   {"label": "Natural Freq ωₙ (rad/s)",  "default": 2.0,  "min": 0.1,  "max": 20.0, "step": 0.1},
            "zeta": {"label": "Damping Ratio ζ",           "default": 0.5,  "min": 0.01, "max": 2.0,  "step": 0.01},
        },
    },
    "DC Motor (Position)": {
        "tf_display": "G(s) = Km / (s(τₘs + 1))",
        "physical_context": "DC servo motor with position output — type-1 system with natural integrator",
        "params": {
            "Km":    {"label": "Motor Gain (Km)",          "default": 1.0, "min": 0.1,  "max": 5.0, "step": 0.1},
            "tau_m": {"label": "Motor Time Const τₘ (s)",  "default": 0.5, "min": 0.01, "max": 5.0, "step": 0.01},
        },
    },
    "Integrating Plant": {
        "tf_display": "G(s) = K / s",
        "physical_context": "Pure integrator: liquid level control, velocity→position, robot joint angle",
        "params": {
            "K": {"label": "Integrator Gain (K)", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
        },
    },
    "Unstable Plant": {
        "tf_display": "G(s) = K / (s − a)",
        "physical_context": "Linearised inverted pendulum, exothermic CSTR, unstable aircraft pitch mode",
        "params": {
            "K": {"label": "Gain (K)",              "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1},
            "a": {"label": "Unstable Pole a (> 0)", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1},
        },
    },
    "Third Order": {
        "tf_display": "G(s) = K / ((τ₁s+1)(τ₂s+1)(τ₃s+1))",
        "physical_context": "Cascaded industrial processes, multi-stage heat exchangers, complex fluid systems",
        "params": {
            "K":    {"label": "DC Gain (K)",           "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
            "tau1": {"label": "Time Constant τ₁ (s)",  "default": 3.0, "min": 0.1, "max": 20.0, "step": 0.1},
            "tau2": {"label": "Time Constant τ₂ (s)",  "default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1},
            "tau3": {"label": "Time Constant τ₃ (s)",  "default": 0.5, "min": 0.1, "max":  5.0, "step": 0.1},
        },
    },
    "Dead-time (Padé 1st)": {
        "tf_display": "G(s) = K·e^{−Ls}/(τs+1)  ≈  K(2−Ls)/((2+Ls)(τs+1))",
        "physical_context": (
            "Process control with transport delay: pipelines, heat exchangers, chemical reactors. "
            "Padé 1st order: accurate when L ≪ τ. Adds RHP zero at s=2/L → initial undershoot."
        ),
        "params": {
            "K":   {"label": "DC Gain (K)",          "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
            "tau": {"label": "Time Constant τ (s)",  "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1},
            "L":   {"label": "Dead Time L (s)",       "default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1},
        },
    },
    "Flexible / Resonant": {
        "tf_display": "G(s) = K / ((τs+1)(s²/ωₙ² + 2ζ/ωₙ·s + 1))",
        "physical_context": (
            "Servo with structural resonance, robot arm with joint compliance, "
            "satellite antenna with flexible appendage. The resonant mode limits achievable bandwidth."
        ),
        "params": {
            "K":    {"label": "DC Gain (K)",              "default": 1.0, "min": 0.1,  "max": 5.0,  "step": 0.1},
            "tau":  {"label": "Lag τ (s)",                "default": 0.5, "min": 0.01, "max": 5.0,  "step": 0.05},
            "wn":   {"label": "Resonant ωₙ (rad/s)",      "default": 5.0, "min": 0.5,  "max": 50.0, "step": 0.5},
            "zeta": {"label": "Modal Damping ζ",           "default": 0.05,"min": 0.01, "max": 0.5,  "step": 0.01},
        },
    },
}


def get_plant_tf(plant_name: str, params: Dict) -> Tuple[List[float], List[float]]:
    """Return (numerator, denominator) coefficient lists, highest power first."""
    if plant_name == "First Order":
        return [float(params["K"])], [float(params["tau"]), 1.0]

    if plant_name == "Second Order":
        K, wn, z = params["K"], params["wn"], params["zeta"]
        return [float(K * wn**2)], [1.0, float(2*z*wn), float(wn**2)]

    if plant_name == "DC Motor (Position)":
        return [float(params["Km"])], [float(params["tau_m"]), 1.0, 0.0]

    if plant_name == "Integrating Plant":
        return [float(params["K"])], [1.0, 0.0]

    if plant_name == "Unstable Plant":
        return [float(params["K"])], [1.0, float(-params["a"])]

    if plant_name == "Third Order":
        d = np.polymul(
            np.polymul([float(params["tau1"]), 1.0], [float(params["tau2"]), 1.0]),
            [float(params["tau3"]), 1.0]
        )
        return [float(params["K"])], d.tolist()

    if plant_name == "Dead-time (Padé 1st)":
        K, tau, L = float(params["K"]), float(params["tau"]), float(params["L"])
        if L < 1e-6:
            return [K], [tau, 1.0]
        # e^{-Ls} ≈ (1 - Ls/2) / (1 + Ls/2)
        # G_padé = K * (-L/2·s + 1) / ((L/2·s + 1)(τs + 1))
        pade_n = np.array([-L / 2, 1.0])
        pade_d = np.array([ L / 2, 1.0])
        num = K * pade_n
        den = np.polymul(pade_d, [tau, 1.0])
        return num.tolist(), den.tolist()

    if plant_name == "Flexible / Resonant":
        K, tau, wn, zeta = (float(params["K"]), float(params["tau"]),
                            float(params["wn"]), float(params["zeta"]))
        # G = K / ((τs+1)(s²/ωₙ²+ 2ζ/ωₙ·s + 1))
        # Denominator = (τs+1)(s²/ωₙ²+ 2ζ/ωₙ·s + 1)
        lag_d = [tau, 1.0]
        res_d = [1.0 / wn**2, 2.0 * zeta / wn, 1.0]
        den = np.polymul(lag_d, res_d)
        return [K], den.tolist()

    return [1.0], [1.0]
