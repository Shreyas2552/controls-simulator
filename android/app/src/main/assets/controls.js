/**
 * controls.js — Pure-JavaScript Control Systems Algorithms
 * Mirrors the Python modules (plants.py, analysis.py, pid_controller.py, filters.py)
 * No external dependencies except Plotly (for charts).
 */

"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// Complex number helpers  [re, im]
// ─────────────────────────────────────────────────────────────────────────────
const C = {
    of: (re, im = 0) => [re, im],
    add: ([ar, ai], [br, bi]) => [ar + br, ai + bi],
    sub: ([ar, ai], [br, bi]) => [ar - br, ai - bi],
    mul: ([ar, ai], [br, bi]) => [ar * br - ai * bi, ar * bi + ai * br],
    div([ar, ai], [br, bi]) {
        const d = br * br + bi * bi;
        return [(ar * br + ai * bi) / d, (ai * br - ar * bi) / d];
    },
    abs: ([r, i]) => Math.hypot(r, i),
    arg: ([r, i]) => Math.atan2(i, r),
    scale: ([r, i], s) => [r * s, i * s],
};

// ─────────────────────────────────────────────────────────────────────────────
// Polynomial arithmetic  (coefficients highest-power first)
// ─────────────────────────────────────────────────────────────────────────────
const Poly = {
    // Evaluate at real x (Horner)
    val(p, x) { return p.reduce((acc, c) => acc * x + c, 0); },

    // Evaluate at complex x = [re, im]
    valC(p, x) {
        let acc = C.of(0);
        for (const c of p) acc = C.add(C.mul(acc, x), C.of(c));
        return acc;
    },

    // Multiply two polynomials
    mul(a, b) {
        const r = new Array(a.length + b.length - 1).fill(0);
        for (let i = 0; i < a.length; i++)
            for (let j = 0; j < b.length; j++)
                r[i + j] += a[i] * b[j];
        return r;
    },

    // Add two polynomials (pad shorter to same length)
    add(a, b) {
        const n = Math.max(a.length, b.length);
        const pa = [...new Array(n - a.length).fill(0), ...a];
        const pb = [...new Array(n - b.length).fill(0), ...b];
        return pa.map((v, i) => v + pb[i]);
    },

    scale: (p, s) => p.map(v => v * s),

    // Roots via Durand-Kerner (Weierstrass) method
    roots(coeffs, maxIter = 400) {
        const n = coeffs.length - 1;
        if (n === 0) return [];
        const a = coeffs[0];
        const p = coeffs.map(v => v / a);            // monic
        // Estimate radius
        const r = Math.max(...p.slice(1).map(Math.abs)) ** (1 / n) + 1e-6;
        // Initial approximations on a circle
        let z = Array.from({ length: n }, (_, k) => {
            const th = 2 * Math.PI * k / n + 0.2;
            return C.of(r * Math.cos(th), r * Math.sin(th));
        });
        for (let iter = 0; iter < maxIter; iter++) {
            const zNew = z.map((zi, i) => {
                const pv = Poly.valC(p, zi);
                let prod = C.of(1);
                for (let j = 0; j < n; j++)
                    if (j !== i) prod = C.mul(prod, C.sub(zi, z[j]));
                const denom_abs = C.abs(prod);
                if (denom_abs < 1e-30) return zi;
                return C.sub(zi, C.div(pv, prod));
            });
            const maxΔ = Math.max(...zNew.map((zn, i) => C.abs(C.sub(zn, z[i]))));
            z = zNew;
            if (maxΔ < 1e-12) break;
        }
        return z;   // array of [re, im]
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// Transfer-function → State-Space (Controllable Canonical Form)
// ─────────────────────────────────────────────────────────────────────────────
function tf2ss(num, den) {
    const n = den.length - 1;
    const d0 = den[0];
    const denN = den.map(v => v / d0);
    const numN = num.map(v => v / d0);
    // Pad numerator to length n+1
    const numPad = [...new Array(n + 1 - numN.length).fill(0), ...numN];
    const D = numPad[0];
    // Strictly-proper numerator coefficients b_i
    const b = numPad.slice(1).map((v, i) => v - D * denN[i + 1]);

    // A — companion form
    const A = Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => {
            if (i < n - 1) return j === i + 1 ? 1 : 0;
            return -denN[n - j]; // bottom row: [-d_n, ..., -d_1]
        })
    );
    const B = new Array(n).fill(0); B[n - 1] = 1;
    const Cv = [...b].reverse(); // C row = [b_0, b_1, ..., b_{n-1}]
    return { A, B, C: Cv, D, n };
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix helpers (for RK4)
// ─────────────────────────────────────────────────────────────────────────────
function mvMul(A, x) { return A.map(row => row.reduce((s, a, j) => s + a * x[j], 0)); }
function vAdd(a, b) { return a.map((v, i) => v + b[i]); }
function vScale(a, s) { return a.map(v => v * s); }

// ─────────────────────────────────────────────────────────────────────────────
// Step Response — RK4 ODE integration
// ─────────────────────────────────────────────────────────────────────────────
function stepResponse(num, den, tEnd, nPts = 500) {
    if (num.length === 0 || den.length === 0) return { t: [], y: [] };
    const { A, B, C: Cv, D, n } = tf2ss(num, den);
    const dt = tEnd / (nPts - 1);
    const t = [], y = [];
    let x = new Array(n).fill(0);
    const u = 1.0;
    const f = x_ => vAdd(mvMul(A, x_), vScale(B, u));

    for (let k = 0; k < nPts; k++) {
        t.push(k * dt);
        y.push(Cv.reduce((s, ci, i) => s + ci * x[i], 0) + D * u);
        if (k < nPts - 1) {
            const k1 = f(x);
            const k2 = f(vAdd(x, vScale(k1, 0.5 * dt)));
            const k3 = f(vAdd(x, vScale(k2, 0.5 * dt)));
            const k4 = f(vAdd(x, vScale(k3, dt)));
            x = vAdd(x, vScale(vAdd(k1, vAdd(vScale(k2, 2), vAdd(vScale(k3, 2), k4))), dt / 6));
        }
    }
    return { t, y };
}

// ─────────────────────────────────────────────────────────────────────────────
// Bode Data
// ─────────────────────────────────────────────────────────────────────────────
function bodeData(num, den, nPts = 400) {
    const freqs = Array.from({ length: nPts }, (_, i) =>
        10 ** (-2 + i * 6 / (nPts - 1))
    );
    const magdB = [], phaseRaw = [];
    for (const w of freqs) {
        const jw = C.of(0, w);
        const nv = Poly.valC(num, jw);
        const dv = Poly.valC(den, jw);
        if (C.abs(dv) < 1e-30) { magdB.push(-200); phaseRaw.push(0); continue; }
        const H = C.div(nv, dv);
        magdB.push(20 * Math.log10(C.abs(H) + 1e-15));
        phaseRaw.push(C.arg(H) * 180 / Math.PI);
    }
    // Unwrap phase
    const phase = [...phaseRaw];
    for (let i = 1; i < phase.length; i++) {
        let d = phase[i] - phase[i - 1];
        if (d > 180) d -= 360;
        if (d < -180) d += 360;
        phase[i] = phase[i - 1] + d;
    }
    return { freqs, magdB, phase };
}

// ─────────────────────────────────────────────────────────────────────────────
// Stability Margins
// ─────────────────────────────────────────────────────────────────────────────
function stabilityMargins(olNum, olDen) {
    const { freqs, magdB, phase } = bodeData(olNum, olDen, 800);
    let wgc = null, pm = null, wpc = null, gm = null;

    for (let i = 0; i < magdB.length - 1; i++) {
        if (magdB[i] >= 0 && magdB[i + 1] < 0) {
            const t = (0 - magdB[i]) / (magdB[i + 1] - magdB[i]);
            wgc = freqs[i] + t * (freqs[i + 1] - freqs[i]);
            pm = 180 + (phase[i] + t * (phase[i + 1] - phase[i]));
            break;
        }
    }
    for (let i = 0; i < phase.length - 1; i++) {
        if (phase[i] >= -180 && phase[i + 1] < -180) {
            const t = (-180 - phase[i]) / (phase[i + 1] - phase[i]);
            wpc = freqs[i] + t * (freqs[i + 1] - freqs[i]);
            gm = -(magdB[i] + t * (magdB[i + 1] - magdB[i]));
            break;
        }
    }
    if (pm === null) pm = Infinity;
    if (gm === null) gm = Infinity;
    return { wgc, wpc, pm_deg: pm, gm_db: gm };
}

// ─────────────────────────────────────────────────────────────────────────────
// Root Locus
// ─────────────────────────────────────────────────────────────────────────────
function rootLocus(olNum, olDen, nGains = 200) {
    const gains = Array.from({ length: nGains }, (_, i) => i * 20 / (nGains - 1));
    const olPoles = Poly.roots(olDen);
    const olZeros = olNum.length < olDen.length ? Poly.roots(olNum) : [];

    // Group locus paths by branch (sort roots by continuity)
    const nBranches = olDen.length - 1;
    const paths = Array.from({ length: nBranches }, () => ({ re: [], im: [] }));

    let prevRoots = null;
    for (const K of gains) {
        const clDen = Poly.add(olDen, Poly.scale(olNum, K));
        let roots = Poly.roots(clDen);

        // Sort by proximity to previous roots (greedy nearest-neighbour)
        if (prevRoots) {
            const used = new Array(roots.length).fill(false);
            const sorted = new Array(roots.length);
            for (let i = 0; i < prevRoots.length; i++) {
                let best = -1, bestDist = Infinity;
                for (let j = 0; j < roots.length; j++) {
                    if (!used[j]) {
                        const d = C.abs(C.sub(roots[j], prevRoots[i]));
                        if (d < bestDist) { bestDist = d; best = j; }
                    }
                }
                if (best >= 0) { sorted[i] = roots[best]; used[best] = true; }
            }
            roots = sorted;
        }
        prevRoots = roots;
        roots.forEach((r, i) => {
            if (i < nBranches) {
                paths[i].re.push(r ? r[0] : 0);
                paths[i].im.push(r ? r[1] : 0);
            }
        });
    }

    return { paths, olPoles, olZeros };
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance metrics
// ─────────────────────────────────────────────────────────────────────────────
function performanceMetrics(t, y, ref = 1.0) {
    const ss = y[y.length - 1];
    const peak = Math.max(...y);
    const os = ref > 0 ? Math.max(0, (peak - ref) / ref * 100) : 0;

    // Rise time: 10% → 90%
    const t10 = t.find((_, i) => y[i] >= 0.1 * ref);
    const t90 = t.find((_, i) => y[i] >= 0.9 * ref);
    const riseTime = (t10 !== undefined && t90 !== undefined) ? t90 - t10 : null;

    // Settling time (2% band)
    let settleTime = null;
    for (let i = t.length - 1; i >= 0; i--) {
        if (Math.abs(y[i] - ref) > 0.02 * Math.abs(ref)) {
            settleTime = t[i];
            break;
        }
    }

    return { ss, peak, os, riseTime, settleTime, ssError: Math.abs(ref - ss) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Plant Models  (mirrors modules/plants.py)
// ─────────────────────────────────────────────────────────────────────────────
const PLANTS = {
    "First Order": {
        label: "G(s) = K / (τs + 1)",
        params: {
            K:   { label: "DC Gain (K)",          default: 1.0, min: 0.1, max: 10, step: 0.1 },
            tau: { label: "Time constant τ (s)",  default: 1.0, min: 0.1, max: 20, step: 0.1 },
        },
        tf({ K, tau }) { return { num: [K], den: [tau, 1] }; },
    },
    "Second Order": {
        label: "G(s) = K·ωₙ² / (s² + 2ζωₙs + ωₙ²)",
        params: {
            K:    { label: "DC Gain (K)",      default: 1.0, min: 0.1, max: 10, step: 0.1 },
            wn:   { label: "Natural freq ωₙ",  default: 1.0, min: 0.1, max: 10, step: 0.1 },
            zeta: { label: "Damping ratio ζ",  default: 0.5, min: 0.01, max: 2,  step: 0.01 },
        },
        tf({ K, wn, zeta }) {
            return { num: [K * wn * wn], den: [1, 2 * zeta * wn, wn * wn] };
        },
    },
    "Integrating": {
        label: "G(s) = K / s",
        params: {
            K: { label: "Gain K", default: 1.0, min: 0.1, max: 10, step: 0.1 },
        },
        tf({ K }) { return { num: [K], den: [1, 0] }; },
    },
    "Unstable First Order": {
        label: "G(s) = K / (τs − 1)",
        params: {
            K:   { label: "Gain K",  default: 1.0, min: 0.1, max: 10, step: 0.1 },
            tau: { label: "τ (s)",   default: 1.0, min: 0.1, max: 20, step: 0.1 },
        },
        tf({ K, tau }) { return { num: [K], den: [tau, -1] }; },
    },
    "Third Order": {
        label: "G(s) = K / ((τ₁s+1)(τ₂s+1)(τ₃s+1))",
        params: {
            K:    { label: "Gain K",  default: 1.0, min: 0.1, max: 10, step: 0.1 },
            tau1: { label: "τ₁ (s)", default: 1.0, min: 0.1, max: 10, step: 0.1 },
            tau2: { label: "τ₂ (s)", default: 0.5, min: 0.1, max: 10, step: 0.1 },
            tau3: { label: "τ₃ (s)", default: 0.2, min: 0.1, max: 10, step: 0.1 },
        },
        tf({ K, tau1, tau2, tau3 }) {
            const d = Poly.mul(Poly.mul([tau1, 1], [tau2, 1]), [tau3, 1]);
            return { num: [K], den: d };
        },
    },
    "Non-Minimum Phase": {
        label: "G(s) = K(−αs+1) / (τs+1)²",
        params: {
            K:   { label: "Gain K",       default: 1.0, min: 0.1, max: 10, step: 0.1 },
            tau: { label: "τ (s)",        default: 1.0, min: 0.1, max: 10, step: 0.1 },
            a:   { label: "RHP zero 1/α", default: 0.5, min: 0.01, max: 5, step: 0.05 },
        },
        tf({ K, tau, a }) {
            return { num: Poly.scale([-a, 1], K), den: Poly.mul([tau, 1], [tau, 1]) };
        },
    },
    "Dead-time (Padé 1st)": {
        label: "G(s) ≈ K(−L/2·s+1) / ((L/2·s+1)(τs+1))",
        params: {
            K:   { label: "Gain K",    default: 1.0, min: 0.1, max: 10, step: 0.1 },
            tau: { label: "τ (s)",     default: 3.0, min: 0.1, max: 20, step: 0.1 },
            L:   { label: "Dead-time L (s)", default: 1.0, min: 0, max: 10, step: 0.1 },
        },
        tf({ K, tau, L }) {
            if (L < 1e-6) return { num: [K], den: [tau, 1] };
            const pNum = [-L / 2, 1];
            const pDen = [L / 2, 1];
            const num = Poly.scale(pNum, K);
            const den = Poly.mul(pDen, [tau, 1]);
            return { num, den };
        },
    },
    "Flexible / Resonant": {
        label: "G(s) = K / ((τs+1)(s²/ωₙ²+2ζ/ωₙ·s+1))",
        params: {
            K:    { label: "Gain K",          default: 1.0, min: 0.1, max: 10, step: 0.1 },
            tau:  { label: "Lag τ (s)",       default: 0.5, min: 0.1, max: 5, step: 0.1 },
            wn:   { label: "Resonant ωₙ",     default: 5.0, min: 0.5, max: 50, step: 0.5 },
            zeta: { label: "Damping ζ",       default: 0.05, min: 0.001, max: 0.5, step: 0.005 },
        },
        tf({ K, tau, wn, zeta }) {
            const resDen = [1 / (wn * wn), 2 * zeta / wn, 1];
            const den = Poly.mul([tau, 1], resDen);
            return { num: [K], den };
        },
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// PID Controller TF  (with derivative filter N)
// PID(s) = Kp + Ki/s + Kd·N·s/(s+N)
//        = [(Kp+Kd·N)s² + (Kp·N+Ki)s + Ki·N] / [s² + N·s]
// ─────────────────────────────────────────────────────────────────────────────
function pidTF(Kp, Ki, Kd, N) {
    const num = [Kp + Kd * N, Kp * N + Ki, Ki * N];
    const den = [1, N, 0];
    return { num, den };
}

// ─────────────────────────────────────────────────────────────────────────────
// Build Open-Loop and Closed-Loop TFs
// ─────────────────────────────────────────────────────────────────────────────
function buildOLCL(pNum, pDen, cNum, cDen) {
    const olNum = Poly.mul(pNum, cNum);
    const olDen = Poly.mul(pDen, cDen);
    const clNum = olNum;
    const clDen = Poly.add(olDen, olNum);
    return { olNum, olDen, clNum, clDen };
}

// ─────────────────────────────────────────────────────────────────────────────
// Nyquist data  (positive frequencies only; caller mirrors for ω < 0)
// ─────────────────────────────────────────────────────────────────────────────
function nyquistData(olNum, olDen, nPts = 500) {
    const freqs = Array.from({ length: nPts }, (_, i) =>
        10 ** (-3 + i * 7 / (nPts - 1))
    );
    const re = [], im = [];
    for (const w of freqs) {
        const H = C.div(Poly.valC(olNum, C.of(0, w)), Poly.valC(olDen, C.of(0, w)));
        re.push(H[0]);
        im.push(H[1]);
    }
    return { freqs, re, im };
}

// Export everything on the window object (for use from inline HTML scripts)
window.CS = {
    Poly, C, tf2ss, stepResponse, bodeData, stabilityMargins,
    rootLocus, performanceMetrics, PLANTS, pidTF, buildOLCL, nyquistData,
};
