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

// ─────────────────────────────────────────────────────────────────────────────
// Matrix library — 2D arrays (array of rows)
// ─────────────────────────────────────────────────────────────────────────────
const Mat = {
    zeros: (n, m = n) => Array.from({ length: n }, () => new Array(m).fill(0)),
    eye:   (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0)),
    copy:  (A) => A.map(r => [...r]),
    add:   (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j])),
    sub:   (A, B) => A.map((r, i) => r.map((v, j) => v - B[i][j])),
    scale: (A, s) => A.map(r => r.map(v => v * s)),
    trans: (A) => A[0].map((_, j) => A.map(r => r[j])),
    mul(A, B) {
        const n = A.length, m = B[0].length, p = B.length;
        return Array.from({ length: n }, (_, i) =>
            Array.from({ length: m }, (_, j) =>
                A[i].reduce((s, _, k) => s + A[i][k] * B[k][j], 0)));
    },
    mvm: (A, v) => A.map(row => row.reduce((s, a, j) => s + a * v[j], 0)),
    norm: (A) => Math.sqrt(A.reduce((s, r) => s + r.reduce((rs, v) => rs + v * v, 0), 0)),
    // Gauss-Jordan inverse
    inv(A) {
        const n = A.length;
        const M = A.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => i === j ? 1 : 0)]);
        for (let col = 0; col < n; col++) {
            let pivot = col;
            for (let row = col + 1; row < n; row++)
                if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
            [M[col], M[pivot]] = [M[pivot], M[col]];
            const d = M[col][col];
            if (Math.abs(d) < 1e-14) return null;
            for (let j = 0; j < 2 * n; j++) M[col][j] /= d;
            for (let row = 0; row < n; row++) {
                if (row === col) continue;
                const f = M[row][col];
                for (let j = 0; j < 2 * n; j++) M[row][j] -= f * M[col][j];
            }
        }
        return M.map(r => r.slice(n));
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// Discretise state-space: A_d = I + A*dt + (A*dt)²/2,  B_d = (A_d-I)*A⁻¹*B ≈ B*dt + A*B*dt²/2
// B_col: n×1 2D array
// ─────────────────────────────────────────────────────────────────────────────
function discretizeSS(A, B_col, dt) {
    const n = A.length;
    const Adt  = Mat.scale(A, dt);
    const Adt2 = Mat.scale(Mat.mul(Adt, Adt), 0.5);
    const Ad   = Mat.add(Mat.add(Mat.eye(n), Adt), Adt2);
    // Bd ≈ B*dt + A*B*dt²/2
    const ABdt2 = Mat.scale(Mat.mul(A, B_col), 0.5 * dt * dt);
    const Bd   = Mat.add(Mat.scale(B_col, dt), ABdt2);
    return { Ad, Bd };
}

// ─────────────────────────────────────────────────────────────────────────────
// DARE solver — value iteration  (single input: R is scalar)
// P_{k+1} = Qd + Ad'·P·Ad − (Ad'·P·Bd)·(R + Bd'·P·Bd)⁻¹·(Bd'·P·Ad)
// ─────────────────────────────────────────────────────────────────────────────
function solveDARE(Ad, Bd_col, Qd, R) {
    const AdT = Mat.trans(Ad);
    let P = Mat.copy(Qd);
    for (let k = 0; k < 4000; k++) {
        const PBd  = Mat.mul(P, Bd_col);          // n×1
        const BtPB = Bd_col.reduce((s, row, i) => s + row[0] * PBd[i][0], 0); // scalar B'PB
        const denom = R + BtPB;
        const AtPBd = Mat.mul(AdT, PBd);           // n×1
        // correction = (Ad'PBd)(Bd'PAd)/denom = outer(AtPBd,AtPBd)/denom (n×n)
        const corr  = Mat.scale(Mat.mul(AtPBd, Mat.trans(AtPBd)), 1 / denom);
        const AtPAd = Mat.mul(AdT, Mat.mul(P, Ad)); // n×n
        const Pnew  = Mat.sub(Mat.add(Qd, AtPAd), corr);
        const diff  = Mat.norm(Mat.sub(Pnew, P));
        P = Pnew;
        if (diff < 1e-9) break;
    }
    return P;
}

// ─────────────────────────────────────────────────────────────────────────────
// LQR design: returns gain K (1×n flat array) and dc-feedforward Nbar (scalar)
// ─────────────────────────────────────────────────────────────────────────────
function lqrDesign(A, B_col, C_row, Q, R, dt = 0.01) {
    const { Ad, Bd } = discretizeSS(A, B_col, dt);
    const n = A.length;
    // Q_d = Q*dt (maps continuous cost to discrete)
    const Qd = Mat.scale(Q, dt);
    const P  = solveDARE(Ad, Bd, Qd, R * dt);

    // K = (R + Bd'PBd)^{-1} * Bd'*P*Ad  (1×n flat array)
    const PBd  = Mat.mul(P, Bd);
    const BtPB = Bd.reduce((s, row, i) => s + row[0] * PBd[i][0], 0);
    const denom = R * dt + BtPB;
    const BtPA = Mat.mul(Mat.trans(Bd), Mat.mul(P, Ad)); // 1×n
    const K = BtPA[0].map(v => v / denom);               // flat 1D array

    // Closed-loop A for Nbar: Acl = A - B*K
    const BK = Mat.mul(B_col, [K]); // n×n
    const Acl = Mat.sub(A, BK);
    const AclInv = Mat.inv(Acl);
    if (!AclInv) return { K, Nbar: 1 };
    // Nbar = -1 / (C*(Acl^{-1})*B)
    const CAclInvB = Mat.mvm(Mat.mul([C_row], AclInv), B_col.map(r => r[0]));
    const Nbar = -1 / CAclInvB[0];
    return { K, Nbar };
}

// ─────────────────────────────────────────────────────────────────────────────
// Kalman design: returns observer gain L (n×1 flat array)
// Dual of LQR: swap A→A', B→C', Q→Qn, R→Rn
// ─────────────────────────────────────────────────────────────────────────────
function kalmanDesign(A, B_col, C_row, Qn, Rn, dt = 0.01) {
    const At = Mat.trans(A);
    const Ct = C_row.map(v => [[v]]); // just for shape — C_row is 1D
    // Build Ct as n×1 2D column
    const CtCol = C_row.map(v => [v]);
    const { Ad: AdT, Bd: CtdCol } = discretizeSS(At, CtCol, dt);
    const Qnd = Mat.scale(Mat.eye(A.length), Qn * dt);
    const Pe = solveDARE(AdT, CtdCol, Qnd, Rn * dt);
    // L = Pe*C' / Rn  (n×1 flat)
    const PeCt = Mat.mul(Pe, CtCol);   // n×1
    const L = PeCt.map(row => row[0] / Rn);
    return L;
}

// ─────────────────────────────────────────────────────────────────────────────
// LQR closed-loop simulation — RK4
// Returns { t, y, x_hist, u_hist }
// ─────────────────────────────────────────────────────────────────────────────
function simulateLQR(A, B_col, C_row, K, Nbar, ref, tEnd, nPts = 600) {
    const n = A.length;
    const dt = tEnd / (nPts - 1);
    const t = [], y = [], u_hist = [], x_hist = [];
    let x = new Array(n).fill(0);

    const f = (xv, uv) => {
        const Ax = Mat.mvm(A, xv);
        const Bu = B_col.map((row, i) => Ax[i] + row[0] * uv);
        return Bu;
    };

    for (let k = 0; k < nPts; k++) {
        t.push(k * dt);
        const u = -K.reduce((s, ki, i) => s + ki * x[i], 0) + Nbar * ref;
        y.push(C_row.reduce((s, ci, i) => s + ci * x[i], 0));
        u_hist.push(u);
        x_hist.push([...x]);
        if (k < nPts - 1) {
            const k1 = f(x, u);
            const k2 = f(vAdd(x, vScale(k1, 0.5 * dt)), u);
            const k3 = f(vAdd(x, vScale(k2, 0.5 * dt)), u);
            const k4 = f(vAdd(x, vScale(k3, dt)), u);
            x = vAdd(x, vScale(vAdd(k1, vAdd(vScale(k2, 2), vAdd(vScale(k3, 2), k4))), dt / 6));
        }
    }
    return { t, y, u_hist, x_hist };
}

// ─────────────────────────────────────────────────────────────────────────────
// LQG simulation — RK4 with Kalman estimator
// ─────────────────────────────────────────────────────────────────────────────
function simulateLQG(A, B_col, C_row, K, Nbar, L, ref, tEnd, qStd, rStd, seed, nPts = 600) {
    const n = A.length;
    const dt = tEnd / (nPts - 1);

    // Simple seeded random (Mulberry32)
    let s = (seed || 42) >>> 0;
    const rng = () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    const randn = () => { const u1 = rng(), u2 = rng(); return Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2); };

    const t = [], y_true = [], y_meas = [], y_est = [], u_hist = [];
    let x = new Array(n).fill(0);
    let xhat = new Array(n).fill(0);

    const f_plant = (xv, uv) => {
        const Ax = Mat.mvm(A, xv);
        return B_col.map((row, i) => Ax[i] + row[0] * uv);
    };
    const f_obs = (xhv, uv, innov) => {
        const Axh = Mat.mvm(A, xhv);
        const BuL = B_col.map((row, i) => Axh[i] + row[0] * uv + L[i] * innov);
        return BuL;
    };

    for (let k = 0; k < nPts; k++) {
        t.push(k * dt);
        const u   = -K.reduce((s, ki, i) => s + ki * xhat[i], 0) + Nbar * ref;
        const ytrue = C_row.reduce((s, ci, i) => s + ci * x[i], 0);
        const ymeas = ytrue + rStd * randn();
        const yest  = C_row.reduce((s, ci, i) => s + ci * xhat[i], 0);
        const innov = ymeas - yest;

        y_true.push(ytrue);
        y_meas.push(ymeas);
        y_est.push(yest);
        u_hist.push(u);

        if (k < nPts - 1) {
            // Plant (with process noise on states)
            const w = new Array(n).fill(0).map(() => qStd * randn());
            const k1p = f_plant(x, u).map((v, i) => v + w[i]);
            const k2p = f_plant(vAdd(x, vScale(k1p, 0.5 * dt)), u).map((v, i) => v + w[i]);
            const k3p = f_plant(vAdd(x, vScale(k2p, 0.5 * dt)), u).map((v, i) => v + w[i]);
            const k4p = f_plant(vAdd(x, vScale(k3p, dt)), u).map((v, i) => v + w[i]);
            x = vAdd(x, vScale(vAdd(k1p, vAdd(vScale(k2p, 2), vAdd(vScale(k3p, 2), k4p))), dt / 6));

            // Observer
            const k1o = f_obs(xhat, u, innov);
            const k2o = f_obs(vAdd(xhat, vScale(k1o, 0.5 * dt)), u, innov);
            const k3o = f_obs(vAdd(xhat, vScale(k2o, 0.5 * dt)), u, innov);
            const k4o = f_obs(vAdd(xhat, vScale(k3o, dt)), u, innov);
            xhat = vAdd(xhat, vScale(vAdd(k1o, vAdd(vScale(k2o, 2), vAdd(vScale(k3o, 2), k4o))), dt / 6));
        }
    }
    return { t, y_true, y_meas, y_est, u_hist };
}

// ─────────────────────────────────────────────────────────────────────────────
// State-Space Plants for LQR/LQG
// ─────────────────────────────────────────────────────────────────────────────
const SS_PLANTS = {
    "Mass-Spring-Damper": {
        desc: "2nd-order: cart position & velocity",
        states: ["Position x (m)", "Velocity ẋ (m/s)"],
        refState: 0,
        params: {
            m:    { label: "Mass m (kg)",        default: 1.0, min: 0.1, max: 5,   step: 0.1 },
            k:    { label: "Spring k (N/m)",     default: 2.0, min: 0.1, max: 20,  step: 0.1 },
            b:    { label: "Damper b (N·s/m)",   default: 0.5, min: 0,   max: 5,   step: 0.1 },
        },
        defaultQ: [10, 1],
        defaultR: 1,
        ss({ m, k, b }) {
            const A = [[0, 1], [-k / m, -b / m]];
            const B = [[0], [1 / m]];
            const Cv = [1, 0];
            return { A, B, Cv };
        },
    },
    "DC Motor": {
        desc: "2nd-order: armature current & angular velocity",
        states: ["Current i (A)", "Speed ω (rad/s)"],
        refState: 1,
        params: {
            R:  { label: "Resistance R (Ω)",  default: 1.0, min: 0.1, max: 10, step: 0.1 },
            L:  { label: "Inductance L (H)",  default: 0.5, min: 0.01, max: 2, step: 0.01 },
            Km: { label: "Motor const Km",     default: 0.5, min: 0.1, max: 5,  step: 0.1 },
            Kb: { label: "Back-EMF Kb",        default: 0.5, min: 0.1, max: 5,  step: 0.1 },
            J:  { label: "Inertia J (kg·m²)", default: 0.1, min: 0.01, max: 1, step: 0.01 },
            B:  { label: "Friction B",         default: 0.1, min: 0,   max: 2,  step: 0.05 },
        },
        defaultQ: [1, 10],
        defaultR: 1,
        ss({ R, L, Km, Kb, J, B }) {
            const A = [[-R / L, -Kb / L], [Km / J, -B / J]];
            const Bc = [[1 / L], [0]];
            const Cv = [0, 1];
            return { A, B: Bc, Cv };
        },
    },
    "Inverted Pendulum": {
        desc: "4th-order: cart position, velocity, angle, angular rate",
        states: ["Cart x (m)", "Cart ẋ (m/s)", "Angle θ (rad)", "Ang. rate θ̇ (rad/s)"],
        refState: 0,
        params: {
            M:  { label: "Cart mass M (kg)",  default: 1.0, min: 0.5, max: 5,   step: 0.1 },
            mp: { label: "Pole mass m (kg)",  default: 0.2, min: 0.05, max: 1,  step: 0.05 },
            lp: { label: "Pole half-len (m)", default: 0.5, min: 0.1, max: 2,   step: 0.05 },
        },
        defaultQ: [1, 0.1, 100, 10],
        defaultR: 1,
        ss({ M, mp, lp }) {
            const g = 9.81;
            const A = [
                [0, 1, 0, 0],
                [0, 0, -(mp * g) / M, 0],
                [0, 0, 0, 1],
                [0, 0, (M + mp) * g / (M * lp), 0],
            ];
            const Bc = [[0], [1 / M], [0], [-1 / (M * lp)]];
            const Cv = [1, 0, 0, 0];
            return { A, B: Bc, Cv };
        },
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// MPC — Build prediction matrices Phi (Np×n) and Psi (Np×Nc)  [SISO]
// ─────────────────────────────────────────────────────────────────────────────
function buildPredMatrices(Ad, Bd_col, C_row, Np, Nc) {
    const n = Ad.length;
    const AdPow = [Mat.eye(n)];
    for (let k = 1; k <= Np; k++) AdPow.push(Mat.mul(AdPow[k - 1], Ad));

    // cAkB[k] = C * Ad^k * Bd  (scalar, SISO)
    const cAkB = AdPow.map(Ak => {
        const AkBd = Mat.mul(Ak, Bd_col);
        return C_row.reduce((s, ci, i) => s + ci * AkBd[i][0], 0);
    });

    // Phi[i] = 1D row = C * Ad^(i+1),  Np×n
    const Phi = AdPow.slice(1).map(Ak =>
        Array.from({ length: n }, (_, j) => C_row.reduce((s, ck, k) => s + ck * Ak[k][j], 0))
    );

    // Psi[i,j]: hold-last-input convention
    const Psi = Array.from({ length: Np }, (_, i) =>
        Array.from({ length: Nc }, (_, j) => {
            if (i < j) return 0;
            if (j < Nc - 1) return cAkB[i - j];
            let acc = 0;
            for (let l = 0; l <= i - (Nc - 1); l++) acc += cAkB[l];
            return acc;
        })
    );
    return { Phi, Psi };
}

// Precompute offline part: MinvPsiTQy (Nc×Np) for given Qy, Ru
function precomputeMPC(Phi, Psi, Qy, Ru) {
    const Nc = Psi[0].length;
    const PsiT = Psi[0].map((_, j) => Psi.map(r => r[j]));       // Nc×Np
    const M = Mat.mul(PsiT, Psi).map((row, i) =>
        row.map((v, j) => Qy * v + (i === j ? Ru : 0)));          // Nc×Nc
    const Minv = Mat.inv(M);
    if (!Minv) return null;
    return Mat.scale(Mat.mul(Minv, PsiT), Qy);                    // Nc×Np
}

// Simulate MPC (discrete-time receding horizon)
function simulateMPC(A, B_col, C_row, Np, Nc, Qy, Ru, ref, tEnd, dt, uMin, uMax, qNoise, seed) {
    const { Ad, Bd } = discretizeSS(A, B_col, dt);
    const { Phi, Psi } = buildPredMatrices(Ad, Bd, C_row, Np, Nc);
    const MinvPsiTQy = precomputeMPC(Phi, Psi, Qy, Ru);
    if (!MinvPsiTQy) return { t: [], y: [], u: [], constrained: [] };

    const n = A.length;
    const steps = Math.floor(tEnd / dt);
    const skip  = Math.max(1, Math.floor(steps / 800));
    const t_out = [], y_out = [], u_out = [], constrained = [];
    let x = new Array(n).fill(0);

    let s = (seed || 42) >>> 0;
    const rng = () => { s |= 0; s = s + 0x6D2B79F5 | 0; let tt = Math.imul(s ^ s >>> 15, 1 | s); tt = tt + Math.imul(tt ^ tt >>> 7, 61 | tt) ^ tt; return ((tt ^ tt >>> 14) >>> 0) / 4294967296; };
    const randn = () => { const u1 = rng(), u2 = rng(); return Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2); };

    for (let k = 0; k < steps; k++) {
        // err[i] = ref - Phi[i]·x
        const err = Phi.map(row => ref - row.reduce((s, v, j) => s + v * x[j], 0));
        // u_opt = MinvPsiTQy[0] · err
        const uOpt = MinvPsiTQy[0].reduce((s, v, i) => s + v * err[i], 0);
        const uApp = Math.max(uMin, Math.min(uMax, uOpt));

        if (k % skip === 0) {
            t_out.push(+(k * dt).toFixed(5));
            y_out.push(C_row.reduce((s, ci, i) => s + ci * x[i], 0));
            u_out.push(uApp);
            constrained.push(Math.abs(uApp - uOpt) > 1e-6);
        }

        // x_{k+1} = Ad*x + Bd*u + noise
        const Adx = Mat.mvm(Ad, x);
        const noise = qNoise > 0 ? new Array(n).fill(0).map(() => qNoise * randn()) : new Array(n).fill(0);
        x = Bd.map((row, i) => Adx[i] + row[0] * uApp + noise[i]);
    }
    return { t: t_out, y: y_out, u: u_out, constrained };
}

// Export everything on the window object (for use from inline HTML scripts)
window.CS = {
    Poly, C, tf2ss, stepResponse, bodeData, stabilityMargins,
    rootLocus, performanceMetrics, PLANTS, pidTF, buildOLCL, nyquistData,
    Mat, discretizeSS, solveDARE, lqrDesign, kalmanDesign,
    simulateLQR, simulateLQG, SS_PLANTS,
    buildPredMatrices, precomputeMPC, simulateMPC,
};
