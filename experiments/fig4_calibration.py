"""Calibrate the Lindauer human base case to the paper's Figure 4 through the Julia Tesseract.

    python experiments/fig4_calibration.py <out.json>

Figure 4 of Lindauer et al. 2017 is a one-at-a-time sensitivity analysis: log10 V(26 wk)/V(0) at 5 and
0.5 mg/kg Q3W with each of 16 parameters multiplied by 5 and by 0.2 (64 bars), plus the two base values in the
caption (-1.29 / -0.39). The paper does not state the base case's blood volume or tumour growth scenario.
QSPAtlas.jl chose them by a discrete search (V_blood 5 L, medium growth: L0 0.0036/day, SLtg 6.247e-7) and then
matched 58 of the 64 bars. Here the four unstated values (V_blood, L0, SLtg, KdegPD1) are fitted by gradient, through
the served Tesseract and tesseract-jax, to the 60 of the 66 numbers that are not QSPAtlas's known misfits. The fit
starts from the published values with a 5 L blood volume (as published, some solves fail), so V_blood is in effect
set by the start and only adjusted; L0, SLtg and KdegPD1 are found.

The digitised bars are read at run time from QSPAtlas.jl (they derive from the paper's figure, CC BY-NC-ND 4.0;
they are not copied into this repository).
"""

import json
import sys
import time
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

QSPATLAS = "https://raw.githubusercontent.com/singhharsh1708/QSPAtlas.jl/46d8f75be7322ed40af095b7cbe5e64d35f25811"
REF = "models/lindauer_2017/reference"
NAMES = {"L": "Lfrac"}                         # Figure 4 label -> raw parameter name
UNKNOWNS = ["V_blood", "L0", "SLtg", "KdegPD1"]
AS_PUBLISHED = np.log([1400.0, 0.0088, 2.575e-6, 0.00246])   # mouse V_blood, Table S3 row 1, KdegPD1 as printed
START = np.log([5.0e6, 0.0088, 2.575e-6, 0.00246])           # as published, with a human blood volume (5 L)
QSPATLAS_FIX = np.log([5.0e6, 0.0036, 6.247e-7, 0.00246])    # 5 L, medium growth (QSPAtlas Fix list)
BOUNDS = [(np.log(1e2), np.log(1e8)), (np.log(1e-4), np.log(0.1)), (np.log(1e-8), np.log(1e-4)),
          (np.log(1e-4), np.log(0.1))]


def fetch(name):
    url = f"{QSPATLAS}/{REF}/{name}.csv"
    text = urllib.request.urlopen(url, timeout=60).read().decode()
    order = next((line for line in text.splitlines() if line.startswith("# order:")), None)
    order = [n.strip() for n in order.split(":", 1)[1].split(",")] if order else None
    ys = [float(line.split(",")[1]) for line in text.splitlines() if line and line[0].isdigit()]
    return order, ys


def targets():
    out = []  # (label, dose, parameter or None, multiplier, value)
    for dose_id, dose in (("5mgkg", 5.0), ("0p5mgkg", 0.5)):
        for mult_id, mult in (("x5", 5.0), ("x0p2", 0.2)):
            order, ys = fetch(f"fig4_{dose_id}_{mult_id}")
            for name, y in zip(order, ys, strict=True):
                out.append((f"{name} x{mult:g} @ {dose:g} mg/kg", dose, NAMES.get(name, name), mult, y))
    _, base = fetch("fig4_base_case")
    out += [("base @ 5 mg/kg", 5.0, None, 1.0, base[0]), ("base @ 0.5 mg/kg", 0.5, None, 1.0, base[1])]
    return out


def main(outfile):
    tg = targets()
    y_obs = jnp.array([t[4] for t in tg])
    # QSPAtlas's six known misfits (report.md): gamma x5 at both doses and the four L bars
    outlier = np.array([t[2] == "Lfrac" or (t[2] == "gamma" and t[3] == 5.0) for t in tg])
    log = {"targets": len(tg), "outliers": int(outlier.sum())}
    with Tesseract.from_source("lindauer/tesseract_api.py", startup_timeout=900) as lindauer:

        def predict(x, which=range(len(tg))):
            values = jnp.exp(x)
            preds = []
            for _, dose, param, mult, _ in (tg[i] for i in which):
                if param is None:                      # caption base value
                    fields, theta = UNKNOWNS, values
                elif param in UNKNOWNS:                # a fitted parameter, perturbed
                    fields, theta = UNKNOWNS, values.at[UNKNOWNS.index(param)].multiply(mult)
                else:                                  # a fixed parameter, perturbed from its base value
                    fields = UNKNOWNS + [param]
                    theta = jnp.concatenate([values, jnp.array([BASE_VALUES[param] * mult])])
                tv = apply_tesseract(lindauer, {
                    "theta": theta, "fields": fields, "scenario": "fast_allom", "dose_mgkg": dose,
                    "interval_days": 21.0, "times_days": jnp.array([0.0, 182.0]),
                })["tumour_volume"]
                preds.append(jnp.log10(tv[-1] / tv[0]))
            return jnp.stack(preds)

        # Fitted to the 60 bars without QSPAtlas's six known misfits: those residuals are structural and large, would
        # dominate a sum of squares over all 66, and two of them fail to solve at some points.
        good = np.flatnonzero(~outlier)

        def loss(x):
            return jnp.sum((predict(x, good) - y_obs[good]) ** 2)

        vg = jax.value_and_grad(loss)

        def describe(x):
            # The same measures at every point, over the bars whose solve succeeded: sse_fit is the fitted objective
            # (the 60 good bars; NaN as an objective if one fails, then summed over the rest); rms over the good bars,
            # over every finite bar, and over the six misfits; and the bars whose solve failed.
            r = np.asarray(predict(jnp.asarray(x))) - np.asarray(y_obs)
            ok = ~np.isnan(r)

            def rms(mask):
                return float(np.sqrt(np.mean(r[mask] ** 2))) if mask.any() else float("nan")

            return {"sse_fit": float(np.sum(r[ok & ~outlier] ** 2)), "rms_good": rms(ok & ~outlier),
                    "rms_finite": rms(ok), "rms_misfits": rms(ok & outlier),
                    "nan_targets": [tg[i][0] for i in np.flatnonzero(~ok)]}

        # As published (a mouse blood volume in man) some perturbed solves fail and return NaN, so the fit starts
        # from the published values with a human blood volume (5 L, which the fit then only adjusts); the failing
        # bars are recorded.
        for label, x in (("as_published", AS_PUBLISHED), ("start", START), ("qspatlas_fix", QSPATLAS_FIX)):
            log[label] = describe(x)
            print(label, log[label], flush=True)

        calls = {"n": 0}

        def fun(x):
            calls["n"] += 1
            v, g = vg(jnp.asarray(x))
            v = float(v)
            print(f"  eval {calls['n']:3d} loss {v:.5g}  " + "  ".join(
                f"{n} {np.exp(xi):.4g}" for n, xi in zip(UNKNOWNS, x, strict=True)), flush=True)
            if not np.isfinite(v):
                return 1e6, np.zeros(len(x))
            return v, np.asarray(g, dtype=float)

        t0 = time.perf_counter()
        res = minimize(fun, START, jac=True, method="L-BFGS-B", bounds=BOUNDS, options={"maxiter": 60})
        log["calibration_s"] = time.perf_counter() - t0
        log["value_and_grad_evals"] = calls["n"]
        log["fit"] = {"values": dict(zip(UNKNOWNS, np.exp(res.x).tolist(), strict=True)), **describe(res.x),
                      "message": str(res.message)}
    print(json.dumps(log, indent=1))
    with open(outfile, "w") as f:
        json.dump(log, f, indent=1)


# Base values of the parameters that are varied but not fitted: human_raw in lindauer_model.jl (Table 1 human
# values and the species-invariant Table 1 / S2 values); the same in every Table S3 scenario, which sets only L0
# and SLtg (both fitted here). The 5 mg/kg panels vary V2, the 0.5 mg/kg panels EC50TP instead.
BASE_VALUES = {"gamma": 2.28, "FR": 0.715, "Tmulti": 4.32, "N_PD1_TC": 10000.0, "N_Tcell": 1000.0, "EMAXTP": 94.7,
               "EC50TP": 1.46, "V1": 2877.0, "CL": 167.0, "PLQ": 12.7, "Lfrac": 0.002, "CLup": 0.0366, "V2": 2854.0}

if __name__ == "__main__":
    main(sys.argv[1])
