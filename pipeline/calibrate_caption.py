"""Allometry (JAX Tesseract) -> Lindauer (Julia Tesseract, from_source) -> log ratio, calibrated.

Unknowns: log V_blood [uL] and the allometric exponent. Targets: the paper's Fig 4 base case,
log10 V(26 wk)/V(0) = -1.29 at 5 mg/kg and -0.39 at 0.5 mg/kg (Q3W). Start: as published (V_blood 1400 uL,
exponent -0.25, fast tumour growth L0 = 0.0088/day), which gives -6.0 / -4.7.
"""

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

MOUSE = jnp.array([1.98e-5, 0.0194])  # SLtg, KdegPD1 in mouse (Table 1 / S2)
TARGETS = jnp.array([-1.29, -0.39])
DOSES = (5.0, 0.5)
X0 = np.array([np.log(1400.0), -0.25])


def make_loss(allometry, lindauer):
    def log_ratios(x):
        human = apply_tesseract(allometry, {"mouse_values": MOUSE, "exponent": x[1]})["human_values"]
        theta = jnp.concatenate([jnp.exp(x[:1]), human])
        out = []
        for dose in DOSES:
            tv = apply_tesseract(
                lindauer,
                {
                    "theta": theta,
                    "fields": ["V_blood", "SLtg", "KdegPD1"],
                    "scenario": "fast_allom",
                    "dose_mgkg": dose,
                    "interval_days": 21.0,
                    "times_days": jnp.array([0.0, 182.0]),
                },
            )["tumour_volume"]
            out.append(jnp.log10(tv[-1] / tv[0]))
        return jnp.stack(out)

    def loss(x):
        return jnp.sum((log_ratios(x) - TARGETS) ** 2)

    return log_ratios, loss


def main():
    log = {}
    allometry = Tesseract.from_tesseract_api("allometry/tesseract_api.py")
    with Tesseract.from_source("lindauer/tesseract_api.py", startup_timeout=900) as lindauer:
        log_ratios, loss = make_loss(allometry, lindauer)
        vg = jax.value_and_grad(loss)

        t0 = time.perf_counter()
        v0, g0 = vg(jnp.asarray(X0))
        log["first value_and_grad s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        vg(jnp.asarray(X0))
        log["warm value_and_grad s"] = time.perf_counter() - t0
        print("start log ratios", np.asarray(log_ratios(jnp.asarray(X0))), "loss", float(v0), "grad", np.asarray(g0))

        # chain gradient against central finite differences
        h = 1e-5
        fd = []
        for k in range(2):
            e = np.zeros(2); e[k] = h
            fd.append((float(loss(jnp.asarray(X0 + e))) - float(loss(jnp.asarray(X0 - e)))) / (2 * h))
        log["grad AD"] = np.asarray(g0).tolist()
        log["grad FD"] = fd
        print("grad AD", np.asarray(g0), "FD", np.asarray(fd))

        calls = {"n": 0}

        def fun(x):
            calls["n"] += 1
            v, g = vg(jnp.asarray(x))
            if not np.isfinite(float(v)):  # failed solve: reject the step
                print(f"  iter {calls['n']:3d}  V_blood {np.exp(x[0]):12.1f} uL  exponent {x[1]:+.4f}  solve failed", flush=True)
                return 1e6, np.zeros(2)
            print(f"  iter {calls['n']:3d}  V_blood {np.exp(x[0]):12.1f} uL  exponent {x[1]:+.4f}  loss {float(v):.3e}", flush=True)
            return float(v), np.asarray(g, dtype=float)

        t0 = time.perf_counter()
        res = minimize(fun, X0, jac=True, method="L-BFGS-B", bounds=[(np.log(100.0), np.log(1e8)), (-0.6, 0.0)],
                       options={"maxiter": 60, "gtol": 1e-10})
        log["calibration s"] = time.perf_counter() - t0
        log["value_and_grad evaluations"] = calls["n"]
        log["result"] = {"V_blood_uL": float(np.exp(res.x[0])), "exponent": float(res.x[1]), "loss": float(res.fun),
                         "log_ratios": np.asarray(log_ratios(jnp.asarray(res.x))).tolist(), "message": str(res.message)}
    print(json.dumps(log, indent=1))
    with open(sys.argv[1] if len(sys.argv) > 1 else "calibrate_caption.json", "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
