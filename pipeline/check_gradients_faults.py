"""Tesseract-jax check_gradients (PR #300) on the chain, clean and with one injected fault at a time.

Usage: python pipeline/check_gradients_faults.py <clean|lindauer_vjp|allometry_vjp|glue> [--rule <rms|elementwise|floor>]
The fault reaches the served Julia Tesseract through STIFF_FAULT in the environment it inherits.
"""

import os
import sys
import time

FAULT = sys.argv[1]
RULE = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--rule" else None
os.environ["STIFF_FAULT"] = FAULT

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from tesseract_core import Tesseract  # noqa: E402
from tesseract_jax import apply_tesseract, check_gradients  # noqa: E402

jax.config.update("jax_enable_x64", True)

MOUSE = jnp.array([1.98e-5, 0.0194])
TARGETS = jnp.array([-1.29, -0.39])
# a point near the calibrated one, where the loss is not at its minimum
PARAMS = {"log_V_blood": jnp.asarray(jnp.log(3.0e6)), "exponent": jnp.asarray(-0.15)}


def main():
    allometry = Tesseract.from_tesseract_api("allometry/tesseract_api.py")
    with Tesseract.from_source("lindauer/tesseract_api.py", startup_timeout=900) as lindauer:

        def loss(p):
            human = apply_tesseract(allometry, {"mouse_values": MOUSE, "exponent": p["exponent"]})["human_values"]
            if FAULT == "glue":
                # injected glue fault: SLtg detached from the graph (value unchanged, gradient path lost)
                human = human.at[0].set(jax.lax.stop_gradient(human[0]))
            theta = jnp.concatenate([jnp.exp(p["log_V_blood"])[None], human])
            ratios = []
            for dose in (5.0, 0.5):
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
                ratios.append(jnp.log10(tv[-1] / tv[0]))
            return jnp.sum((jnp.stack(ratios) - TARGETS) ** 2)

        if RULE:  # step rule for the finite differences (experiments/gradcheck_grid.py)
            sys.path.insert(0, "experiments")
            from gradcheck_grid import patch

            patch(RULE)
        loss(PARAMS)  # warm the served process (first-call compile) outside the timing
        t0 = time.perf_counter()
        report = check_gradients(loss, PARAMS)
        dt = time.perf_counter() - t0
    print(f"=== fault: {FAULT}   check_gradients took {dt:.1f} s   passed: {report.passed}")
    print(report)


if __name__ == "__main__":
    main()
