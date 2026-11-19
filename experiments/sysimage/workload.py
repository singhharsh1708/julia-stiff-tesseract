"""Experiment 2 workload and timer: the Tesseract in-process through juliacall.

    python experiments/sysimage/workload.py <out.json>

Timings are for this process: load (juliacall + packages), first and warm apply, first and warm VJP. Run it with
PYTHON_JULIACALL_TRACE_COMPILE=<file> to record what gets compiled, or PYTHON_JULIACALL_SYSIMAGE=<so> to time a
custom system image.
"""

import json
import os
import sys
import time

import numpy as np

t0 = time.perf_counter()
from tesseract_core import Tesseract  # noqa: E402

tess = Tesseract.from_tesseract_api("lindauer/tesseract_api.py")
t_load = time.perf_counter() - t0

inputs = {
    "theta": np.array([5.0e6, 0.0036, 6.247e-7, 0.00246]),
    "fields": ["V_blood", "L0", "SLtg", "KdegPD1"],
    "scenario": "med_growth",
    "dose_mgkg": 5.0,
    "interval_days": 21.0,
    "times_days": np.array([0.0, 91.0, 182.0]),
}
ct = {"tumour_volume": np.array([0.0, 0.0, 1.0])}


def timed(f):
    t = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t


out, first_apply = timed(lambda: tess.apply(inputs))
_, warm_apply = timed(lambda: tess.apply(inputs))
vjp = lambda: tess.vector_jacobian_product(inputs, ["theta"], ["tumour_volume"], ct)  # noqa: E731
_, first_vjp = timed(vjp)
_, warm_vjp = timed(vjp)
tv = np.asarray(out["tumour_volume"])
res = {
    "sysimage": os.environ.get("PYTHON_JULIACALL_SYSIMAGE", ""),
    "load_s": t_load,
    "first_apply_s": first_apply,
    "warm_apply_s": warm_apply,
    "first_vjp_s": first_vjp,
    "warm_vjp_s": warm_vjp,
    "log10_ratio": float(np.log10(tv[-1] / tv[0])),
}
print(json.dumps(res))
with open(sys.argv[1], "w") as f:
    json.dump(res, f)
