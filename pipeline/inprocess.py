"""The Tesseract in-process (from_tesseract_api): import cost, first and warm calls."""

import time

import numpy as np

t0 = time.perf_counter()
from tesseract_core import Tesseract  # noqa: E402

tess = Tesseract.from_tesseract_api("lindauer/tesseract_api.py")
t_import = time.perf_counter() - t0

inputs = {
    "theta": np.array([5.0e6, 0.0036, 6.247e-7, 0.00246]),
    "fields": ["V_blood", "L0", "SLtg", "KdegPD1"],
    "scenario": "med_growth",
    "dose_mgkg": 5.0,
    "interval_days": 21.0,
    "times_days": np.array([0.0, 91.0, 182.0]),
}
t0 = time.perf_counter(); out = tess.apply(inputs); t_first = time.perf_counter() - t0
t0 = time.perf_counter(); out = tess.apply(inputs); t_warm = time.perf_counter() - t0
tv = np.asarray(out["tumour_volume"])
print(f"load tesseract_api (Julia start + packages): {t_import:.1f} s")
print(f"apply first {t_first:.2f} s, warm {t_warm * 1e3:.1f} ms; log10 V(26wk)/V0 = {np.log10(tv[-1] / tv[0]):.3f}")
ct = {"tumour_volume": np.array([0.0, 0.0, 1.0 / tv[-1]])}
t0 = time.perf_counter()
g = tess.vector_jacobian_product(inputs, vjp_inputs=["theta"], vjp_outputs=["tumour_volume"], cotangent_vector=ct)
t_vjp1 = time.perf_counter() - t0
t0 = time.perf_counter()
g = tess.vector_jacobian_product(inputs, vjp_inputs=["theta"], vjp_outputs=["tumour_volume"], cotangent_vector=ct)
t_vjp2 = time.perf_counter() - t0
print(f"vjp first {t_vjp1:.2f} s, warm {t_vjp2 * 1e3:.1f} ms; d ln V(26wk) / d theta = {np.asarray(g['theta'])}")
