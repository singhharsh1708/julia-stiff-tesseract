"""Step 6: the built image (Tesseract.from_image): start-up, first and warm calls."""

import time

import numpy as np
from tesseract_core import Tesseract

INPUTS = {"theta": np.array([5.0e6, 0.0036, 6.247e-7, 0.00246]), "fields": ["V_blood", "L0", "SLtg", "KdegPD1"],
          "scenario": "med_growth", "dose_mgkg": 5.0, "interval_days": 21.0, "times_days": np.array([0.0, 91.0, 182.0])}
CT = {"tumour_volume": np.array([0.0, 0.0, 1.0])}
t0 = time.perf_counter()
with Tesseract.from_image("lindauer-julia:0.1.0", output_path=str(__import__("pathlib").Path(__file__).resolve().parents[1] / ".tess-out")) as tess:
    print(f"healthy after {time.perf_counter() - t0:.1f} s", flush=True)
    for label, f in [("apply", lambda: tess.apply(INPUTS)),
                     ("vjp", lambda: tess.vector_jacobian_product(INPUTS, ["theta"], ["tumour_volume"], CT))]:
        t0 = time.perf_counter(); out = f(); t1 = time.perf_counter() - t0
        t0 = time.perf_counter(); f(); t2 = time.perf_counter() - t0
        print(f"{label} first {t1:.2f} s, warm {t2 * 1e3:.1f} ms", flush=True)
        if label == "apply":
            tv = np.asarray(out["tumour_volume"]); print(f"log10 ratio {np.log10(tv[-1] / tv[0]):.3f}", flush=True)
