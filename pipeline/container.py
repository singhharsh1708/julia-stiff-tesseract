"""The built image through Tesseract.from_image: start-up, first and warm calls.

    tesseract build lindauer --tag <tag>
    python pipeline/container.py <image:tag> [out.json]

The output folder is put under this repository: on macOS with colima's default mounts, from_image's default
(a tempfile.mkdtemp() folder under /var/folders) is not shared with the VM and every call fails.
"""

import json
import pathlib
import sys
import time

import numpy as np
from tesseract_core import Tesseract

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "lindauer-julia:0.2.0"
INPUTS = {"theta": np.array([5.0e6, 0.0036, 6.247e-7, 0.00246]), "fields": ["V_blood", "L0", "SLtg", "KdegPD1"],
          "scenario": "med_growth", "dose_mgkg": 5.0, "interval_days": 21.0, "times_days": np.array([0.0, 91.0, 182.0])}
CT = {"tumour_volume": np.array([0.0, 0.0, 1.0])}
out_dir = pathlib.Path(__file__).resolve().parents[1] / ".tess-out"
out_dir.mkdir(exist_ok=True)

res = {"image": IMAGE}
t0 = time.perf_counter()
with Tesseract.from_image(IMAGE, output_path=str(out_dir)) as tess:
    res["healthy_s"] = time.perf_counter() - t0
    calls = {"apply": lambda: tess.apply(INPUTS),
             "vjp": lambda: tess.vector_jacobian_product(INPUTS, ["theta"], ["tumour_volume"], CT)}
    for label, f in calls.items():
        t0 = time.perf_counter()
        out = f()
        res[f"first_{label}_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        f()
        res[f"warm_{label}_s"] = time.perf_counter() - t0
        if label == "apply":
            tv = np.asarray(out["tumour_volume"])
            res["log10_ratio"] = float(np.log10(tv[-1] / tv[0]))
print(json.dumps(res))
if len(sys.argv) > 2:
    pathlib.Path(sys.argv[2]).write_text(json.dumps(res))
