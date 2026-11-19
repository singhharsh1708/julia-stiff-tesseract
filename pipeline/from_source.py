"""The Tesseract through Tesseract.from_source (own venv, own process, HTTP)."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tesseract_core import Tesseract

INPUTS = {
    "theta": np.array([5.0e6, 0.0036, 6.247e-7, 0.00246]),
    "fields": ["V_blood", "L0", "SLtg", "KdegPD1"],
    "scenario": "med_growth",
    "dose_mgkg": 5.0,
    "interval_days": 21.0,
    "times_days": np.array([0.0, 91.0, 182.0]),
}
CT = {"tumour_volume": np.array([0.0, 0.0, 1.0])}


def timed(f):
    t0 = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t0


def run(label, startup_timeout):
    print(f"--- {label} (startup_timeout={startup_timeout} s)", flush=True)
    t0 = time.perf_counter()
    try:
        with Tesseract.from_source("lindauer/tesseract_api.py", startup_timeout=startup_timeout) as tess:
            print(f"healthy after {time.perf_counter() - t0:.1f} s", flush=True)
            out, t = timed(lambda: tess.apply(INPUTS))
            tv = np.asarray(out["tumour_volume"])
            print(f"apply first {t:.2f} s; log10 ratio {np.log10(tv[-1] / tv[0]):.3f}", flush=True)
            _, t = timed(lambda: tess.apply(INPUTS))
            print(f"apply warm {t * 1e3:.1f} ms", flush=True)
            vjp = lambda: tess.vector_jacobian_product(  # noqa: E731
                INPUTS, vjp_inputs=["theta"], vjp_outputs=["tumour_volume"], cotangent_vector=CT
            )
            _, t = timed(vjp)
            print(f"vjp first {t:.2f} s", flush=True)
            _, t = timed(vjp)
            print(f"vjp warm {t * 1e3:.1f} ms", flush=True)
            with ThreadPoolExecutor(4) as pool:
                outs, t = timed(lambda: list(pool.map(lambda _: tess.apply(INPUTS), range(8))))
            same = all(np.allclose(o["tumour_volume"], tv) for o in outs)
            print(f"8 applies from 4 client threads: {t:.2f} s, all equal: {same}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED after {time.perf_counter() - t0:.1f} s: {type(e).__name__}: {str(e)[:400]}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "default"):
        run("default timeout", 30.0)
    if mode in ("all", "long"):
        run("long timeout", 900.0)
