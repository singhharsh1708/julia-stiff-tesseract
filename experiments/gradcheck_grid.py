"""How should check_gradients (tesseract-jax#300) size its finite-difference steps?

    python experiments/gradcheck_grid.py <toy|chain> <out.json>

Step rules (each patched into tesseract_jax.gradient_check._scaled_direction):
  rms          as in #300: every element of an input stepped by that input's RMS
  elementwise  each element by its own magnitude (RMS for exact zeros)
  floor        each element by max(|x_i|, 1e-3 * RMS)
  user_scale   each element by a typical scale, for the pipeline parameter and the Tesseract input alike

toy:   a JAX Tesseract with x = [3e6, 2e-6, 1e-12, -1e-3] (mixed scales, a sign, an entry near zero that matters
       on an O(1) scale); a wrong derivative injected on each entry in turn; and a NaN at a perturbed point.
chain: the allometry (JAX) -> Lindauer (Julia, from_source) chain with the four faults of
       pipeline/check_gradients_faults.py.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

RULES = ["rms", "elementwise", "floor", "user_scale"]


def patch(rule, user_scale=None):
    import jax
    import tesseract_jax.gradient_check as gc

    def scaled_direction(key, x):
        direction = {}
        for k, (name, leaf) in zip(jax.random.split(key, len(x)), x.items(), strict=True):
            leaf = np.asarray(leaf)
            rms = float(np.sqrt(np.mean(np.square(leaf, dtype=np.float64)))) if leaf.size else 0.0
            if rule == "rms":
                scale = np.full(leaf.shape, rms or 1.0)
            elif rule == "elementwise":
                scale = np.where(leaf != 0, np.abs(leaf), rms or 1.0)
            elif rule == "floor":
                scale = np.maximum(np.abs(leaf), 1e-3 * (rms or 1.0))
            else:
                scale = np.broadcast_to(np.asarray(user_scale.get(name, rms or 1.0)), leaf.shape)
            z = np.asarray(jax.random.normal(k, leaf.shape, leaf.dtype))
            direction[name] = np.asarray(z * scale, dtype=leaf.dtype)
        return direction

    gc._scaled_direction = scaled_direction


def row_summary(report):
    text = str(report)
    lines = [ln for ln in text.splitlines() if ln.strip().startswith(("pipeline", "#"))]
    verdict = text.strip().splitlines()[-1]
    return {"passed": bool(report.passed), "rows": [" ".join(ln.split()) for ln in lines], "verdict": verdict}


def toy(rule):
    import jax
    import jax.numpy as jnp
    from tesseract_core import Tesseract
    from tesseract_jax import apply_tesseract, check_gradients

    jax.config.update("jax_enable_x64", True)
    # typical scales, given for the pipeline parameter and for the Tesseract's own input (which is what the
    # pushforward of the pipeline direction would supply automatically)
    patch(rule, {"params['x']": np.array([1e6, 1e-6, 1.0, 1e-3]), "x": np.array([1e6, 1e-6, 1.0, 1e-3])})
    tess = Tesseract.from_tesseract_api("experiments/gradcheck_toy/tesseract_api.py")
    w = jnp.array([1.0, 1.0, 1.0])
    loss = lambda p: jnp.sum(w * apply_tesseract(tess, {"x": p["x"]})["y"])  # noqa: E731
    params = {"x": jnp.array([3e6, 2e-6, 1e-12, -1e-3])}
    out = {}
    for case in ["none", "0", "1", "2", "3", "nan"]:
        os.environ.pop("TOY_FAULT", None)
        os.environ.pop("TOY_NAN", None)
        nan_log = None
        if case == "nan":
            nan_log = tempfile.NamedTemporaryFile(suffix=".log", delete=False).name
            os.environ["TOY_NAN"] = nan_log
        elif case != "none":
            os.environ["TOY_FAULT"] = case
        out[case] = row_summary(check_gradients(loss, params))
        if nan_log:  # how many applies returned NaN: zero would mean this case tested nothing
            with open(nan_log) as f:
                out[case]["nan_applies"] = len(f.readlines())
            os.remove(nan_log)
    return out


def chain(rule):
    out = {}
    for fault in ["clean", "lindauer_vjp", "allometry_vjp", "glue"]:
        r = subprocess.run([sys.executable, "pipeline/check_gradients_faults.py", fault, "--rule", rule],
                           capture_output=True, text=True, timeout=3600)
        m = re.search(r"^=== fault: .*passed: (True|False)$", r.stdout, re.MULTILINE)
        report = r.stdout[m.start():].strip().splitlines() if m else []
        out[fault] = {"returncode": r.returncode, "report": report, "stderr": r.stderr[-600:] if r.returncode else ""}
        if m:
            out[fault]["passed"] = m.group(1) == "True"
    return out


if __name__ == "__main__":
    what, outfile = sys.argv[1], sys.argv[2]
    results = {}
    for rule in RULES if what == "toy" else ["rms", "elementwise", "floor"]:
        results[rule] = toy(rule) if what == "toy" else chain(rule)
        print(rule, json.dumps(results[rule])[:400], flush=True)
    with open(outfile, "w") as f:
        json.dump(results, f, indent=1)
