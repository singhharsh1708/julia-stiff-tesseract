"""Does a NaN from a served Tesseract pass through tesseract-jax without an error?"""

import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)
with Tesseract.from_source("pipeline/nan_toy/tesseract_api.py", startup_timeout=300) as t:
    loss = lambda x: jnp.sum(apply_tesseract(t, {"x": x})["y"] ** 2)  # noqa: E731
    print("apply:", apply_tesseract(t, {"x": jnp.ones(3)})["y"])
    v, g = jax.value_and_grad(loss)(jnp.ones(3))
    print("value_and_grad:", v, g)
