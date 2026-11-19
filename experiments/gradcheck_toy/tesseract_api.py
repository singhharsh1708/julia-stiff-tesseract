"""Toy Tesseract for testing check_gradients' step sizes: one input vector mixing scales, signs and a near-zero
entry. TOY_FAULT=<i> scales the VJP's entry i by 1.5 (a wrong derivative); TOY_NAN=1 makes apply return NaN when
x[1] moves above 2.5e-6 (as a failed solve does at a perturbed point)."""

import os

import jax
import jax.numpy as jnp
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

jax.config.update("jax_enable_x64", True)


class InputSchema(BaseModel):
    x: Differentiable[Array[(4,), Float64]]


class OutputSchema(BaseModel):
    y: Differentiable[Array[(3,), Float64]]


def _f(x):
    return jnp.stack([jnp.log(x[0]) + 1e6 * x[1], x[2] + x[3] ** 2, jnp.sin(1e3 * x[3])])


def apply(inputs: InputSchema) -> OutputSchema:
    x = jnp.asarray(inputs.x)
    y = _f(x)
    if os.environ.get("TOY_NAN") and x[1] > 2.5e-6:
        y = jnp.full(3, jnp.nan)
    return OutputSchema(y=y)


def abstract_eval(abstract_inputs):
    return {"y": ShapeDType(shape=(3,), dtype="float64")}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    _, pullback = jax.vjp(_f, jnp.asarray(inputs.x))
    (g,) = pullback(jnp.asarray(cotangent_vector["y"]))
    fault = os.environ.get("TOY_FAULT")
    if fault:
        g = g.at[int(fault)].multiply(1.5)
    return {"x": g}
