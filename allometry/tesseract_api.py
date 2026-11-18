"""Allometric scaling of mouse rate constants to man: human = mouse * (BW_human / BW_mouse) ** exponent.

Lindauer et al. 2017 scale the tumour-kill rate SLtg and the PD-1 turnover KdegPD1 this way (Table S3 'allom'
scenarios; 20 g mouse, 70 kg man, exponent -0.25).
"""

import os
from typing import Any

import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

jax.config.update("jax_enable_x64", True)


class InputSchema(BaseModel):
    mouse_values: Differentiable[Array[(None,), Float64]] = Field(description="Mouse rate constants.")
    exponent: Differentiable[Float64] = Field(default=-0.25, description="Allometric exponent.")
    bw_ratio: float = Field(default=70.0 / 0.02, description="Human / mouse body weight.")


class OutputSchema(BaseModel):
    human_values: Differentiable[Array[(None,), Float64]] = Field(description="Scaled to man.")


def _scale(mouse_values, exponent, bw_ratio):
    return mouse_values * jnp.asarray(bw_ratio) ** exponent


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(human_values=_scale(inputs.mouse_values, inputs.exponent, inputs.bw_ratio))


def abstract_eval(abstract_inputs: Any) -> dict:
    return {"human_values": ShapeDType(shape=abstract_inputs.mouse_values.shape, dtype="float64")}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    _, pullback = jax.vjp(
        lambda m, e: _scale(m, e, inputs.bw_ratio), jnp.asarray(inputs.mouse_values), jnp.asarray(inputs.exponent)
    )
    ct = cotangent_vector.get("human_values", jnp.zeros_like(inputs.mouse_values))
    d_mouse, d_exp = pullback(jnp.asarray(ct))
    if os.environ.get("STIFF_FAULT") == "allometry_vjp":
        # injected fault for the gradient-check test: d/d exponent without the ln(bw_ratio) factor
        d_exp = d_exp / jnp.log(inputs.bw_ratio)
    grads = {"mouse_values": d_mouse, "exponent": d_exp}
    return {k: v for k, v in grads.items() if k in vjp_inputs}


def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    m, e = jnp.asarray(inputs.mouse_values), jnp.asarray(inputs.exponent)
    dm = jnp.asarray(tangent_vector.get("mouse_values", jnp.zeros_like(m)))
    de = jnp.asarray(tangent_vector.get("exponent", 0.0))
    _, out = jax.jvp(lambda m, e: _scale(m, e, inputs.bw_ratio), (m, e), (dm, de))
    return {"human_values": out}
