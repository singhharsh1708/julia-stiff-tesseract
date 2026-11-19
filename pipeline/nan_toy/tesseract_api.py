"""Toy Tesseract whose apply and VJP return NaN, as a failed solve does."""

import numpy as np
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float64]]


class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]]


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(y=np.full_like(inputs.x, np.nan))


def abstract_eval(abstract_inputs):
    return {"y": ShapeDType(shape=abstract_inputs.x.shape, dtype="float64")}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return {"x": np.full_like(inputs.x, np.nan)}
