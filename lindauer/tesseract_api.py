"""Lindauer et al. 2017 (pembrolizumab PK / PD-1 occupancy / tumour growth) solved in Julia.

The model is QSPAtlas.jl's ModelingToolkit rebuild, generated once into plain Julia (../tools/generate.jl), solved
with Rodas5P with a restart at each Q3W dose (julia/src/StiffTesseract.jl). Derivatives come from forward-mode AD
through the solver (ForwardDiff): the fastest warm route in CI run 1 up to 16 parameters, with no compile wait.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

_JULIA_PROJECT = Path(__file__).parent / "julia"

from juliacall import Main as jl  # noqa: E402

jl.seval(f'import Pkg; Pkg.activate(raw"{_JULIA_PROJECT}"; io = devnull)')
jl.seval("using StiffTesseract")
_ST = jl.StiffTesseract


class InputSchema(BaseModel):
    theta: Differentiable[Array[(None,), Float64]] = Field(
        description="Values of the raw Lindauer parameters named in `fields` (paper units)."
    )
    fields: list[str] = Field(
        description="Names of the raw parameters in `theta`, e.g. V_blood, L0, SLtg, KdegPD1."
    )
    scenario: str = Field(
        default="med_growth", description="Table S3 tumour growth scenario for the rest."
    )
    dose_mgkg: float = Field(default=5.0, description="Dose per administration [mg/kg].")
    interval_days: float = Field(default=21.0, description="Dosing interval [days].")
    times_days: Array[(None,), Float64] = Field(
        description="Sample times in [0, 182] days."
    )


class OutputSchema(BaseModel):
    tumour_volume: Differentiable[Array[(None,), Float64]] = Field(
        description="Tumour volume at the sample times [mm^3]."
    )
    receptor_occupancy: Differentiable[Array[(None,), Float64]] = Field(
        description="PD-1 receptor occupancy in the tumour at the sample times [%]."
    )


def _args(inputs: InputSchema) -> tuple:
    return (
        np.asarray(inputs.theta, dtype=np.float64),
        jl.Vector[jl.String](list(inputs.fields)),
        inputs.scenario,
        float(inputs.dose_mgkg),
        float(inputs.interval_days),
        np.asarray(inputs.times_days, dtype=np.float64),
    )


def apply(inputs: InputSchema) -> OutputSchema:
    tv, ro = _ST.simulate(*_args(inputs))
    return OutputSchema(tumour_volume=np.asarray(tv), receptor_occupancy=np.asarray(ro))


def abstract_eval(abstract_inputs: Any) -> dict:
    n = abstract_inputs.times_days.shape
    return {
        "tumour_volume": ShapeDType(shape=n, dtype="float64"),
        "receptor_occupancy": ShapeDType(shape=n, dtype="float64"),
    }


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]) -> dict:
    J = np.asarray(_ST.jacobian(*_args(inputs)))
    n = len(inputs.times_days)
    rows = {"tumour_volume": J[:n], "receptor_occupancy": J[n:]}
    return {out: {"theta": rows[out]} for out in jac_outputs}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict:
    n = len(inputs.times_days)
    ct_tv = np.asarray(cotangent_vector.get("tumour_volume", np.zeros(n)), dtype=np.float64)
    ct_ro = np.asarray(cotangent_vector.get("receptor_occupancy", np.zeros(n)), dtype=np.float64)
    g = np.asarray(_ST.vjp(*_args(inputs), ct_tv, ct_ro))
    if os.environ.get("STIFF_FAULT") == "lindauer_vjp" and "SLtg" in inputs.fields:
        # injected fault for the gradient-check test: SLtg's derivative off by the day/hour factor
        g = g.copy()
        g[list(inputs.fields).index("SLtg")] *= 24.0
    return {"theta": g} if "theta" in vjp_inputs else {}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict:
    tangent = np.asarray(
        tangent_vector.get("theta", np.zeros(len(inputs.theta))), dtype=np.float64
    )
    d_tv, d_ro = _ST.jvp(*_args(inputs), tangent)
    out = {"tumour_volume": np.asarray(d_tv), "receptor_occupancy": np.asarray(d_ro)}
    return {k: v for k, v in out.items() if k in jvp_outputs}
