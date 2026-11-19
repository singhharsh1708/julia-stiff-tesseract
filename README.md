# julia-stiff-tesseract

A stiff pharmacology model in Julia served as a [Tesseract](https://github.com/pasteurlabs/tesseract-core),
differentiated through the solver, chained with a JAX Tesseract and calibrated with
[tesseract-jax](https://github.com/pasteurlabs/tesseract-jax). What got in the way, and what would make Julia
Tesseracts easier, is in [docs/findings.md](docs/findings.md).

The model is Lindauer et al. 2017 (pembrolizumab: two-compartment PK, FcRn recycling, PD-1 binding in blood and
tumour, occupancy-driven tumour growth inhibition) as rebuilt in
[QSPAtlas.jl](https://github.com/singhharsh1708/QSPAtlas.jl): 11 states, 27 parameters, Rodas5P, a dose every
three weeks for 26 weeks. `lindauer/julia/src/lindauer_model.jl` is QSPAtlas's ModelingToolkit file, unchanged
(MIT). `tools/generate.jl` turns it into plain Julia once, so the Tesseract loads OrdinaryDiffEq and ForwardDiff
but not ModelingToolkit; `tools/test.jl` checks the generated code against ModelingToolkit and QSPAtlas's numbers.
No data from the paper is stored here; the Figure 4 experiment reads QSPAtlas's digitised values at run time.

## Layout

| path | what |
|---|---|
| `lindauer/` | the Julia Tesseract (juliacall): `tesseract_api.py`, config, Julia project with the generated model |
| `allometry/` | a JAX Tesseract: mouse-to-human allometric scaling, the first stage of the chain |
| `pipeline/` | the Tesseract in-process, via `from_source`, from the built image; the chain calibration; `check_gradients` with injected faults |
| `experiments/` | the CI experiments: VJP routes, the EnzymeVJP probe, a system image, Figure 4 calibration, `check_gradients` step rules |
| `tools/` | development environment (ModelingToolkit, SciMLSensitivity, Zygote, Enzyme): code generation and tests |
| `results/` | raw results per run, each with a README |
| `docs/` | findings, the reflection that set the experiments, the working log |

## Running

```sh
julia --project=tools -e 'using Pkg; Pkg.instantiate()'
julia --project=tools tools/test.jl                 # generated model vs ModelingToolkit and QSPAtlas

julia --project=lindauer/julia -e 'using Pkg; Pkg.instantiate()'
source env.sh                                       # juliacall uses this Julia and lindauer/julia
uv venv && uv pip install "tesseract-core[runtime]==1.15.0" juliacall==0.9.37 numpy jax scipy
.venv/bin/python pipeline/inprocess.py

tesseract build lindauer                            # the image (about 5 GB)
```

`pipeline/check_gradients_faults.py` and `experiments/gradcheck_grid.py` need `check_gradients` from the
unreleased tesseract-jax#300; `.github/workflows/experiments.yml` installs it from a pinned commit.

## CI

- `test.yml`, on pushes to main and on pull requests: the Julia tests, a check that the committed generated code
  is current, and the Tesseract's apply and VJP through juliacall.
- `experiments.yml`, on demand and when the code it runs changes: each experiment in its own job, then a summary
  table in the run summary. Saved runs are in `results/`.

`pipeline/from_source.py`, `calibrate_caption.py` and `nan_roundtrip.py` are the first pass's integration runs;
CI does not run them (their results are in `results/local-2026-11-18/`).
