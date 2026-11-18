# julia-stiff-tesseract

A stiff pharmacology model in Julia behind a [Tesseract](https://github.com/pasteurlabs/tesseract-core), chained
with a JAX Tesseract and calibrated with [tesseract-jax](https://github.com/pasteurlabs/tesseract-jax).

The model is Lindauer et al. 2017 (pembrolizumab: two-compartment PK, FcRn recycling, PD-1 binding in blood and
tumour, occupancy-driven tumour growth inhibition) as rebuilt in
[QSPAtlas.jl](https://github.com/singhharsh1708/QSPAtlas.jl): 11 states in ModelingToolkit, solved with Rodas5P,
doses every three weeks for 26 weeks. `julia/src/lindauer_model.jl` is QSPAtlas's model file, unchanged (MIT).
No data from the paper is included.

- [WRITEUP.md](WRITEUP.md): what got in the way, with numbers.
- [REFLECTION.md](REFLECTION.md): what the first pass did weakly and what the experiments below test.
- [notes/friction.md](notes/friction.md) and `notes/raw-*.log`: the running log and raw outputs behind the numbers.

## Layout

| path | what |
|---|---|
| `julia/` | Julia project; `src/StiffTesseract.jl` solves the model and gives VJP / JVP by ForwardDiff |
| `tesseract_api.py`, `tesseract_config.yaml` | the Julia Tesseract (juliacall) |
| `allometry/` | the JAX Tesseract (mouse-to-human allometric scaling) |
| `scripts/` | steps 2-6: in-process, `from_source`, calibration, gradient checks, the container |
| `experiments/` | experiments run in CI (see `.github/workflows/experiments.yml`) |

## Running locally

```sh
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
source scripts_env.sh                     # juliacall uses this Julia and project
uv venv && uv pip install "tesseract-core[runtime]==1.15.0" juliacall==0.9.37 jax numpy scipy
.venv/bin/python scripts/step2_inprocess.py
# steps 4-5 also need tesseract-jax; step 5 uses check_gradients from the unreleased tesseract-jax#300
```

## Experiments in CI

1. **VJP routes** at 2, 8 and 16 parameters: ForwardDiff through the solve; Zygote with SciMLSensitivity's
   adjoints (GaussAdjoint with automatic backends, with EnzymeVJP and with ReverseDiffVJP; InterpolatingAdjoint
   with EnzymeVJP); and the Enzyme call used by tesseract-core's Julia template (branch `jpb/julia`). First call,
   warm call, and agreement with ForwardDiff, one process per route and size.
2. **First-call latency**: a custom system image (PackageCompiler) built from what a Python-driven call
   compiles, loaded through juliacall, against no system image.

Results are in each run's summary and artifacts.
