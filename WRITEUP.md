# A stiff Julia model behind a Tesseract

Harsh Singh, November 2026. Code and notes: https://github.com/singhharsh1708/julia-stiff-tesseract

## What I built

The pembrolizumab model of Lindauer et al. 2017 (two-compartment PK, FcRn recycling, PD-1 binding in blood and
tumour, occupancy-driven tumour growth inhibition), from my QSPAtlas.jl: 11 states in ModelingToolkit, stiff (fast
binding next to weeks of tumour growth), Rodas5P, Q3W doses for 26 weeks. It runs as a Julia Tesseract through
juliacall and `Tesseract.from_source`, with VJP and JVP from forward-mode AD (ForwardDiff) through the solver.
In a chain, an allometric-scaling Tesseract (JAX; mouse rate constants to man) feeds it, and JAX glue turns tumour
volume into the paper's endpoint, log10 V(26 wk)/V(0).

The Julia side matches QSPAtlas: -6.006 / -4.746 at 5 / 0.5 mg/kg as published (QSPAtlas records -6.0 / -4.7) and
-1.302 / -0.401 with its fix. ForwardDiff through the solver agrees with finite differences to 1e-6 to 1e-9, and
so does the chain gradient through both Tesseracts and the glue: (-10.458701, 300.790373) against
(-10.458719, 300.790361).

Calibrating blood volume and the allometric exponent to the paper's -1.29 / -0.39 with `jax.value_and_grad` and
L-BFGS-B took 17 evaluations in 2.4 s once warm, about 0.14 s each (per evaluation, each of the two doses is solved
once plainly and once with dual numbers). It landed on 4.49 L and exponent -0.094. QSPAtlas fixed the same gap
differently, with 5 L and a slower tumour-growth scenario at exponent -0.25, so the fit shows calibration through
the chain working; it does not single out one biological explanation.

## check_gradients (tesseract-jax#300) on a real model

Same chain, one plausible bug at a time: the Julia Tesseract's d/dSLtg off by the day/hour factor 24; the
allometry Tesseract's d/d exponent missing ln(body-weight ratio); SLtg detached with `stop_gradient` in the glue.

As submitted, #300 flagged each injected fault at the pipeline level and got the JAX Tesseract right, but its rows
for the Julia Tesseract failed in every run, the clean one included, with a NaN finite difference. Steps are
scaled by each input's RMS, and the parameter vector mixes a volume in uL (~3e6) with rate constants (~6e-6 and
~6e-3), so the step threw the small entries far out of range; the finite difference came back NaN, most likely
from failed solves (the server's stderr was not captured). For the clean run and the glue
fault its verdict pointed at the Julia Tesseract, with a warning that the finite difference was uncertain. So at
this point it could not localise reliably.

With elementwise relative steps (RMS only for zero entries), the four runs at the same point and seed:

| injected bug | pipeline | allometry (JAX) | Lindauer (Julia) | verdict |
|---|---|---|---|---|
| none | ok | ok | ok | gradients agree |
| Julia VJP unit slip | FAIL | ok | FAIL | start at the Julia Tesseract |
| allometry VJP slip | FAIL | FAIL | ok | start at the allometry Tesseract |
| `stop_gradient` in glue | FAIL | ok | ok | the JAX code between the Tesseracts (its hint list starts with `stop_gradient`) |

Each check took 4.5-4.9 s. Per-Tesseract checks cannot see the glue bug, and a pipeline check alone cannot say
where a bug is. Elementwise steps barely probe entries near zero, though, so that needs a test (and maybe a floor)
before it goes into #300. And when `apply` returns NaN at a perturbed point, #300 calls the finite difference
"too uncertain" rather than saying so, which hides the cause; that needs a clearer message too.

## The Julia template on jpb/julia, on this model

I found the branch after my first draft and tried it on the same model. Its `PYTHON_JULIACALL_EXE` /
`PYTHON_JULIACALL_PROJECT` setup works and is cleaner than what I had: I had gone through juliapkg, which on
first import picked the newest Julia juliaup had (1.13) over the project's 1.10 and built a second environment.

For the gradients, a standalone probe (julia/scripts/enzyme_probe.jl, enzyme_sens_timing.jl) makes the two
`autodiff` calls of julia_recipes.jl at 32bf06d on this solve, on Julia 1.10.12 (the template's image uses 1.11),
Enzyme 0.13.221, OrdinaryDiffEq 7.8.1, LinearSolve 5.18.3, ModelingToolkit 11.46.1:

- With the model built lazily inside the differentiated function, both modes fail in about a minute in the
  symbolic MTK build (IllegalTypeAnalysisException). Building it at include time and passing it in avoids that.
- Reverse mode (the VJP) works once SciMLSensitivity is loaded: DiffEqBase's Enzyme rule then hands the solve to
  SciMLSensitivity's adjoint, and the result agrees with ForwardDiff to 6e-10. Without it, Enzyme goes into
  Rodas5P and fails after about four minutes with MixedReturnException on LinearSolve's rule for the
  `LinearProblem` that Rodas5P builds in its cache.
- With SciMLSensitivity loaded, the first VJP in a process took 495 s, almost all compilation; later VJPs took
  1.6 s, also at a new point. ForwardDiff on the same two-parameter VJP takes about 3 s the first time and
  75 ms after. The adjoint's cost does not grow with the number of parameters and ForwardDiff's does, so the
  gap narrows for larger fits.
- Forward mode (the JVP), run without SciMLSensitivity, was still compiling about 21 minutes in when my time
  limit stopped it. DiffEqBase's Enzyme rule covers reverse mode only, so loading SciMLSensitivity should not
  change that, though I have not run it.

So for a stiff SciML model the template gives a correct VJP but no JVP within 21 minutes. Going by this probe,
a served Tesseract's first gradient request would wait about eight minutes, and each later one is about 20 times
slower than ForwardDiff for this two-parameter fit. The gradient
endpoints in the template are plain Python, so a user can swap them, as I did; what would help is a supported
way to do it that keeps the recipe's input flattening, for example `julia_vjp` / `julia_jvp` accepting a Julia
derivative function or an AD backend, plus a note that DiffEq solves need SciMLSensitivity loaded. This model
could serve as the template's stiff test case.

## Integration follow-ups, most useful first

1. `from_image` with colima's default mounts on macOS. With default arguments every call failed with a
   `PermissionError` on `/tesseract/output_data/run_<id>`: the SDK mounts a `tempfile.mkdtemp()` folder, which on
   macOS is under /var/folders, and colima shares only $HOME with its VM by default, so the container sees an empty
   root-owned directory. An `output_path` under $HOME fixes it, but the error does not point at the cause.
2. A failed solve ended the optimisation. An unbounded L-BFGS step reached parameters where Rodas5P stopped at
   MaxIters; my error came back as HTTP 500, then as a JAX `CpuCallback` error that ended the run. A NaN from the
   Tesseract (QSPAtlas's convention for a failed solve) passes cleanly through `from_source` and tesseract-jax, and
   the caller can then treat a non-finite loss as a rejected step. Documenting that convention would help.
3. First-call latency. `from_source` was healthy after 8.6 s with warm caches (31.1 s cold, which includes building
   `.tesseract-venv`). Every new process then spends about 13 s on its first `apply` and 3.5 s on its first VJP;
   about 3 s of each is Julia compiling the model and its derivative, and the rest of the apply is juliacall's
   conversions. Warm calls take 16-46 ms (apply) and 75 ms (VJP). The SDK docstrings mention Julia start-up
   under `skip_health_check`; the first call is the part a user does not expect, since the health check
   passes before it.
4. Container numbers. `tesseract build` took 12 minutes for a 4.77 GB image (Julia, the precompiled environment,
   the Python runtime). From the image: healthy after 13 s, first apply 23 s, first VJP 5.7 s, then 29 / 116 ms.

What worked without trouble: `from_source` passes the environment through, eight concurrent requests to the
served Julia Tesseract returned the same correct result, and NaN round-trips cleanly.

On the Julia side, ModelingToolkit 11 silently dropped a new `u0` when I restarted the solve at each dose (both
`remake` and the attached initialization data rebuild it from the parameters), which gave a tumour with no drug
instead of an error; building the problem with `build_initializeprob = false` fixed it.
