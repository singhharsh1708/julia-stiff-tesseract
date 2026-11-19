# Friction log (numbers as measured)

Machine: Apple M-series, 10 cores, macOS; Julia 1.10.12 (juliaup); other jobs were running on the machine, so
wall times are upper bounds. Packages were already downloaded in ~/.julia (no network time counted).

## Julia environment

- 2026-11-18: fresh project, `Pkg.add(OrdinaryDiffEq, SciMLSensitivity, Zygote, ForwardDiff)` + precompile:
  184 s wall, 539 s CPU.

## juliacall (Python <-> Julia bridge)

- `import juliacall` in a fresh venv (juliacall 0.9.37) runs juliapkg on first import: it picks the newest Julia
  juliaup has (1.13.3 here), not the default channel (1.10.12, which the Julia project was precompiled with),
  creates `.venv/julia_env`, updates every registry, installs PythonCall and precompiles 12 packages (25 s).
  Activating our project inside that session would then precompile OrdinaryDiffEq / ModelingToolkit again for a
  second Julia version. Fix: PYTHON_JULIAPKG_EXE (pin the Julia binary) and PYTHON_JULIAPKG_PROJECT (use our
  project, juliapkg adds PythonCall to it). Nothing in tesseract-core's docs mentions this; a Julia user hits it
  on the first `from_source`.
- Adding ModelingToolkit 11.46.1 (+ SciMLStructures, SymbolicIndexingInterface): 58 more packages, 243 s wall.
  Julia environment from scratch, packages already on disk: about 7 minutes of precompilation.

## The model in Julia (step 1)

- Dose-by-restart reproduces QSPAtlas's callback dosing exactly: log10 V(26 wk)/V(0) = -6.006 / -4.746 as
  published (QSPAtlas -6.0 / -4.7), -1.302 / -0.401 with V_blood 5 L, med_growth (QSPAtlas -1.302 / -0.401).
- Forward-mode AD (ForwardDiff through Rodas5P, analytic Jacobian from MTK) against central finite differences in
  log parameter space: max relative difference 1.1e-6 (V_blood), 2.6e-8 (L0), 2.3e-9 (SLtg), 3.0e-7 (KdegPD1).
- Timings, 14 sample times, 4 parameters: `using StiffTesseract` 6.6-17 s (machine load varied); first simulate
  3.5-4.4 s (compilation); warm simulate 16 ms; first Jacobian 3.5 s, warm Jacobian 72 ms (28 outputs x 4).
- Two silent traps in ModelingToolkit 11 when restarting a solve with a new state (Julia-side, not Tesseract):
  1. `remake(prob; u0 = u, p = p)` with an MTKParameters `p` returns `u0` rebuilt from the initial values stored
     in `p`; the passed `u0` is dropped without a warning.
  2. Even `ODEProblem(prob.f, u, tspan, p)` loses `u`: while `prob.f` carries `initialization_data`, every
     solve rebuilds `u0` from `p`'s initial values, and `initializealg = NoInit()` does not stop it.
  Both gave a tumour with no drug (receptor occupancy 0 %) instead of an error. Fix: build the template with
  `build_initializeprob = false`. Cost about an hour; anyone wrapping an MTK model with restarts or events will
  meet it.

## Through juliacall, in-process (step 2, `Tesseract.from_tesseract_api`)

- With PYTHON_JULIAPKG_EXE / PYTHON_JULIAPKG_PROJECT pinned, juliapkg adds PythonCall to our project
  (Project.toml +8 lines) and precompiles 8 packages (19 s) once. Same Julia, no second environment.
- First ever run: loading tesseract_api 68.7 s; second run: 7.0 s.
- Every new process, even with warm caches: first apply 12.0 s, first VJP 3.2 s (Julia compiles the model code
  on first call); warm apply 23 ms, warm VJP 69 ms. So a fresh process needs about 22 s before its first
  gradient, and in-process results match pure Julia (log10 ratio -1.302).
- Of the ~12 s first apply in a fresh process, about 3 s is Julia compiling the model (template 1.8 s + first
  simulate 1.5 s in a fresh Julia; 3.3 s in results/local-2026-11-18/raw-spike.log) and the rest is juliacall/PythonCall compiling the
  conversions; the first VJP (~3.5 s) is mostly the model's ForwardDiff compile (first Jacobian 3.1 s).

## Julia-side breakage met on the way (not Tesseract's, but a Julia user wrapping a model will meet it)

- (Withdrawn) A one-parameter ForwardDiff crash ("type Dual has field offset -2147483648") was seen only while a
  PrecompileTools workload was in place; it does not reproduce without it (checked again after removing the
  workaround: 1 and 4 parameters fine).
- A PrecompileTools `@compile_workload` (to cut first-call latency) made the four-parameter Jacobian crash in
  `ijl_egal__bitstag` or spin for 10+ minutes, where it takes 68 ms without. It saved 2.5 s of an 11 s first call.
  Removed. So the cold-start cost per process stays: ~6.5 s load + 11 s first apply + 3 s first VJP.

## from_source (step 3)

- `Tesseract.from_source("tesseract_api.py")` passes this process's environment on, so the two PYTHON_JULIAPKG_*
  variables are enough; nothing in the Tesseract config can carry them (a container build will need them in the
  image instead).
- Cold (no .tesseract-venv yet): healthy after 31.1 s, i.e. at the 30 s default startup_timeout (it passed, just).
  Warm caches: healthy after 8.6 s.
- First call in each new server process: apply 12.9-13.2 s, VJP 3.4-3.5 s; warm apply 16-46 ms, warm VJP 73-76 ms.
  A client with a short HTTP timeout would fail the first call. Moving the warm-up into tesseract_api's import
  would push startup past 30 s. Neither the startup_timeout nor the first-call cost is visible from the docs for a
  compiled-language Tesseract.
- Threads: 8 concurrent applies from 4 client threads against the served Tesseract all returned the same, correct
  result (0.12 s). juliacall from the server's request handling worked here.

## Chain + calibration with tesseract-jax (step 4)

Chain: allometry Tesseract (JAX, in-process; mouse SLtg / KdegPD1 -> man, exponent differentiable) -> Lindauer
Tesseract (Julia, from_source) at 5 and 0.5 mg/kg -> log10 V(26 wk)/V(0) -> squared error against the paper's
-1.29 / -0.39. Unknowns: log V_blood, allometric exponent. Start: as published (1400 uL, -0.25): -5.994 / -4.733.
- Chain gradient (jax.value_and_grad through both Tesseracts and the glue) against central FD:
  (-10.458701, 300.790373) vs (-10.458719, 300.790361).
- First value_and_grad 16.0 s (Julia first-call compile in the served process); warm 0.21 s per evaluation
  (two 26-week stiff solves with forward-mode derivatives, two HTTP round trips, the JAX side).
- L-BFGS-B with bounds: 17 evaluations, 2.4 s, loss 5.6e-15, log ratios -1.29000 / -0.39000. Result V_blood
  4.49 L (QSPAtlas's discrete fix: 5 L, the physiological value), exponent -0.094 (allometry: -0.25). Two targets
  are met by more than one parameter set (QSPAtlas changed the growth scenario instead of the exponent), so this
  shows the mechanics and the cost, not a unique answer.
- Without bounds the first step threw the exponent to -1.25 (it is 30x more sensitive than log V_blood), and a
  later step reached parameters where Rodas5P hit MaxIters. The Julia error came back as HTTP 500 and then as a JAX
  `CpuCallback` error that ended the optimisation: a pipeline has no way to say "this point failed, step back".
  Workaround: the Tesseract returns NaN on a failed solve (QSPAtlas's own convention) and the optimiser treats a
  non-finite loss as a rejected step.

## check_gradients (tesseract-jax#300) on the chain (step 5)

Point: log V_blood = ln 3e6, exponent -0.15. One injected fault per run, each a plausible bug:
`lindauer_vjp` (Julia Tesseract's d/dSLtg off by the day/hour factor 24), `allometry_vjp` (d/d exponent missing
ln(bw_ratio)), `glue` (SLtg detached with stop_gradient between the Tesseracts; value unchanged).

As in PR #300 (steps scaled by each input's RMS):
- pipeline row right every time (ok clean; FAIL for each fault); allometry row right (FAIL only for its fault);
- both Lindauer rows FAIL in every run, clean included, with a NaN finite difference: `theta` mixes scales
  (V_blood ~3e6 uL, SLtg ~6e-6, KdegPD1 ~6e-3), so an RMS-scaled step (~3e5) throws SLtg out of range and the
  solve fails. The verdict then blames the Julia Tesseract for the clean run and for the glue fault.
  => As written, #300 does NOT localise better than per-Tesseract checks on a real model.

Change: step each element by its own magnitude (RMS only for zero entries). Then (4.5-4.9 s per check):
- clean: all rows ok, "Gradients agree."
- lindauer_vjp: Lindauer rows FAIL, allometry ok -> "Start there" at the Julia Tesseract;
- allometry_vjp: allometry FAIL, Lindauer ok -> points at the allometry Tesseract;
- glue: every Tesseract ok, pipeline FAIL -> "The problem is in the JAX code between the Tesseracts: look for
  jax.lax.stop_gradient ..." -- the injected bug.
Per-Tesseract checks alone cannot find the glue fault; the pipeline check alone cannot say where. With the fix,
#300 does both. The 48 existing tests still pass; the fix needs a mixed-scale test and docstring updates before
it goes to #300 (local commit only on the PR branch).

## Container (step 6, `tesseract build` + `Tesseract.from_image`, colima on macOS, arm64)

- Build: 712 s (11.9 min), image 4.77 GB (Julia 1.10.12 + the precompiled environment + Python runtime). Julia
  installed and the environment precompiled through `custom_build_steps`; the image runs as a random UID, so the
  precompiled depot (built as root) is read-only behind a writable depot in the user's home
  (JULIA_DEPOT_PATH="/home/tesseract-user/.julia:/opt/julia-depot:"), juliapkg offline, both via ENV lines.
- `Tesseract.from_image("lindauer-julia:0.1.0")` with defaults: healthy after 9-11 s, then every call fails with
  "PermissionError: /tesseract/output_data/run_<id>" (HTTP 500). The SDK mounts a `tempfile.mkdtemp()` folder as
  output_data; on macOS that is under /var/folders, which colima does not share with its VM by default, so the
  mount is an empty root-owned directory. Same with `user="9999:9999"` or an explicit output_path under /tmp. Works
  with output_path under $HOME. Not Julia-specific: any Tesseract image on a stock colima setup will hit it, and
  the error does not point at the cause.
- With output_path under $HOME: healthy after 13.1 s; first apply 23.0 s, first VJP 5.7 s; warm 29 ms / 116 ms;
  result correct (-1.302). The prebuilt depot is used (no recompilation), but the first call is slower than
  from_source (13 s).

## The Julia template on tesseract-core branch jpb/julia (started 2026-04, reworked 2026-09-28): Enzyme on this model

Found after the first draft. The template's runtime recipes (julia_recipes.jl)
differentiate the user's `apply_jl` with Enzyme: `autodiff(set_runtime_activity(Reverse), Const(f), Active,
Duplicated(diff_args, shadows))` for the VJP and the Forward analogue for the JVP. julia/scripts/enzyme_probe.jl (commit ff3a387)
makes exactly those calls on the Lindauer solve (Julia 1.10.12; Enzyme 0.13.221, OrdinaryDiffEq 7.8.1,
OrdinaryDiffEqRosenbrock 2.7.5, LinearSolve 5.18.3, ModelingToolkit 11.46.1, SciMLBase 3.58.0):
- With the model built lazily inside the differentiated function: both modes fail in 50-60 s with
  IllegalTypeAnalysisException inside `build_lindauer` (Enzyme walks into the symbolic MTK build). Lesson for
  template users: build the model at include time and pass it in.
- With the prebuilt template passed in: reverse fails after 224.6 s of compilation with MixedReturnException
  ("Custom Rule for method returns type SciMLBase.LinearProblem{...}": the linear solve inside Rodas5P's steps);
  forward mode was still compiling inside libEnzyme's type analysis when the 1500 s `timeout` stopped it
  (signal 15, about 21 minutes into forward mode; earlier I misread the stack as a crash).
- ForwardDiff through the same solve agrees with finite differences to 1e-6..1e-9 (step 1).
(Superseded by the next bullet: with SciMLSensitivity loaded, reverse mode does work.) Ask: a way for `apply_jl` authors to supply their own VJP/JVP (or choose the AD).
- The template's PYTHON_JULIACALL_EXE / PYTHON_JULIACALL_PROJECT pair works here and skips juliapkg entirely
  (load 7.3 s, first apply 12.5 s, as with the JULIAPKG pair); my JULIAPKG route was a detour. The template also
  sets JULIA_PKG_PRECOMPILE_AUTO=0 and a shared depot.
- Correction: with `using SciMLSensitivity`, Enzyme reverse mode works through the solve
  (DiffEqBase's Enzyme rule hands it to SciMLSensitivity's adjoint). enzyme_sens_timing.jl: first call 494.9 s,
  second 1.65 s, at a new point 1.62 s; VJP agrees with ForwardDiff to 6.4e-10 (results/local-2026-11-18/raw-enzyme-sens.log, load ~11-16).
  The MixedReturnException above is what happens without SciMLSensitivity loaded.
