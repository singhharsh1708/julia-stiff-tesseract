# Reflection: what we did, what was weak, what we could use better

Written 2026-11-18 after the first full pass (steps 1-6 and the template probe), before sending anything.
The experiments it proposes are now in `experiments/` and run in CI; results are in `results/`.
The question is not "what broke" (findings.md has that) but "what would a better version of this work be".

## 1. The derivative strategy is the core design choice, and we made it by default

What we did: ForwardDiff through the whole stiff solve, Jacobian per call, VJP = J^T v. Chosen because it was the
first thing that worked and is cheap for 2-4 parameters.

What is weak:
- Its cost grows with the number of parameters. Every result we have is for 2-4 parameters, so we cannot say
  where it stops being the right choice, and the "Enzyme is 20x slower" line in the write-up is only true there.
- We compared three routes but not the one SciML recommends for this case: SciMLSensitivity adjoints driven by
  Zygote/ReverseDiff (`sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(true))` or `InterpolatingAdjoint`; the
  compiled tape, `true`, is wrong once the model has the guard `if`, see experiments/routes.jl), with
  no Enzyme over the user's code. The Enzyme route only worked because DiffEqBase hands the solve to that same
  machinery; we never measured the machinery without Enzyme's 8-minute compile in front of it.
- The VJP endpoint computes the full Jacobian and throws most of it away. Fine for 2 outputs x 4 parameters;
  wasteful for a long time course.

Better: measure all routes at 2, 8 and 16 parameters (the tornado figure has 16) for first-call and warm cost:
ForwardDiff, SciMLSensitivity adjoint without Enzyme, Enzyme via the template. That turns an anecdote into a
crossover chart, which is what a template author needs to choose a default. Cost: about an hour of runs.

## 2. The calibration does not show why gradients matter

What we did: 2 parameters, 2 targets, a fit that is not unique, and that lands somewhere other than QSPAtlas's fix.

What is weak: with 2 parameters a finite-difference gradient costs 4 extra solves, so nothing here needs
autodiff at all. The demo proves the plumbing, not the value.

Better: QSPAtlas already holds the paper's Fig 4 tornado targets: 64 bars plus 2 base values, from 16 parameters
perturbed 5x and 0.2x. That is a real, published, multi-output calibration. Fitting several parameters to those 66
numbers, in log space, is a problem where finite differences cost 2 x 16 solves per gradient and autodiff costs one
pass; it also gives an identifiable fit we can compare with the paper. It is the version of step 4 worth showing.
Cost: half a day (targets exist; the loss needs the tornado protocol, which QSPAtlas's protocol.jl already encodes).

## 3. First-call latency has a standard fix we did not try

What we did: measured it (13 s first apply, 3.5 s first VJP, 8 minutes for the Enzyme route), tried a
PrecompileTools workload once, saw a crash, removed it.

What is weak: we stopped at the first failure, and never separated how much is juliacall's conversion code versus
the model. The standard Julia answer for a long-running service is a custom system image (PackageCompiler) built
from a workload that goes through juliacall itself; juliacall can load one via `PYTHON_JULIACALL_SYSIMAGE`. For a
Tesseract image this is natural: build the sysimage in `custom_build_steps`. If it brings the first call from
13 s (or 8 minutes) to under a second, that is the single most useful recommendation for the Julia template.
Also worth one more careful try: the PrecompileTools workload without storing any runtime state, since the crash
may have come from our TEMPLATE Ref, not from PrecompileTools.
Cost: one to two hours (sysimage builds are slow but unattended).

## 4. We used little of what Tesseract and tesseract-jax offer

- Batching: each calibration step makes 4 HTTP calls because the two doses are separate Tesseract calls. Making
  dose an array input (or using tesseract-jax's `vmap_method`) would halve the calls and show the vmap feature on a
  real model.
- `jacobian` endpoint: implemented but unused by our chain; `jax.jacrev` would use it.
- `tesseract-runtime check-gradients` (core's per-Tesseract check) and the `test_cases` in tesseract_config: we
  never ran core's own checker, so our "per-Tesseract checks cannot see the glue bug" is argued, not shown side by
  side. Running it on both Tesseracts at the same point would make the #300 comparison concrete.
- `env:` in tesseract_config.yaml for the JULIACALL variables, instead of `ENV` lines in custom steps. (Tried
  2026-11-19: `from_source` applies `env:` too, so image paths there break local runs; see docs/log.md.)
- The template's base image (julia:1.11-bookworm) instead of downloading Julia ourselves; probably smaller image.
- Profiling (`runtime_config={"profiling": True}`): we timed from outside; the runtime can tell us how much of a
  warm call is HTTP and encoding versus Julia.

## 5. check_gradients (#300): the fix is right in spirit, untested in its weak spot

Elementwise steps fixed this model, but: entries near zero are barely probed; a signed parameter crossing zero gets
a jump in step size; and NaN at a perturbed point is reported as "uncertain" instead of "apply returned NaN".
Better options to weigh before touching #300: (a) elementwise with a floor of c x RMS; (b) let the user pass a
per-input scale; (c) recommend log parameters in the Tesseract schema, which also helps the optimiser. A small test
grid (near-zero entry with a wrong derivative; signed entry; NaN at the perturbed point) decides between them.

## 6. Model and solver choices worth questioning

- abstol 1e-14 comes from QSPAtlas's reproduction runs. For calibration, 1e-10 is probably enough and faster;
  measure the change in the endpoint and in the gradient before relaxing.
- Dose by restart was chosen for AD. A `PresetTimeCallback` is the idiomatic MTK/DiffEq way and is supported by
  SciMLSensitivity adjoints; worth knowing whether it would have avoided the u0 traps (it would not need restarts).
- We expose tumour volume and occupancy only; the paper's other observables (plasma concentration) would make the
  calibration targets richer if data were available (it is not public; s007 stays out).

## 7. Process: what to do differently next time

- Look at the maintainers' branches and config schema before calling something a gap (cost us a draft).
- Capture raw logs (including server stderr) for every number at the time, not afterwards from memory.
- Record machine load with each timing; several numbers here were taken under load 10-19.
- Decide the comparison grid (parameters x routes) before running, instead of one point per route.

## Proposed next experiments, most useful first

1. Derivative routes at 2 / 8 / 16 parameters: ForwardDiff vs SciMLSensitivity adjoint (no Enzyme) vs template
   Enzyme; first-call and warm. (Answers: what should the template default to for DiffEq models.)
2. A system image through juliacall: first call before and after. (Answers: can first-call latency be removed.)
3. The tornado calibration (16 parameters, 66 targets, log space). (Answers: does the chain earn its keep.)
4. Core's per-Tesseract checker next to #300 at the same point, and the near-zero / signed / NaN test grid.
5. Small ones: dose batching, template base image, tolerance relaxation, profiling a warm call (`env:` in config was
   tried and reverted, see docs/log.md).
