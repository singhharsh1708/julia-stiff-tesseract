# A stiff Julia model behind a Tesseract

Harsh Singh, November 2026. Code, CI and raw results: https://github.com/singhharsh1708/julia-stiff-tesseract

The model is Lindauer et al. 2017 (pembrolizumab: two-compartment PK, FcRn recycling, PD-1 binding in blood and
tumour, occupancy-driven tumour growth inhibition) from my QSPAtlas.jl: 11 states, 27 parameters, stiff (binding
in seconds next to tumour growth over weeks), Rodas5P, a dose every three weeks for 26 weeks. It runs as a Julia
Tesseract through juliacall (in-process, `from_source` and the built image), with VJP and JVP by ForwardDiff
through the solver. In a chain, an allometric-scaling Tesseract (JAX) feeds it and JAX glue turns tumour volume
into the paper's endpoint, log10 V(26 wk)/V(0).

Every number below comes from a GitHub Actions run (ubuntu-latest, Julia 1.10.12) unless marked local, and every
experiment is a job in `.github/workflows/experiments.yml`. Runs: 44317966027 (ModelingToolkit model),
44504779436 and 44514707281 (generated model; the last is the reference run, saved in results/ci-run-3).

## 1. Calibration through the chain: the paper's Figure 4

Figure 4 of the paper is a one-at-a-time sensitivity analysis: log10 V(26 wk)/V(0) at 5 and 0.5 mg/kg with each of
16 parameters multiplied by 5 and by 0.2 (64 bars), plus two base values in the caption. The paper does not give
the human blood volume or the tumour growth scenario of its base case; QSPAtlas chose them by a discrete search
and matched 58 of the 64 bars. Here L0, SLtg and KdegPD1 (and the blood volume, started at 5 L) are fitted by
L-BFGS-B with `jax.value_and_grad` through the served Julia Tesseract to the 60 numbers that are not QSPAtlas's
known misfits (the digitised values are read from QSPAtlas at run time and not stored here).

| point | RMS over the 60 (log10) | failed solves |
|---|---|---|
| as published (a mouse blood volume, 1.4 mL, in man) | 4.65 (over the 59 that solve) | 3 of 66 |
| published values with a 5 L blood volume (start) | 0.349 | 0 |
| QSPAtlas's hand-chosen fix | 0.0192 | 0 |
| gradient fit, 34 evaluations (618 s in CI, 283 s on a laptop) | 0.0053 | 0 |

The fit lands on L0 0.00355/day, SLtg 6.35e-7, KdegPD1 0.00253 and 5.11 L, next to QSPAtlas's 0.0036, 6.247e-7,
0.00246 and 5 L, reproducibly (CI and local agree to ten digits). With four parameters the gradient is not cheaper
than finite differences here: one ForwardDiff VJP costs about 6.5 applies (112 vs 17 ms warm), so a value and
gradient costs about 7.5 sweeps against 5 for forward differences, and the fastest adjoint here (section 3) costs
4.4-5.7 applies per VJP, so the cost advantage needs more parameters. What the exact gradient should buy here is
accuracy without a finite-difference step to choose for parameters that span 1e-7 to 5e6 (a fit by finite
differences was not run). As published, three perturbed solves fail, and the optimiser needs the Tesseract to
return NaN for them rather than raise (section 6, item 3); in CI two of the 34 evaluations (line-search trial
points) returned NaN, which the script treats as a rejected step (job log of run 44514707281).

## 2. What a Julia Tesseract costs, and how to cut it

ModelingToolkit dominated load and first-call time, so `tools/generate.jl` writes the compiled right-hand side and
Jacobian out as plain Julia once (checked against ModelingToolkit and QSPAtlas in `tools/test.jl`, and against a
fresh regeneration in CI). Same workload through juliacall (4 parameters), fresh process each time, one CI run
each (run 2 was 30-40 % slower than run 3 on the same code):

| | load | first apply | first VJP | warm apply / VJP |
|---|---|---|---|---|
| ModelingToolkit model (run 1) | 12 s | 31 s | 12 s | 26-33 ms / 150 ms |
| generated model (run 3) | 5 s | 5.7 s | 8.2 s | 17 ms / 112 ms |
| generated model + custom system image | 2.5 s | 0.043 s | 0.115 s | 18 ms / 121 ms |

The system image (PackageCompiler, from statements traced while running the same workload through juliacall,
loaded with `PYTHON_JULIACALL_SYSIMAGE`) removes the first-call cost for requests like the traced one, measured
in-process; a request with another number of parameters compiles a new ForwardDiff chunk size, and I did not
measure it inside a built image. It takes 384-764 s to build, is 699 MiB, and needs 13.5 GiB of memory at peak
(the first CI attempt, without extra swap, was killed, probably for memory). For the template this suggests a
system image built in `custom_build_steps` from a juliacall workload covering the expected request shapes, with
two caveats: Docker's default memory on a laptop is likely too small for the build, and `create_sysimage` targets
the build machine's CPU unless `cpu_target` is set (the template sets `JULIA_CPU_TARGET` for portability).

The image (`tesseract build`) now takes 438-516 s and is 0.7-0.9 GB in CI (amd64); with ModelingToolkit it was
12 minutes and 4.77 GB in a local arm64 build, so not like for like. From the image, healthy after 10-13 s, first apply 5.7-6.1 s,
first VJP 7.6-8.4 s.

## 3. Which derivative route

VJP of tumour volume at 26 weeks with respect to 2, 8 and 16 of the Figure 4 parameters, one process per route and
size; error against ForwardDiff through the same solve.

| route | first call | warm call (2 / 8 / 16 parameters) | error |
|---|---|---|---|
| ForwardDiff through the solve (what the Tesseract uses) | 15 s | 0.11 / 0.24 / 0.41 s | reference |
| GaussAdjoint + EnzymeVJP, Rodas5P with finite-difference Jacobians | 69-86 s | 0.097 / 0.076 / 0.075 s | 3e-8 to 7e-8 |
| GaussAdjoint + ReverseDiffVJP (no compiled tape) | 39-64 s | 2.1 / 1.5 / 2.6 s | 5e-8 to 6e-8 |
| GaussAdjoint + EnzymeVJP, defaults | 155-237 s | 118 / 145 / 94 s | 2 % to 55 % wrong |
| InterpolatingAdjoint + EnzymeVJP, defaults | 806-912 s | 12-14 minutes | 0.15 % to 20 % wrong |
| jpb/julia template recipe, ModelingToolkit model (run 1) | 331-474 s | 1.6 / 1.6 / 1.3 s | 7e-10 to 3e-9 |

ForwardDiff is the right default up to 16 parameters because it is correct with no hidden setting and has the
shortest first call; on speed alone it loses to an adjoint with EnzymeVJP and finite-difference solver Jacobians
after about 170-400 warm calls at 16 parameters (run 3: (71.6 - 14.7) / (0.407 - 0.075) s; run 2: 396) and
330-630 at 8. At 2 parameters ForwardDiff was faster in run 2 (67 vs 76 ms) and the adjoint needs over 6000 calls
in run 3; the Figure 4 fit's calls have 4 or 5 parameters, where I did not measure it. That adjoint is flat in the
parameter count (0.075-0.097 s warm), but only with that non-default setting (section 4). InterpolatingAdjoint + EnzymeVJP is wrong
too, by a cause I did not isolate.

The template's Enzyme recipe was accurate on the ModelingToolkit model (with SciMLSensitivity loaded, which then
picks the sensitivity method for the solve inside; probably its forward sensitivities at this size, not logged).
On the generated model it fails to compile: Enzyme reports illegal type analysis inside my dose-loop function
(its message suggests a Union), also with the function's retcode check removed (a local run, script not kept), on
Julia 1.10; the template's image uses 1.11, which I did not try. Run 1's function had the same shape and compiled,
so the cause may be solve code inlined there now that the model is plainly typed; I did not isolate it. A function
barrier, `FullSpecialize` and `strictAliasing!(false)` did not help. For a stiff DiffEq model the
template would be safer with a pluggable derivative (for example `julia_vjp` accepting a Julia derivative function
or an AD backend, keeping the recipe's input flattening) and a note that DiffEq solves need SciMLSensitivity
loaded for reverse mode.

## 4. Two wrong-gradient traps on this model that raise no error

Both return a gradient with no error, which a Tesseract's VJP endpoint would pass on as correct. The first looks
unreported; the second is documented behaviour of an opt-in setting.

1. **GaussAdjoint + EnzymeVJP with a Rosenbrock solver.** Rodas5P takes the adjoint ODE's time derivative with
   ForwardDiff, so SciMLSensitivity runs Enzyme's reverse mode on the right-hand side with dual numbers in the
   state. Reverse mode over duals returns zero for the derivative along the dual direction, so d/dt of the
   adjoint's right-hand side is zero; the adjoint solve then stops at maxiters and SciMLSensitivity returns the
   cut-short gradient with only a warning (85 % wrong on a 21-day solve). Injecting only the zero time derivative
   into ReverseDiffVJP reproduces the wrong gradient to 8 digits; KenCarp4 (no time derivative) or
   `Rodas5P(autodiff = AutoFiniteDiff())` give the right one. Enzyme's VJP of the right-hand side itself is exact.
   ModelingToolkit and the dose restarts are not involved. The zero time derivative should be general (it is what
   Enzyme's reverse mode over duals gives), but it was measured on this model only, and whether it stalls the
   adjoint depends on the problem: Robertson and a two-state toy with a non-integer power showed no gradient error
   (local runs, not kept), so there is no reproducer yet that does not need this model. Scripts:
   `experiments/enzymevjp_*.jl`.
2. **`ReverseDiffVJP(true)` with a branch in the model.** The model's guard (drug effect linear in occupancy below
   0.01 %, as in the paper's control stream) is an `if` in the generated code. A compiled ReverseDiff tape keeps
   the branch it was recorded on, so d/dgamma came back as 0 (true value -10393, a local check) at 8 and 16
   parameters. SciMLSensitivity documents that `compile = true` requires a function without branching, and its
   automatic choice checks for branches; set by hand, the failure is silent. `ReverseDiffVJP()` is right.

I found no SciMLSensitivity issue matching the first. For Tesseracts the lesson is that a wrong gradient can come
from settings deep inside the user's code, which is the case for a pipeline-level gradient check.

## 5. check_gradients (tesseract-jax#300) on the real chain

Same chain, one plausible bug at a time: the Julia Tesseract's d/dSLtg off by the day/hour factor 24, the allometry
Tesseract's d/d exponent missing ln(body-weight ratio), and SLtg detached with `stop_gradient` in the glue.

| injected bug | #300 as submitted (steps scaled by each input's RMS) | elementwise relative steps |
|---|---|---|
| none | blames the Julia Tesseract (its finite difference is NaN) | gradients agree |
| Julia VJP unit slip | blames the Julia Tesseract | blames the Julia Tesseract |
| allometry VJP slip | blames the allometry Tesseract | blames the allometry Tesseract |
| `stop_gradient` in the glue | blames the Julia Tesseract | blames the JAX code between the Tesseracts |

The parameter vector mixes a volume in uL (about 3e6) with rate constants (about 6e-6), so RMS-scaled steps throw
the small entries out of range, the solve fails, and the finite difference is NaN; #300 then reports it as "too
uncertain", which hides the cause. Elementwise steps fix the chain, but a toy Tesseract (`experiments/gradcheck_
toy`) shows their blind spot: a wrong derivative on an entry near zero (1e-12) that matters on an O(1) scale is
missed. Steps scaled by a typical size per entry flag every injected fault on the toy (one only at the pipeline
level, where the verdict cannot tell that Tesseract from the glue) and pass the clean case; there the scales
were given by hand for the pipeline input and the Tesseract input alike, and the toy's glue is the identity.
A direction for #300, not yet tested on the chain: take a per-entry scale for the pipeline's inputs from the user
(or default to the elementwise magnitude), and size each Tesseract's steps from how that scaled direction moves
the Tesseract's own inputs. `jax.jvp` through the glue would need a JVP endpoint on every upstream Tesseract;
recording the inputs each Tesseract receives at the pipeline's finite-difference steps (#300 records only the
base-point forward pass) would give the same scales to first order without one. Two
smaller points: a NaN from `apply` should be reported as such, and a NaN at the coarse steps only is currently
dropped silently (the check still passes). The fault injection uses hooks in both Tesseracts' VJPs that are off
unless `STIFF_FAULT` is set.

## 6. Integration follow-ups, most useful first

1. **`env:` in tesseract_config.yaml also applies to `from_source`, by design, with no per-call override.** Putting
   the juliacall settings there (image paths such as `/tesseract/julia`) broke every `from_source` run, while the
   image and `from_tesseract_api` worked; tesseract-core applies `env:` to the local server as an image would have
   it, over the caller's environment. They now sit in a build-step `ENV` line. An option for variables that apply
   only inside the image, or an `environment=` argument on `from_source`, would let the `env:` section hold them.
2. **`from_image` with colima's default mounts on macOS.** Every call failed with a `PermissionError` on
   `/tesseract/output_data/run_<id>`: the SDK mounts a `tempfile.mkdtemp()` folder under /var/folders, which colima
   does not share with its VM by default. An `output_path` under $HOME fixes it, but the error does not point there.
3. **A failed solve ended the optimisation.** An error raised in the Tesseract came back as HTTP 500 and then as a
   JAX `CpuCallback` error that ended the run. Returning NaN (QSPAtlas's convention for a failed solve) passes cleanly
   through `from_source` and tesseract-jax, and the caller can treat a non-finite loss as a rejected step; the
   Figure 4 fit relies on it. Documenting that convention would help.
4. **First-call latency** (section 2): the health check passes before Julia has compiled anything, so the first
   request is the slow one; a system image removed it in-process for the traced request, not yet tried inside a
   built image.

What worked without trouble: `from_source` passes the caller's environment through, concurrent requests to the
served Julia Tesseract returned the same result, and NaN round-trips cleanly. On the Julia side, ModelingToolkit 11
silently dropped a new `u0` when the solve was restarted at each dose (both `remake` and the attached initialization
data rebuild it from the parameters), which gave a tumour with no drug instead of an error; the generated model
avoids it.
