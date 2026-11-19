# CI run 44504779436 (2026-11-19), ubuntu-latest, Julia 1.10.12, commit b0a4b4c

The generated (ModelingToolkit-free) model. Two parts of this run are not valid and are not reproduced below:
the Figure 4 calibration and the gradcheck chain went through `Tesseract.from_source`, which failed on every call
because the config's `env:` held image-only paths (docs/log.md), and both jobs reported success. Figure 4 was
rerun locally (results/local-2026-11-19); both run again in CI on the fixed commit. The gradcheck toy grid (a JAX
Tesseract, valid here) is left out too; the same grid is in results/local-2026-11-19. Two findings from this run:

- GaussAdjoint + `ReverseDiffVJP(true)` is wrong at 8 and 16 parameters (max rel. err 1): the compiled tape keeps
  the branch of the guard `if` it was recorded on, and gives d/dgamma = 0 (local check: -10393 expected) and a
  wrong d/dSLtg. `ReverseDiffVJP()` (no compiled tape) is right to 4e-8; routes.jl now uses it. In CI run 1 the
  compiled tape was right because that model was built without the guard (`guard = false`), so had no branch.
- The jpb/julia template's Enzyme call (route `enzyme`) failed to compile on the generated model
  (IllegalTypeAnalysisException, a Union in `StiffTesseract._simulate`); on the ModelingToolkit model in CI run 1
  it worked. Tried locally, none of which helped: a function barrier so the solve loop sees concrete vectors (the
  error moves into the loop), `ODEFunction{true, FullSpecialize}` instead of the default AutoSpecialize (same
  error), and `Enzyme.API.strictAliasing!(false)` (fails instead with "No matching function wrapper was found").
  Not followed further; the template route's numbers are from CI run 1.

## VJP routes

| route | params | first call | warm (min of 2, or second call) | max rel. err vs ForwardDiff | maxiters warnings | other warnings | note |
|---|---|---|---|---|---|---|---|
| forwarddiff | 2 | 6.71 s | 0.0673 s | 0 | 0 | none |  |
| forwarddiff | 8 | 9.58 s | 0.168 s | 0 | 0 | none |  |
| forwarddiff | 16 | 10.1 s | 0.306 s | 0 | 0 | none |  |
| enzyme | 2 | - | - | - | - | - | IllegalTypeAnalysisException: Enzyme compilation failed due to illegal type analysis.  This usually  |
| enzyme | 8 | - | - | - | - | - | IllegalTypeAnalysisException: Enzyme compilation failed due to illegal type analysis.  This usually  |
| enzyme | 16 | - | - | - | - | - | IllegalTypeAnalysisException: Enzyme compilation failed due to illegal type analysis.  This usually  |
| adjoint-gauss-enzyme | 2 | 235 s | 144 s | 0.0195 | 5 | none |  |
| adjoint-gauss-enzyme | 8 | 243 s | 150 s | 0.0249 | 5 | none |  |
| adjoint-gauss-enzyme | 16 | 129 s | 83.5 s | 0.549 | 5 | none |  |
| adjoint-gauss-enzyme-fdjac | 2 | 65.4 s | 0.0764 s | 2.63e-08 | 0 | none |  |
| adjoint-gauss-enzyme-fdjac | 8 | 66.9 s | 0.0769 s | 3.56e-08 | 0 | none |  |
| adjoint-gauss-enzyme-fdjac | 16 | 93 s | 0.0966 s | 7.45e-08 | 0 | none |  |
| adjoint-gauss-reversediff | 2 | 45.3 s | 0.37 s | 4.64e-08 | 0 | none |  |
| adjoint-gauss-reversediff | 8 | 42.1 s | 0.376 s | 1 | 0 | none |  |
| adjoint-gauss-reversediff | 16 | 72.3 s | 0.681 s | 1 | 0 | none |  |
| adjoint-interp-enzyme | 2 | 602 s | 547 s | 0.00148 | 5 | none |  |
| adjoint-interp-enzyme | 8 | 844 s | 751 s | 0.00252 | 5 | none |  |
| adjoint-interp-enzyme | 16 | 934 s | 823 s | 0.201 | 5 | none |  |

## Where EnzymeVJP goes wrong (GaussAdjoint, 2 parameters, vs ForwardDiff on the same function)

| variant | EnzymeVJP first / second | EnzymeVJP max rel. err | maxiters | ReverseDiffVJP first / second | ReverseDiffVJP max rel. err | maxiters |
|---|---|---|---|---|---|---|
| restart | 134 s / 87.2 s | 0.0206 | 10 | 13.7 s / 0.516 s | 3.84e-08 | 0 |
| restart+fdjac | 80.4 s / 0.193 s | 2.63e-08 | 0 | 17.7 s / 0.884 s | 2.95e-08 | 0 |
| restart+kencarp4 | 53.4 s / 0.0968 s | 7.13e-06 | 0 | 16.4 s / 0.21 s | 6.53e-06 | 0 |
| single | 77.5 s / 15.6 s | 0.847 | 2 | 15.3 s / 0.278 s | 8.02e-08 | 0 |

ReverseDiffVJP as is: max rel. err 8e-08, 0 maxiters warnings; with the time derivative zeroed as EnzymeVJP has it: 0.84741232, 1 maxiters warnings

Component checks (max rel. error at t = 1, 24, 240 h): generated jacobian_in_lambda [0.076, 0.52, 0.038]; generated plain_vjp [1.7e-17, 1.7e-17, 5.8e-19]; generated time_derivative [1.0, 1.0, 1.0]; branchless jacobian_in_lambda [1.2e-17, 1.2e-17, 5.8e-18]; branchless plain_vjp [7e-16, 1.3e-16, 4.3e-18]; branchless time_derivative [1.0, 1.0, 1.0]

## First call through juliacall, with and without a system image

| run | load | first apply | warm apply | first VJP | warm VJP |
|---|---|---|---|---|---|
| sysimage-baseline-1 | 8.23 s | 7.82 s | 0.0224 s | 11.1 s | 0.148 s |
| sysimage-baseline-2 | 6.95 s | 7.87 s | 0.0223 s | 11.2 s | 0.146 s |
| sysimage-traced | 6.91 s | 7.8 s | 0.0223 s | 11 s | 0.146 s |
| sysimage-with-1 | 4.03 s | 0.0607 s | 0.0239 s | 0.149 s | 0.154 s |
| sysimage-with-2 | 3.39 s | 0.0597 s | 0.0243 s | 0.148 s | 0.154 s |

Image build: 764 s, 699 MiB, peak memory 13.5 GiB

## Container

- build_s 438, size_bytes 710250147
- image lindauer-julia:ci, healthy_s 12.9, first_apply_s 5.66, warm_apply_s 0.0216, log10_ratio -1.3, first_vjp_s 7.6, warm_vjp_s 0.115

