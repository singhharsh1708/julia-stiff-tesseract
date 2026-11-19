# CI run 44514707281 (2026-11-19), ubuntu-latest, Julia 1.10.12, commit 0d58a17

The reference run for docs/findings.md: the generated model, with `from_source` working again (the Figure 4 and
gradcheck-chain jobs fail on a crashed run since this commit). The tables below are the run's summary
(`experiments/summarise.py`); the JSON files are the jobs' outputs. Notes:

- `enzyme` (the jpb/julia template recipe) fails to compile on the generated model; its numbers are from CI run 1.
- `adjoint-gauss-reversediff` uses `ReverseDiffVJP()` (no compiled tape), which is right at all sizes but slower
  warm than the compiled tape was in run 2 (wrong at 8 and 16 parameters there).

## VJP routes

| route | params | first call | warm (min of 2, or second call) | max rel. err vs ForwardDiff | maxiters warnings | other warnings | note |
|---|---|---|---|---|---|---|---|
| forwarddiff | 2 | 15.2 s | 0.108 s | 0 | 0 | none |  |
| forwarddiff | 8 | 15.6 s | 0.238 s | 0 | 0 | none |  |
| forwarddiff | 16 | 14.7 s | 0.407 s | 0 | 0 | none |  |
| enzyme | 2 | - | - | - | - | - | IllegalTypeAnalysisException: Enzyme compilation failed due to illegal type analysis.  This usually  |
| enzyme | 8 | - | - | - | - | - | IllegalTypeAnalysisException: Enzyme compilation failed due to illegal type analysis.  This usually  |
| enzyme | 16 | - | - | - | - | - | IllegalTypeAnalysisException: Enzyme compilation failed due to illegal type analysis.  This usually  |
| adjoint-gauss-enzyme | 2 | 194 s | 118 s | 0.0195 | 5 | none |  |
| adjoint-gauss-enzyme | 8 | 237 s | 145 s | 0.0249 | 5 | none |  |
| adjoint-gauss-enzyme | 16 | 155 s | 93.6 s | 0.549 | 5 | none |  |
| adjoint-gauss-enzyme-fdjac | 2 | 85.8 s | 0.0968 s | 2.63e-08 | 0 | none |  |
| adjoint-gauss-enzyme-fdjac | 8 | 69.2 s | 0.0756 s | 3.56e-08 | 0 | none |  |
| adjoint-gauss-enzyme-fdjac | 16 | 71.6 s | 0.0749 s | 7.45e-08 | 0 | none |  |
| adjoint-gauss-reversediff | 2 | 64.4 s | 2.06 s | 4.64e-08 | 0 | none |  |
| adjoint-gauss-reversediff | 8 | 39.2 s | 1.5 s | 5.87e-08 | 0 | none |  |
| adjoint-gauss-reversediff | 16 | 61.8 s | 2.62 s | 5.87e-08 | 0 | none |  |
| adjoint-interp-enzyme | 2 | 912 s | 836 s | 0.00148 | 5 | none |  |
| adjoint-interp-enzyme | 8 | 806 s | 741 s | 0.00252 | 5 | none |  |
| adjoint-interp-enzyme | 16 | 902 s | 821 s | 0.201 | 5 | none |  |

## Where EnzymeVJP goes wrong (GaussAdjoint, 2 parameters, vs ForwardDiff on the same function)

| variant | EnzymeVJP first / second | EnzymeVJP max rel. err | maxiters | ReverseDiffVJP first / second | ReverseDiffVJP max rel. err | maxiters |
|---|---|---|---|---|---|---|
| restart | 232 s / 146 s | 0.0206 | 10 | 26.9 s / 0.724 s | 3.84e-08 | 0 |
| restart+fdjac | 53.9 s / 0.129 s | 2.63e-08 | 0 | 11.6 s / 0.505 s | 2.95e-08 | 0 |
| restart+kencarp4 | 66.1 s / 0.101 s | 7.13e-06 | 0 | 20.7 s / 0.266 s | 6.53e-06 | 0 |
| single | 98.2 s / 18.3 s | 0.847 | 2 | 18.7 s / 0.342 s | 8.02e-08 | 0 |

ReverseDiffVJP as is: max rel. err 8e-08, 0 maxiters warnings; with the time derivative zeroed as EnzymeVJP has it: 0.84741232, 1 maxiters warnings

Component checks (max rel. error at t = 1, 24, 240 h): generated jacobian_in_lambda [0.076, 0.52, 0.038]; generated plain_vjp [1.7e-17, 1.7e-17, 5.8e-19]; generated time_derivative [1.0, 1.0, 1.0]; branchless jacobian_in_lambda [1.2e-17, 1.2e-17, 5.8e-18]; branchless plain_vjp [7e-16, 1.3e-16, 4.3e-18]; branchless time_derivative [1.0, 1.0, 1.0]

## First call through juliacall, with and without a system image

| run | load | first apply | warm apply | first VJP | warm VJP |
|---|---|---|---|---|---|
| sysimage-baseline-1 | 5.94 s | 5.74 s | 0.0167 s | 8.22 s | 0.112 s |
| sysimage-baseline-2 | 4.93 s | 5.61 s | 0.0174 s | 8.22 s | 0.112 s |
| sysimage-traced | 5.03 s | 5.7 s | 0.0164 s | 8.12 s | 0.114 s |
| sysimage-with-1 | 2.58 s | 0.0426 s | 0.0177 s | 0.115 s | 0.121 s |
| sysimage-with-2 | 2.42 s | 0.0427 s | 0.0176 s | 0.114 s | 0.121 s |

Image build: 384 s, 699 MiB, peak memory 13.5 GiB

## Container

- build_s 516, size_bytes 907281654
- image lindauer-julia:ci, healthy_s 10.4, first_apply_s 6.09, warm_apply_s 0.0202, log10_ratio -1.3, first_vjp_s 8.44, warm_vjp_s 0.116

## Figure 4 calibration

| point | rms, fitted bars | rms, all bars | rms, misfits | failed solves | values |
|---|---|---|---|---|---|
| as_published | 4.6516 | 4.680 | 5.08 | 3 (gamma x5 @ 5 mg/kg, SLtg x5 @ 5 mg/kg, gamma x5 @ 0.5 mg/kg) |  |
| start | 0.3494 | 0.672 | 1.94 | 0 |  |
| qspatlas_fix | 0.0192 | 0.575 | 1.90 | 0 |  |
| fit | 0.0053 | 0.563 | 1.87 | 0 | V_blood 5.114e+06, L0 0.003552, SLtg 6.354e-07, KdegPD1 0.002531 |

RMS in log10 units over the bars whose solve succeeded: 60 fitted, 66 in all, 6 misfits when none fail.

Fit: 34 gradient evaluations, 618 s; CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH

## check_gradients step rules

### gradcheck-chain

- rms: clean: flagged; lindauer_vjp: flagged; allometry_vjp: flagged; glue: flagged
- elementwise: clean: agree; lindauer_vjp: flagged; allometry_vjp: flagged; glue: flagged
- floor: clean: flagged; lindauer_vjp: flagged; allometry_vjp: flagged; glue: flagged

### gradcheck-toy

- rms: none: flagged; 0: flagged; 1: flagged; 2: flagged; 3: flagged; nan: flagged (16 NaN applies)
- elementwise: none: agree; 0: flagged; 1: flagged; 2: agree; 3: flagged; nan: agree (4 NaN applies)
- floor: none: flagged; 0: flagged; 1: flagged; 2: flagged; 3: flagged; nan: flagged (16 NaN applies)
- user_scale: none: agree; 0: flagged; 1: flagged; 2: flagged; 3: flagged; nan: agree (4 NaN applies)

