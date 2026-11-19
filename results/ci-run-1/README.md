# CI run 44317966027 (2026-11-18), ubuntu-latest, Julia 1.10.12

On the ModelingToolkit model (before `tools/generate.jl`); the route names and scripts are from commit ff3a387.

## Experiment 1: VJP routes

| route | params | first call | second | warm (min of 2) | max rel. err vs ForwardDiff | note |
|---|---|---|---|---|---|---|
| adjoint-default | 2 | 706 s | 589 s | 590 s | 0.0195  |  |
| adjoint-default | 8 | 779 s | 652 s | 653 s | 0.0249  |  |
| adjoint-default | 16 | 661 s | 558 s | 561 s | 0.549  |  |
| adjoint-gauss-enzyme | 2 | 755 s | 659 s | 653 s | 0.0195  |  |
| adjoint-gauss-enzyme | 8 | 753 s | 659 s | 657 s | 0.0249  |  |
| adjoint-gauss-enzyme | 16 | 465 s | 415 s | 418 s | 0.549  |  |
| adjoint-gauss-reversediff | 2 | 61.5 s | 0.829 s | 0.81 s | 4.58e-08  |  |
| adjoint-gauss-reversediff | 8 | 45 s | 0.558 s | 0.508 s | 5.81e-08  |  |
| adjoint-gauss-reversediff | 16 | 37.1 s | 0.442 s | 0.439 s | 5.49e-07  |  |
| adjoint-interp-enzyme | 2 | 1.37e+03 s | 1.27e+03 s | 1.28e+03 s | 0.00148  |  |
| adjoint-interp-enzyme | 8 | 1.73e+03 s | 1.55e+03 s | 1.56e+03 s | 0.00251  |  |
| adjoint-interp-enzyme | 16 | 1.5e+03 s | 1.39e+03 s | 1.4e+03 s | 0.201  |  |
| enzyme | 2 | 420 s | 1.57 s | 1.57 s | 7.07e-10  |  |
| enzyme | 8 | 474 s | 1.6 s | 1.6 s | 9.34e-10  |  |
| enzyme | 16 | 331 s | 1.35 s | 1.34 s | 2.67e-09  |  |
| forwarddiff | 2 | 7 s | 0.0745 s | 0.0749 s | 0  |  |
| forwarddiff | 8 | 15.4 s | 0.26 s | 0.258 s | 0  |  |
| forwarddiff | 16 | 9.32 s | 0.288 s | 0.286 s | 0  |  |
## Experiment 2: system image through juliacall
| run | load | first apply | warm apply | first VJP | warm VJP | log10 ratio |

System image job: baseline (3 runs, no image): load ~12 s, first apply ~31 s, warm apply 26-33 ms, first VJP ~12 s, warm VJP 150 ms (sysimage-baseline-runs.jsonl). The image build was killed (SIGTERM) after ~13 min in 'compiling incremental system image', 31 min into the job; likely memory on the 16 GB runner, unconfirmed.

## Reading (2026-11-18)

1. ForwardDiff is the right default for this model up to at least 16 parameters (fastest warm, cheapest first
   call). Its cost grows with parameters (0.075 -> 0.29 s); GaussAdjoint + ReverseDiffVJP stays ~0.4-0.8 s, so the
   crossover is roughly 25-30 parameters.
2. The jpb/julia template's Enzyme route is accurate (1e-9) but compiles 5-8 min per process, then 1.3-1.6 s.
3. Every SciMLSensitivity adjoint with EnzymeVJP (including the default) is wrong here (2 % to 55 %, growing with
   parameters) and takes 7-25 min per call; the same GaussAdjoint with ReverseDiffVJP is accurate to 1e-7. Needs a
   minimal reproducer (with and without ModelingToolkit) before reporting anywhere.
4. CI is slower than the Mac: fresh process 12 s load, 31 s first apply, 12 s first VJP.

## Next (agreed order, not started)

1. Rerun the sysimage job with swap and peak-memory recording; dedupe the summary (summary job re-uploads artifacts).
2. Minimal reproducer for the EnzymeVJP adjoint error.
3. Stage 2: Fig 4 tornado calibration (16 params) and the check_gradients step grid (#300).
4. Rewrite docs/findings.md from these results; only then the cover email to Dion and any push of the #300 fix.
