# Generated model, 2026-11-19, MacBook (Apple silicon), Julia 1.10.12

- `local-forwarddiff-2.json`, `local-adjoint-gauss-reversediff-2.json`: `experiments/routes.jl` at 2 parameters on
  the generated model. ForwardDiff: load 1.9 s, first VJP 3.2 s, warm 39 ms. GaussAdjoint + ReverseDiffVJP: first
  19.7 s, warm 0.58 s, 3e-8 from ForwardDiff.
- `gradcheck-toy.json`: `experiments/gradcheck_grid.py toy`. Per rule and case, `passed` is check_gradients'
  overall verdict (true = gradients agree); the case is the injected fault (`none`, the entry with a wrong VJP,
  or `nan` for a NaN at a perturbed point).
