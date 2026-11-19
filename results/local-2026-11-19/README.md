# Generated model, 2026-11-19, MacBook (Apple silicon), Julia 1.10.12

- `local-forwarddiff-2.json`, `local-adjoint-gauss-reversediff-2.json`: `experiments/routes.jl` at 2 parameters on
  the generated model. ForwardDiff: load 1.9 s, first VJP 3.2 s, warm 39 ms. GaussAdjoint + ReverseDiffVJP: first
  19.7 s, warm 0.58 s, 3e-8 from ForwardDiff. `local-adjoint-gauss-enzyme-fdjac-2.json`: GaussAdjoint + EnzymeVJP with
  finite-difference solver Jacobians, first 22.7 s, warm 35 ms, 3e-8.
- `gradcheck-toy.json`: `experiments/gradcheck_grid.py toy`. Per rule and case, `passed` is check_gradients'
  overall verdict (true = gradients agree); the case is the injected fault (`none`, the entry with a wrong VJP,
  or `nan` for a NaN at a perturbed point).
- `enzymevjp-*.json`: `experiments/enzymevjp_probe.jl` (GaussAdjoint, 2 parameters, one 21-day solve). With the
  solver's default ForwardDiff Jacobians, EnzymeVJP is 85 % off and ReverseDiffVJP right to 7e-8; without the
  analytic Jacobian the same; with `Rodas5P(autodiff = AutoFiniteDiff())` EnzymeVJP is right to 7e-8 and its second
  call takes 0.04 s (ReverseDiffVJP 0.4 s). So ModelingToolkit and the dose restarts are not the cause.
- `enzymevjp-mechanism.json`: `experiments/enzymevjp_mechanism.jl`. Enzyme's VJP of the rhs is exact on Float64;
  with ForwardDiff Duals in it (what the adjoint's stiff solver passes), the time derivative of the VJP comes back
  zero and the Jacobian in λ loses the non-integer-power terms (ForwardDiff's `^(::Dual, ::Real)` skips the
  partials when they are zero). Robertson (integer powers only) and a two-state toy with `u^1.5` did not show a
  gradient error above 5e-7, so the size of the effect depends on the problem.
- Not kept: GaussAdjoint with FBDF gave a near-zero gradient with every VJP backend, including ReverseDiffVJP and
  finite-difference Jacobians; a separate problem, not followed up.
