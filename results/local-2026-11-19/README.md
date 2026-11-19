# Generated model, 2026-11-19, MacBook (Apple silicon), Julia 1.10.12

- `local-forwarddiff-2.json`, `local-adjoint-gauss-reversediff-2.json`: `experiments/routes.jl` at 2 parameters on
  the generated model. ForwardDiff: load 2.3 s, first VJP 3.2 s, warm 38 ms. GaussAdjoint + ReverseDiffVJP: first
  19.7 s, warm 0.58 s, 3e-8 from ForwardDiff. `local-adjoint-gauss-enzyme-fdjac-2.json`: GaussAdjoint + EnzymeVJP with
  finite-difference solver Jacobians, first 22.7 s, warm 35 ms, 3e-8.
- `gradcheck-toy.json`: `experiments/gradcheck_grid.py toy`. Per rule and case, `passed` is check_gradients'
  overall verdict (true = gradients agree); the case is the injected fault (`none`, the entry with a wrong VJP,
  or `nan`: apply returns NaN when x[1] moves more than 1 %, and `nan_applies` counts how often it did).
- `enzymevjp-*.json`: `experiments/enzymevjp_probe.jl` (GaussAdjoint, 2 parameters, one 21-day solve). With
  Rodas5P's default ForwardDiff Jacobians, EnzymeVJP is 85 % off and the adjoint solve stops at maxiters (one
  warning per call); ReverseDiffVJP is right to 7e-8. Without the analytic Jacobian: the same error (run before
  the maxiters count was added). With
  `Rodas5P(autodiff = AutoFiniteDiff())`: right to 7e-8, no maxiters, second call 0.05 s (ReverseDiffVJP 0.4 s).
  With KenCarp4 (ForwardDiff Jacobians, no time derivative): right to 1.3e-6, as ReverseDiffVJP (1.4e-6).
  ModelingToolkit and the dose restarts are not involved.
- `enzymevjp-mechanism.json`: `experiments/enzymevjp_mechanism.jl`. Enzyme's VJP of the rhs is exact on Float64.
  With ForwardDiff Duals in the state, as Rodas5P passes them for the adjoint's time derivative, the time
  derivative of the VJP comes back zero. With Duals in λ, the Jacobian in λ loses the non-integer-power terms
  (ForwardDiff's `^(::Dual, ::Real)` skips the partials when they are zero); GaussAdjoint does not take that path.
- `enzymevjp-zero-tderiv.json`: `experiments/enzymevjp_zero_tderiv.jl`. ReverseDiffVJP with only the time
  derivative zeroed gives EnzymeVJP's wrong gradient to 8 digits, with a maxiters warning (10^6 calls at a Dual
  time). So the zero time derivative stalls the Rosenbrock adjoint and the cut-short gradient is returned with a
  warning only. GaussAdjoint builds the adjoint Jacobian with ForwardDiff on the forward rhs (build_adjoint_jac),
  not through the VJP, so the λ-Jacobian error is not reached here. Robertson (integer powers) and a two-state
  toy with `u^1.5` showed no gradient error above 5e-7 (scratch runs, not kept; maxiters not checked).
- `fig4.json`: `experiments/fig4_calibration.py`, L0, SLtg and KdegPD1 (and V_blood, started at 5 L) fitted by
  gradient (L-BFGS-B through the served Tesseract and tesseract-jax) to the 60 Figure 4 numbers that are not
  QSPAtlas's known misfits. RMS over those 60 (log10 units): as published (a mouse blood volume in man) 4.65 over
  the 59 that solve (3 of the 66 solves fail); from the published values with a 5 L blood volume 0.349; after 34 gradient
  evaluations (283 s) 0.0053, against 0.0192 at QSPAtlas's hand-chosen fix. Fitted: V_blood 5.11 L, L0 0.00355/day,
  SLtg 6.35e-7, KdegPD1 0.00253 (QSPAtlas: 5 L, 0.0036, 6.247e-7, 0.00246). RMS over the six misfits stays 1.87.
- Not kept: GaussAdjoint with FBDF gave a near-zero gradient with every VJP backend, including ReverseDiffVJP and
  finite-difference Jacobians; a separate problem (possibly also a stalled adjoint), not followed up.
