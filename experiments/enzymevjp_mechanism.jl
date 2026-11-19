# Why GaussAdjoint(autojacvec = EnzymeVJP()) with Rodas5P() is wrong on this model: the component checks.
#
#   julia --project=tools experiments/enzymevjp_mechanism.jl [out.json]
#
# With its default AutoForwardDiff, the stiff solver of the adjoint ODE builds the adjoint's Jacobian and time
# derivative with ForwardDiff, so SciMLSensitivity calls Enzyme's reverse mode on the user's rhs with Dual numbers
# in it. Reverse mode over Duals is not forward-over-reverse:
#
# 1. Time derivative: the Dual direction is in the state y(t). Reverse mode returns the gradient with respect to
#    y's partials, which is zero (the output's value does not depend on them), so d/dt (J' λ) comes back as 0.
#    This is what breaks the gradient here: the Rosenbrock adjoint solve then stops at maxiters, and
#    SciMLSensitivity returns the cut-short gradient with only a warning. enzymevjp_zero_tderiv.jl injects just this
#    into ReverseDiffVJP and gets EnzymeVJP's wrong gradient to 8 digits; KenCarp4 (no time derivative) and
#    finite-difference solver Jacobians both give the right one (enzymevjp_probe.jl).
# 2. Jacobian in λ: the Dual direction is in λ and y's partials are zero. This is right unless a Dual rule in the
#    rhs takes a shortcut on zero partials; ForwardDiff's `^(::Dual, ::Real)` (and NaNMath.pow) does, so the terms
#    with a non-integer power are lost. GaussAdjoint does not reach this: it builds the adjoint's Jacobian as
#    -(df/du)' with ForwardDiff on the forward rhs (SciMLSensitivity's build_adjoint_jac), not through the VJP.
#    An adjoint that differentiates the VJP itself (InterpolatingAdjoint does not call build_adjoint_jac) may
#    reach it; not measured.
#
# Each check compares with ForwardDiff on the plain rhs. `branchless` is the same rhs with NaNMath.pow(x, y)
# written as exp(y log x), which removes (2) but not (1).

using StiffTesseract, Enzyme, ForwardDiff, NaNMath, JSON3, Random
using ForwardDiff: Dual, value, partials
using OrdinaryDiffEq: ODEProblem, solve, Rodas5P
const ST = StiffTesseract
module Branchless
    using NaNMath
    branchless_pow(x, y) = x > 0 ? exp(y * log(x)) : zero(x)
    src = read(joinpath(pkgdir(Main.StiffTesseract), "src", "generated", "lindauer_rhs.jl"), String)
    include_string(@__MODULE__, replace(src, "(NaNMath.pow)" => "(branchless_pow)"))
end

Random.seed!(1)
p, u0 = ST.vectors(ST.human_raw(; scenario = "med_growth", V_blood = 5.0e6))
u0[ST.I_A1] += ST.dose_nmol(5.0)
sol = solve(ODEProblem(ST.FUNC, u0, (0.0, 504.0), p), Rodas5P(); ST.SOLVE_KW..., dtmax = 24.0)
n = length(u0)
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
result = Dict{String, Any}()

for (label, rhs!) in (("generated", ST.lindauer_rhs!), ("branchless", Branchless.lindauer_rhs!))
    f = (du, u, p, t) -> (rhs!(du, u, p, t); nothing)
    jac(u, t) = ForwardDiff.jacobian(x -> (du = similar(x); rhs!(du, x, p, t); du), u)
    # Enzyme's VJP of the rhs; y and λ may hold Duals, as in SciMLSensitivity's EnzymeVJP stage calls
    function enzyme_vjp(y, λ, t)
        dy = zero(y)
        Enzyme.autodiff(Reverse, Const(f), Const, Duplicated(zero(y), copy(λ)), Duplicated(copy(y), dy), Const(p), Const(t))
        return dy
    end
    r = Dict{String, Any}("plain_vjp" => Float64[], "jacobian_in_lambda" => Float64[], "time_derivative" => Float64[])
    for t in (1.0, 24.0, 240.0)
        u = sol(t); λ = randn(n); J = jac(u, t)
        push!(r["plain_vjp"], relerr(enzyme_vjp(u, λ, t), J' * λ))
        # what Rodas5P computes for the adjoint rhs λ -> J'λ: its Jacobian in λ, all partials at once
        Jλ = ForwardDiff.jacobian(l -> enzyme_vjp(eltype(l).(u), l, t), λ)
        push!(r["jacobian_in_lambda"], relerr(Jλ, Matrix(J')))
        # and its time derivative along y' = f(y)
        ud = (du = similar(u); rhs!(du, u, p, t); du)
        truth = ForwardDiff.derivative(s -> jac(u .+ s .* ud, t)' * λ, 0.0)
        got = partials.(enzyme_vjp([Dual(u[k], ud[k]) for k in 1:n], Dual.(λ, 0.0), t), 1)
        push!(r["time_derivative"], relerr(got, truth))
    end
    result[label] = r
    println(label, ": max rel. error at t = 1, 24, 240 h")
    foreach(k -> println("  ", rpad(k, 20), r[k]), ("plain_vjp", "jacobian_in_lambda", "time_derivative"))
end

# the ForwardDiff shortcut on its own: d/dx of the partials of x^1.5, through Enzyme over a zero-partials Dual
g!(du, u) = (du[1] = u[1]^1.5; nothing)
dy = [Dual(0.0, 0.0)]
Enzyme.autodiff(Reverse, Const(g!), Const, Duplicated([Dual(0.0, 0.0)], [Dual(0.0, 1.0)]), Duplicated([Dual(1.7, 0.0)], dy))
result["pow_partial"] = Dict("enzyme_over_dual" => partials(dy[1], 1), "expected" => 1.5 * sqrt(1.7))
println("x^1.5 at 1.7: partial through Enzyme over Dual ", partials(dy[1], 1), ", expected ", 1.5 * sqrt(1.7))
length(ARGS) > 0 && open(io -> JSON3.write(io, result), ARGS[1], "w")
