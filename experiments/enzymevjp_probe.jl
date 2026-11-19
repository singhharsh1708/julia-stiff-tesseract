# Where does GaussAdjoint(autojacvec = EnzymeVJP()) go wrong on this model?
#
#   julia --project=tools experiments/enzymevjp_probe.jl <variant> <out.json>
#
# CI run 1 (ModelingToolkit model): every EnzymeVJP adjoint was off by 0.15-55 % against ForwardDiff and took
# 7-25 minutes per call, warm calls included; the same GaussAdjoint with ReverseDiffVJP was right to 1e-7 in
# under a second. This changes one thing at a time, on the generated model, 2 parameters, and counts the
# solver's maxiters warnings. experiments/enzymevjp_mechanism.jl has the cause.
#
# The variant is a `+`-joined list of changes from the setting of experiments/routes.jl (doses by restarting the
# solve over 26 weeks, analytic Jacobian, Rodas5P with ForwardDiff Jacobians); `restart` changes nothing.
#
# single   one dose and one solve over the first 21 days (no restarts)
# nojac    no analytic Jacobian in the ODEFunction
# fdjac    Rodas5P(autodiff = AutoFiniteDiff()): no dual numbers through the adjoint's VJP
# kencarp4 KenCarp4() (SDIRK): ForwardDiff Jacobians as before, but no time derivative
#
# Each variant is differentiated with GaussAdjoint + EnzymeVJP and, as a control, GaussAdjoint + ReverseDiffVJP;
# both are compared with ForwardDiff through the same function.

# before the packages: code Enzyme compiles calls the logger in the world it was loaded in
include("maxiters.jl")
using StiffTesseract, LinearAlgebra, ForwardDiff, JSON3, Zygote, SciMLSensitivity
using OrdinaryDiffEq: ODEFunction, ODEProblem, solve, Rodas5P, AutoFiniteDiff
using OrdinaryDiffEqSDIRK: KenCarp4
const ST = StiffTesseract
variant, outfile = ARGS
const CHANGES = split(variant, "+")
issubset(CHANGES, ["restart", "single", "nojac", "fdjac", "kencarp4"]) || error("unknown variant $variant")

fields = ["V_blood", "L0"]
const BASE = ST.human_raw(; scenario = "med_growth", V_blood = 5.0e6)
theta = [Float64(getfield(BASE, Symbol(f))) for f in fields]
const E_A1 = [i == ST.I_A1 ? 1.0 : 0.0 for i in eachindex(ST.UNKNOWNS)]
const FUNC = "nojac" in CHANGES ? ODEFunction{true}(ST.lindauer_rhs!) : ST.FUNC
const ALG = "kencarp4" in CHANGES ? KenCarp4() : "fdjac" in CHANGES ? Rodas5P(autodiff = AutoFiniteDiff()) : Rodas5P()
const T_END = "single" in CHANGES ? 21.0 * 24 : ST.T_END_DAYS * 24
const DOSE_TIMES = collect(0.0:(21.0 * 24):(T_END - 1e-9))

# Tumour volume at the end, non-mutating so Zygote can take it.
function tv_end(th, sensealg)
    p, u = ST.vectors(merge(BASE, NamedTuple{Tuple(Symbol.(fields))}(Tuple(th))))
    p, u = eltype(th).(p), eltype(th).(u)   # concrete under ForwardDiff (a Dual / Float64 mix widens to Real)
    for (k, t0) in enumerate(DOSE_TIMES)
        t1 = k < length(DOSE_TIMES) ? DOSE_TIMES[k + 1] : T_END
        u = u .+ ST.dose_nmol(5.0) .* E_A1
        sol = solve(ODEProblem(FUNC, u, (t0, t1), p), ALG; ST.SOLVE_KW..., dtmax = 24.0, saveat = [t1],
            sensealg = sensealg)
        u = sol.u[end]
    end
    return u[ST.I_TV]
end

result = Dict{String, Any}("variant" => variant, "julia" => string(VERSION))
ref = ForwardDiff.gradient(th -> tv_end(th, nothing), theta)
result["forwarddiff"] = ref
for (label, sensealg) in (("enzymevjp", GaussAdjoint(autojacvec = EnzymeVJP())),
        ("reversediffvjp", GaussAdjoint(autojacvec = ReverseDiffVJP(true))))
    r = Dict{String, Any}()
    try
        t1 = @elapsed (g, n1) = count_maxiters(() -> Zygote.gradient(th -> tv_end(th, sensealg), theta)[1])
        t2 = @elapsed (_, n2) = count_maxiters(() -> Zygote.gradient(th -> tv_end(th, sensealg), theta))
        merge!(r, Dict("first_s" => t1, "second_s" => t2, "gradient" => g, "ok" => true,
            "max_rel_err" => maximum(abs.(g .- ref) ./ abs.(ref)), "maxiters_warnings" => n1 + n2))
    catch e
        merge!(r, Dict("ok" => false, "error" => first(sprint(showerror, e), 2000)))
    end
    result[label] = r
    println(label, " ", JSON3.write(r)); flush(stdout)
end
open(io -> JSON3.write(io, result), outfile, "w")
