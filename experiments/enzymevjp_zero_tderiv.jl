# Is the zero time derivative alone enough to give EnzymeVJP's wrong gradient?
#
#   julia --project=tools experiments/enzymevjp_zero_tderiv.jl [out.json]
#
# Take a backend that is right here (GaussAdjoint + ReverseDiffVJP) and inject only what EnzymeVJP gets wrong in
# the time derivative: the forward solution evaluated at a Dual time returns zero partials, so d/dt of the
# adjoint's right-hand side loses its dependence on y(t). Same 21-day solve as `enzymevjp_probe.jl single`.
# The method below replaces the solution's Dual-time call for the whole process, so this runs on its own.

# before the packages: code Enzyme compiles calls the logger in the world it was loaded in
include("maxiters.jl")
using StiffTesseract, ForwardDiff, Zygote, SciMLSensitivity, JSON3
using OrdinaryDiffEq: ODEProblem, solve, Rodas5P
const SciMLBase = SciMLSensitivity.SciMLBase
const ST = StiffTesseract

fields = ["V_blood", "L0"]
const BASE = ST.human_raw(; scenario = "med_growth", V_blood = 5.0e6)
theta = [Float64(getfield(BASE, Symbol(f))) for f in fields]
const E_A1 = [i == ST.I_A1 ? 1.0 : 0.0 for i in eachindex(ST.UNKNOWNS)]
function tv_end(th, sensealg)
    p, u = ST.vectors(merge(BASE, NamedTuple{Tuple(Symbol.(fields))}(Tuple(th))))
    p, u = eltype(th).(p), eltype(th).(u)
    u = u .+ ST.dose_nmol(5.0) .* E_A1
    sol = solve(ODEProblem(ST.FUNC, u, (0.0, 504.0), p), Rodas5P(); ST.SOLVE_KW..., dtmax = 24.0,
        saveat = [504.0], sensealg)
    return sol.u[end][ST.I_TV]
end
ref = ForwardDiff.gradient(th -> tv_end(th, nothing), theta)
relerr(g) = maximum(abs.(g .- ref) ./ abs.(ref))
sa = GaussAdjoint(autojacvec = ReverseDiffVJP(true))
result = Dict{String, Any}("forwarddiff" => ref)
g, n, _ = count_maxiters(() -> Zygote.gradient(th -> tv_end(th, sa), theta)[1])
result["reversediffvjp"] = Dict("gradient" => g, "max_rel_err" => relerr(g), "maxiters_warnings" => n)

const CALLS = Ref(0)
function (sol::SciMLBase.ODESolution)(t::ForwardDiff.Dual{T, V, N}; kw...) where {T, V, N}
    CALLS[] += 1
    return [ForwardDiff.Dual{T}(x, zero(ForwardDiff.Partials{N, V})) for x in sol(ForwardDiff.value(t); kw...)]
end
g, n, _ = count_maxiters(() -> Zygote.gradient(th -> tv_end(th, sa), theta)[1])
result["reversediffvjp_zero_tderiv"] = Dict("gradient" => g, "max_rel_err" => relerr(g), "maxiters_warnings" => n,
    "dual_time_calls" => CALLS[])
foreach(k -> println(k, " ", JSON3.write(result[k])), ("reversediffvjp", "reversediffvjp_zero_tderiv"))
length(ARGS) > 0 && open(io -> JSON3.write(io, result), ARGS[1], "w")
