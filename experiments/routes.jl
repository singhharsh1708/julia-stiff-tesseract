# VJP routes for the Lindauer solve, at 2, 8 and 16 parameters, on the generated (ModelingToolkit-free) model.
#
#   julia --project=tools experiments/routes.jl <route> <nparams> <out.json>
#
# forwarddiff               ForwardDiff Jacobian through the solve, VJP = J' v (what the Tesseract uses)
# enzyme                    the jpb/julia template's recipe call, autodiff(set_runtime_activity(Reverse), ...),
#                           with SciMLSensitivity loaded (DiffEqBase then hands the solve to its adjoint)
# adjoint-gauss-enzyme      Zygote + SciMLSensitivity GaussAdjoint(autojacvec = EnzymeVJP())
# adjoint-gauss-reversediff Zygote + GaussAdjoint(autojacvec = ReverseDiffVJP(true))
# adjoint-interp-enzyme     Zygote + InterpolatingAdjoint(autojacvec = EnzymeVJP())
# adjoint-gauss-enzyme-fdjac  GaussAdjoint(autojacvec = EnzymeVJP()) with Rodas5P(autodiff = AutoFiniteDiff()), so no
#                           ForwardDiff Duals reach Enzyme (see experiments/enzymevjp_mechanism.jl)
#
# One route and size per process, so the first call measures compilation. The ForwardDiff reference is
# computed after the timings, so it cannot warm up the route being timed.

# before the packages: code Enzyme compiles calls the logger in the world it was loaded in
include("maxiters.jl")
t_load = @elapsed begin
    using StiffTesseract, LinearAlgebra, ForwardDiff, JSON3
    const ST = StiffTesseract
end
route, nparams, outfile = ARGS[1], parse(Int, ARGS[2]), ARGS[3]

# The 16 one-at-a-time parameters of the paper's Fig 4 (QSPAtlas protocol.jl), in plotted order.
const FIG4 = ["gamma", "FR", "SLtg", "L0", "Tmulti", "N_PD1_TC", "V_blood", "N_Tcell", "EMAXTP", "V1",
    "KdegPD1", "CL", "PLQ", "Lfrac", "CLup", "V2"]
fields = nparams == 2 ? ["V_blood", "L0"] : FIG4[1:nparams]
const BASE = ST.human_raw(; scenario = "med_growth", V_blood = 5.0e6)
theta = [Float64(getfield(BASE, Symbol(f))) for f in fields]
ts = collect(0.0:14.0:182.0)
args = ST._args(fields, "med_growth", 5.0, 21.0, ts)
n = length(ts)
ct = vcat(zeros(n - 1), 1.0, zeros(n))          # cotangent: tumour volume at 26 weeks

if route == "forwarddiff"
    vjp() = ST.jacobian(theta, fields, "med_growth", 5.0, 21.0, ts)' * ct
elseif route == "enzyme"
    using Enzyme, SciMLSensitivity
    f(dargs) = dot(ct, vcat(ST._simulate(dargs[1], args...)...))
    function vjp()
        sh = [zeros(length(theta))]
        autodiff(set_runtime_activity(Reverse), Const(f), Active, Duplicated([copy(theta)], sh))
        return sh[1]
    end
elseif startswith(route, "adjoint")
    using Zygote, SciMLSensitivity
    using OrdinaryDiffEq: ODEProblem, solve, Rodas5P, AutoFiniteDiff
    const SENSEALG = Dict(
        "adjoint-gauss-enzyme" => GaussAdjoint(autojacvec = EnzymeVJP()),
        "adjoint-gauss-reversediff" => GaussAdjoint(autojacvec = ReverseDiffVJP(true)),
        "adjoint-interp-enzyme" => InterpolatingAdjoint(autojacvec = EnzymeVJP()),
        "adjoint-gauss-enzyme-fdjac" => GaussAdjoint(autojacvec = EnzymeVJP()),
    )[route]
    const ALG = route == "adjoint-gauss-enzyme-fdjac" ? Rodas5P(autodiff = AutoFiniteDiff()) : Rodas5P()
    const E_A1 = [i == ST.I_A1 ? 1.0 : 0.0 for i in eachindex(ST.UNKNOWNS)]
    # Non-mutating copy of StiffTesseract._simulate, as Zygote needs: no `u[i] += dose`, no filled arrays.
    function simulate_pure(th)
        p, u = ST.vectors(merge(BASE, NamedTuple{Tuple(Symbol.(fields))}(Tuple(th))))
        t_end = ST.T_END_DAYS * 24
        dts = collect(0.0:(21.0 * 24):(t_end - 1e-9))
        out_tv, out_ro = eltype(p)[], eltype(p)[]
        for (k, t0) in enumerate(dts)
            t1 = k < length(dts) ? dts[k + 1] : t_end
            u = u .+ ST.dose_nmol(5.0) .* E_A1
            inseg = [t for t in ts .* 24 if t0 <= t < t1 || (k == length(dts) && t == t1)]
            sol = solve(ODEProblem(ST.FUNC, u, (t0, t1), p), ALG; ST.SOLVE_KW..., dtmax = 24.0,
                saveat = vcat(inseg, t1), sensealg = SENSEALG)
            seg = sol.u[1:length(inseg)]
            out_tv = vcat(out_tv, [s[ST.I_TV] for s in seg])
            out_ro = vcat(out_ro, [100 * s[ST.I_PD1T] * ST.F_IS * s[ST.I_TV] / 1e6 / s[ST.I_MT] for s in seg])
            u = sol.u[end]
        end
        return vcat(out_tv, out_ro)
    end
    vjp() = Zygote.gradient(th -> dot(ct, simulate_pure(th)), theta)[1]
else
    error("unknown route $route")
end

result = Dict{String, Any}("route" => route, "nparams" => nparams, "load_s" => t_load, "julia" => string(VERSION),
    "precompile_s" => tryparse(Int, get(ENV, "precompile_s", "")))
try
    t1 = @elapsed (g1, n1) = count_maxiters(vjp)
    result["maxiters_warnings"] = n1
    if t1 < 300   # a route that takes minutes per call gets one call, so the job ends inside its time limit
        t2 = @elapsed vjp()
        t3 = minimum(@elapsed(vjp()) for _ in 1:2)
        merge!(result, Dict("second_s" => t2, "warm_min_s" => t3))
    end
    ref = ST.jacobian(theta, fields, "med_growth", 5.0, 21.0, ts)' * ct
    merge!(result, Dict("first_s" => t1, "ok" => true,
        "max_rel_err_vs_forwarddiff" => maximum(abs.(g1 .- ref) ./ max.(abs.(ref), 1e-300))))
catch e
    merge!(result, Dict("ok" => false, "error" => first(sprint(showerror, e), 2000)))
end
open(io -> JSON3.write(io, result; allow_inf = true), outfile, "w")
println(JSON3.write(result; allow_inf = true))
