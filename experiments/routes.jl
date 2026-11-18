# Experiment 1: three ways to get the VJP of the Lindauer solve, at 2, 8 and 16 parameters.
#
#   julia --project=julia experiments/routes.jl <route> <nparams> <out.json>
#
# route = forwarddiff : ForwardDiff Jacobian through the solve, VJP = J' * v (what the Tesseract uses)
#         adjoint-*   : Zygote + SciMLSensitivity through a non-mutating copy of the solve, with sensealg
#                       adjoint-default (GaussAdjoint(), backends chosen automatically),
#                       adjoint-gauss-enzyme, adjoint-gauss-reversediff, adjoint-interp-enzyme
#         enzyme      : the jpb/julia template's recipe call, autodiff(set_runtime_activity(Reverse), ...),
#                       with SciMLSensitivity loaded (DiffEqBase then hands the solve to its adjoint)
#
# One route and one size per process, so the first call measures compilation honestly.

t_load = @elapsed begin
    using StiffTesseract, LinearAlgebra, ForwardDiff, JSON3
    const ST = StiffTesseract
end
route, nparams, outfile = ARGS[1], parse(Int, ARGS[2]), ARGS[3]

# Parameter sets: the 16 one-at-a-time parameters of the paper's Fig 4 (QSPAtlas protocol.jl TORNADO_FIELD,
# minus EC50TP which has the smallest effect), taken in that order.
const ALL = ["gamma", "FR", "SLtg", "L0", "Tmulti", "N_PD1_TC", "V_blood", "N_Tcell", "EMAXTP", "V1",
    "KdegPD1", "CL", "PLQ", "Lfrac", "CLup", "V2"]
fields = nparams == 2 ? ["V_blood", "L0"] : ALL[1:nparams]
base = ST.human_raw(; scenario = "med_growth", V_blood = 5.0e6)
theta = [Float64(getfield(base, Symbol(f))) for f in fields]
ts = collect(0.0:14.0:182.0)
args = ST._args(fields, "med_growth", 5.0, 21.0, ts)
n = length(ts)
ct = vcat(zeros(n - 1), 1.0, zeros(n))          # cotangent: tumour volume at 26 weeks

const TP = ST.template()

if route == "forwarddiff"
    vjp() = ST.jacobian(theta, fields, "med_growth", 5.0, 21.0, ts)' * ct
elseif startswith(route, "adjoint")
    using Zygote, SciMLSensitivity
    using OrdinaryDiffEq: ODEProblem, solve, Rodas5P
    const SENSEALG = Dict(
        "adjoint-default" => GaussAdjoint(),
        "adjoint-gauss-enzyme" => GaussAdjoint(autojacvec = EnzymeVJP()),
        "adjoint-gauss-reversediff" => GaussAdjoint(autojacvec = ReverseDiffVJP(true)),
        "adjoint-interp-enzyme" => InterpolatingAdjoint(autojacvec = EnzymeVJP()),
    )[route]
    # Non-mutating copy of StiffTesseract._simulate, as Zygote needs: no `u[i] += dose`, no filling of output
    # arrays; the parameter and state vectors are built by permuting the values of `derive`'s named tuples.
    const D0 = ST.derive(base)
    const PPERM = [findfirst(==(k), collect(keys(D0.p))) for k in sort(collect(keys(TP.pidx)); by = k -> TP.pidx[k])]
    const UPERM = [findfirst(==(k), collect(keys(D0.u0))) for k in sort(collect(keys(TP.uidx)); by = k -> TP.uidx[k])]
    const E_A1 = [i == TP.uidx[:A1] ? 1.0 : 0.0 for i in 1:length(TP.uidx)]
    const I_TV, I_PD1T, I_MT = TP.uidx[:TV], TP.uidx[:PD1t], TP.uidx[:Mt]
    function simulate_pure(th)
        raw = merge(base, NamedTuple{Tuple(Symbol.(fields))}(Tuple(th)))
        d = ST.derive(raw)
        x = collect(values(d.p))[PPERM]
        u = collect(values(d.u0))[UPERM]
        p = TP.repack(x)
        dose = ST.dose_nmol(5.0)
        t_end = ST.T_END_DAYS * 24
        dts = collect(0.0:(21.0 * 24):(t_end - 1e-9))
        tsh = ts .* 24
        out_tv = eltype(x)[]
        out_ro = eltype(x)[]
        for (k, t0) in enumerate(dts)
            t1 = k < length(dts) ? dts[k + 1] : t_end
            u = u .+ dose .* E_A1
            inseg = [t for t in tsh if t0 <= t < t1 || (k == length(dts) && t == t1)]
            sol = solve(ODEProblem(TP.prob.f, u, (t0, t1), p), Rodas5P(); ST.SOLVE_KW..., dtmax = 24.0,
                saveat = vcat(inseg, t1), sensealg = SENSEALG)
            segu = sol.u[1:length(inseg)]
            out_tv = vcat(out_tv, [s[I_TV] for s in segu])
            out_ro = vcat(out_ro, [100 * s[I_PD1T] * ST.F_IS * s[I_TV] / 1e6 / s[I_MT] for s in segu])
            u = sol.u[end]
        end
        return vcat(out_tv, out_ro)
    end
    vjp() = Zygote.gradient(th -> dot(ct, simulate_pure(th)), theta)[1]
elseif route == "enzyme"
    using Enzyme, SciMLSensitivity
    f(dargs) = dot(ct, vcat(ST._simulate(TP, dargs[1], args...)...))
    function vjp()
        sh = [zeros(length(theta))]
        autodiff(set_runtime_activity(Reverse), Const(f), Active, Duplicated([copy(theta)], sh))
        return sh[1]
    end
else
    error("unknown route $route")
end

result = Dict{String, Any}("route" => route, "nparams" => nparams, "load_s" => t_load,
    "julia" => string(VERSION), "threads" => Threads.nthreads(),
    "precompile_s" => tryparse(Int, get(ENV, "precompile_s", "")))
try
    t1 = @elapsed g1 = vjp()
    t2 = @elapsed vjp()
    t3 = minimum(@elapsed(vjp()) for _ in 1:2)
    # The ForwardDiff reference is computed after the timings, so it cannot warm up the route being timed.
    ref = ST.jacobian(theta, fields, "med_growth", 5.0, 21.0, ts)' * ct
    result["first_s"] = t1
    result["second_s"] = t2
    result["warm_min_s"] = t3
    result["max_rel_err_vs_forwarddiff"] = maximum(abs.(g1 .- ref) ./ max.(abs.(ref), 1e-300))
    result["ok"] = true
catch e
    result["ok"] = false
    result["error"] = first(sprint(showerror, e), 2000)
end
open(outfile, "w") do io
    JSON3.write(io, result; allow_inf = true)
end
println(JSON3.write(result; allow_inf = true))
