"""
Lindauer et al. 2017 (pembrolizumab PK / PD-1 occupancy / tumour growth) as a differentiable function, for
a Tesseract to call. The model is QSPAtlas.jl's ModelingToolkit rebuild, included unchanged.

`simulate(theta, fields, scenario, mgkg, interval_days, ts)` returns tumour volume [mm^3] and tumour receptor
occupancy [%] at the times `ts` [days], where `theta` holds the values of the raw parameters named in
`fields` (the rest come from `human_raw(; scenario)`). Doses are Q`interval_days` boluses of `mgkg` mg/kg
x 70 kg into the central compartment, from day 0 to 26 weeks. Instead of a dosing callback, the solve
restarts at each dose with the dose added to A1, so derivatives pass through plain solves.
"""
module StiffTesseract

using ModelingToolkit
using OrdinaryDiffEq: ODEProblem, solve, Rodas5P, ReturnCode
using SciMLStructures: Tunable, canonicalize
using SymbolicIndexingInterface: parameter_index, variable_index
using ForwardDiff

include("lindauer_model.jl")

const BW_KG = 70.0
const T_END_DAYS = 182.0
const SOLVE_KW = (abstol = 1e-14, reltol = 1e-9)   # as in QSPAtlas protocol.jl

dose_nmol(mgkg) = mgkg * BW_KG * 1e6 / MW

"Everything built once per process: the compiled system, a template problem and index maps."
struct Template
    sys::Any
    prob::Any
    repack::Any
    tunables::Vector{Float64}
    pidx::Dict{Symbol, Int}
    uidx::Dict{Symbol, Int}
end

const TEMPLATE = Ref{Union{Nothing, Template}}(nothing)

function template()
    TEMPLATE[] === nothing || return TEMPLATE[]
    sys = build_lindauer()
    raw = human_raw()
    # No initialization problem: with one attached, every solve rebuilds u0 from the initial values stored in
    # p (the template's A1 = 0), whatever u0 the problem holds, so the doses would vanish.
    prob = ODEProblem(sys, problem_pairs(sys, raw), (0.0, T_END_DAYS * 24); jac = true,
        build_initializeprob = false)
    tun, repack, _ = canonicalize(Tunable(), prob.p)
    d = derive(raw)
    pidx = Dict(k => only(parameter_index(prob, getproperty(sys, k)).idx) for k in keys(d.p))
    uidx = Dict(k => variable_index(prob, getproperty(sys, k)) for k in keys(d.u0))
    TEMPLATE[] = Template(sys, prob, repack, collect(tun), pidx, uidx)
    return TEMPLATE[]
end

# The public functions convert their arguments (juliacall passes PyArray, Py strings) to concrete types, so
# the precompiled methods below are the ones that run.
_args(fields, scenario, mgkg, interval_days, ts) =
    (String.(collect(fields)), String(scenario), Float64(mgkg), Float64(interval_days), Float64.(collect(ts)))

"Tumour volume and tumour receptor occupancy at `ts` [days] for raw parameters `theta`."
simulate(theta, fields, scenario, mgkg, interval_days, ts) =
    _simulate(Float64.(collect(theta)), _args(fields, scenario, mgkg, interval_days, ts)...)

_simulate(theta::AbstractVector{<:Real}, fields::Vector{String}, scenario::String, mgkg::Float64,
    interval_days::Float64, ts::Vector{Float64}) = _simulate(template(), theta, fields, scenario, mgkg, interval_days, ts)

# The template is an argument here, so code differentiated with Enzyme does not reach the lazy (symbolic)
# model build in `template()`.
function _simulate(tp::Template, theta::AbstractVector{T}, fields::Vector{String}, scenario::String, mgkg::Float64,
        interval_days::Float64, ts::Vector{Float64}) where {T <: Real}
    raw = merge(human_raw(; scenario), NamedTuple{Tuple(Symbol.(fields))}(Tuple(theta)))
    d = derive(raw)
    x = Vector{T}(tp.tunables)
    for (k, v) in pairs(d.p)
        x[tp.pidx[k]] = v
    end
    u = zeros(T, length(tp.uidx))
    for (k, v) in pairs(d.u0)
        u[tp.uidx[k]] = v
    end
    p = tp.repack(x)
    dose = dose_nmol(mgkg)
    t_end = T_END_DAYS * 24
    dose_times = collect(0.0:(interval_days * 24):(t_end - 1e-9))
    ts_h = sort(Float64.(ts) .* 24)
    i_a1, i_tv, i_pd1t, i_mt = tp.uidx[:A1], tp.uidx[:TV], tp.uidx[:PD1t], tp.uidx[:Mt]
    tv = Vector{T}(undef, length(ts_h))
    ro = Vector{T}(undef, length(ts_h))
    filled = 0
    for (k, t0) in enumerate(dose_times)
        t1 = k < length(dose_times) ? dose_times[k + 1] : t_end
        u[i_a1] += dose
        # sample times in [t0, t1); the final time point belongs to the last segment
        inseg = [t for t in ts_h if t0 <= t < t1 || (k == length(dose_times) && t == t1)]
        # Not remake: with an MTKParameters p, remake also rebuilds u0 from the initial values in p.
        prob = ODEProblem(tp.prob.f, u, (t0, t1), p)
        sol = solve(prob, Rodas5P(); SOLVE_KW..., saveat = vcat(inseg, t1), dtmax = 24.0)
        if sol.retcode != ReturnCode.Success
            # As in QSPAtlas: a failed solve is NaN, never an error, so a caller (an optimiser) can step back.
            @warn "solve failed between $(t0 / 24) and $(t1 / 24) days: $(sol.retcode)"
            fill!(tv, T(NaN)); fill!(ro, T(NaN))
            return tv, ro
        end
        for (j, t) in enumerate(inseg)
            s = sol.u[j]
            filled += 1
            tv[filled] = s[i_tv]
            # ROt = 100 * PD1t / CPD1t, CPD1t = Mt / Vis, Vis = F_IS * TV / 1e6 (model.jl)
            ro[filled] = 100 * s[i_pd1t] * F_IS * s[i_tv] / 1e6 / s[i_mt]
        end
        u = copy(sol.u[end])
    end
    filled == length(ts_h) || error("sample times outside [0, $(T_END_DAYS)] days")
    return tv, ro
end

"Jacobian of the stacked outputs [TV; RO] with respect to `theta`, by forward-mode AD through the solver."
function jacobian(theta, fields, scenario, mgkg, interval_days, ts)
    args = _args(fields, scenario, mgkg, interval_days, ts)
    return ForwardDiff.jacobian(x -> vcat(_simulate(x, args...)...), Float64.(collect(theta)))
end

"v^T J for cotangents of TV and RO."
function vjp(theta, fields, scenario, mgkg, interval_days, ts, ct_tv, ct_ro)
    J = jacobian(theta, fields, scenario, mgkg, interval_days, ts)
    return J' * vcat(Float64.(collect(ct_tv)), Float64.(collect(ct_ro)))
end

"J v for a tangent of theta."
function jvp(theta, fields, scenario, mgkg, interval_days, ts, tangent)
    n = length(ts)
    args = _args(fields, scenario, mgkg, interval_days, ts)
    th, dir = Float64.(collect(theta)), Float64.(collect(tangent))
    g(s) = vcat(_simulate(th .+ s .* dir, args...)...)
    out = ForwardDiff.derivative(g, 0.0)
    return out[1:n], out[(n + 1):end]
end

# No PrecompileTools workload: with one (Julia 1.10.12), the four-parameter Jacobian crashed in
# `ijl_egal__bitstag` or hung for minutes, while it took 72 ms without; and it saved only 2.5 s of the first
# call from Python, most of which is juliacall compiling its own conversions.

end
