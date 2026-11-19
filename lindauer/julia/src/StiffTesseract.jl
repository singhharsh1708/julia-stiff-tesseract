"""
Lindauer et al. 2017 (pembrolizumab PK / PD-1 occupancy / tumour growth) as a differentiable function, for a
Tesseract to call.

The model is QSPAtlas.jl's ModelingToolkit rebuild (`lindauer_model.jl`, unchanged), in its `guard = true` form
(the drug effect linear in receptor occupancy below 0.01 %, as in the paper's control stream). `tools/generate.jl`
turns it into plain Julia once (`generated/`), so this module loads OrdinaryDiffEq and ForwardDiff but not
ModelingToolkit; `tools/test.jl` checks the two agree and match QSPAtlas.

`simulate(theta, fields, scenario, mgkg, interval_days, ts)` returns tumour volume [mm^3] and tumour receptor
occupancy [%] at the times `ts` [days], where `theta` holds the values of the raw parameters named in `fields`
(the rest come from `human_raw(; scenario)`). Doses are Q`interval_days` boluses of `mgkg` mg/kg x 70 kg into
the central compartment, from day 0 to 26 weeks; the solve restarts at each dose with the dose added to A1, so
derivatives pass through plain solves.
"""
module StiffTesseract

using OrdinaryDiffEq: ODEFunction, ODEProblem, solve, Rodas5P, ReturnCode
using ForwardDiff
import NaNMath

include("generated/lindauer_params.jl")
include("generated/lindauer_rhs.jl")

const BW_KG = 70.0
const T_END_DAYS = 182.0
const SOLVE_KW = (abstol = 1e-14, reltol = 1e-9)   # as in QSPAtlas protocol.jl

const FUNC = ODEFunction{true}(lindauer_rhs!; jac = lindauer_jac!)
const I_A1, I_TV, I_PD1T, I_MT = (findfirst(==(k), UNKNOWNS) for k in (:A1, :TV, :PD1t, :Mt))

dose_nmol(mgkg) = mgkg * BW_KG * 1e6 / MW

"Parameter vector (in `PARAMETERS` order) and initial state (in `UNKNOWNS` order) for a raw parameter set."
function vectors(raw)
    d = derive(raw)
    return [getfield(d.p, k) for k in PARAMETERS], [getfield(d.u0, k) for k in UNKNOWNS]
end

# The public functions convert their arguments (juliacall passes PyArray and Py strings) to concrete types.
_args(fields, scenario, mgkg, interval_days, ts) =
    (String.(collect(fields)), String(scenario), Float64(mgkg), Float64(interval_days), Float64.(collect(ts)))

"Tumour volume and tumour receptor occupancy at `ts` [days] for raw parameters `theta`."
simulate(theta, fields, scenario, mgkg, interval_days, ts) =
    _simulate(Float64.(collect(theta)), _args(fields, scenario, mgkg, interval_days, ts)...)

function _simulate(theta::AbstractVector{T}, fields::Vector{String}, scenario::String, mgkg::Float64,
        interval_days::Float64, ts::Vector{Float64}) where {T <: Real}
    raw = merge(human_raw(; scenario), NamedTuple{Tuple(Symbol.(fields))}(Tuple(theta)))
    pv, uv = vectors(raw)
    p = Vector{T}(pv)
    u = Vector{T}(uv)
    dose = dose_nmol(mgkg)
    t_end = T_END_DAYS * 24
    dose_times = collect(0.0:(interval_days * 24):(t_end - 1e-9))
    ts_h = sort(ts .* 24)
    tv = Vector{T}(undef, length(ts_h))
    ro = Vector{T}(undef, length(ts_h))
    filled = 0
    for (k, t0) in enumerate(dose_times)
        t1 = k < length(dose_times) ? dose_times[k + 1] : t_end
        u[I_A1] += dose
        # sample times in [t0, t1); the final time point belongs to the last segment
        inseg = [t for t in ts_h if t0 <= t < t1 || (k == length(dose_times) && t == t1)]
        sol = solve(ODEProblem(FUNC, u, (t0, t1), p), Rodas5P(); SOLVE_KW..., saveat = vcat(inseg, t1), dtmax = 24.0)
        if sol.retcode != ReturnCode.Success
            # As in QSPAtlas: a failed solve is NaN, never an error, so a caller (an optimiser) can step back.
            @warn "solve failed between $(t0 / 24) and $(t1 / 24) days: $(sol.retcode)"
            fill!(tv, T(NaN)); fill!(ro, T(NaN))
            return tv, ro
        end
        for (j, _) in enumerate(inseg)
            s = sol.u[j]
            filled += 1
            tv[filled] = s[I_TV]
            # ROt = 100 * PD1t / CPD1t, CPD1t = Mt / Vis, Vis = F_IS * TV / 1e6 (lindauer_model.jl)
            ro[filled] = 100 * s[I_PD1T] * F_IS * s[I_TV] / 1e6 / s[I_MT]
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
    out = ForwardDiff.derivative(s -> vcat(_simulate(th .+ s .* dir, args...)...), 0.0)
    return out[1:n], out[(n + 1):end]
end

end
