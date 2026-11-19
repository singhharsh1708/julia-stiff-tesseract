# julia --project=tools tools/test.jl
using Test, StiffTesseract, ModelingToolkit
using OrdinaryDiffEq: ODEProblem, solve, Rodas5P
using SymbolicIndexingInterface: variable_index

const ST = StiffTesseract
const TS = collect(0.0:7.0:182.0)

# The same dose-by-restart solve, on the ModelingToolkit system itself (what generate.jl starts from).
function simulate_mtk(scenario, V_blood, mgkg)
    sys = ST_MTK.build_lindauer(guard = true)
    raw = ST_MTK.human_raw(; scenario, V_blood)
    prob = ODEProblem(sys, ST_MTK.problem_pairs(sys, raw), (0.0, 1.0); jac = true, build_initializeprob = false)
    ia1, itv = variable_index(prob, sys.A1), variable_index(prob, sys.TV)
    u, out = copy(prob.u0), Float64[]
    doses = collect(0.0:(21.0 * 24):(182.0 * 24 - 1e-9))
    for (k, t0) in enumerate(doses)
        t1 = k < length(doses) ? doses[k + 1] : 182.0 * 24
        u[ia1] += ST.dose_nmol(mgkg)
        seg = [t for t in TS .* 24 if t0 <= t < t1 || (k == length(doses) && t == t1)]
        sol = solve(ODEProblem(prob.f, u, (t0, t1), prob.p), Rodas5P(); ST.SOLVE_KW..., saveat = vcat(seg, t1), dtmax = 24.0)
        append!(out, [s[itv] for s in sol.u[1:length(seg)]])
        u = copy(sol.u[end])
    end
    return out
end

module ST_MTK
using ModelingToolkit
include(joinpath(@__DIR__, "..", "lindauer", "julia", "src", "lindauer_model.jl"))
end

@testset "generated code matches the ModelingToolkit model" begin
    for (scenario, vb) in (("fast_allom", 1400.0), ("med_growth", 5.0e6)), mgkg in (5.0, 0.5)
        gen = ST.simulate([vb], ["V_blood"], scenario, mgkg, 21.0, TS)[1]
        ref = simulate_mtk(scenario, vb, mgkg)
        @test maximum(abs.(gen .- ref) ./ abs.(ref)) < 1e-8
    end
end

@testset "QSPAtlas Lindauer numbers" begin
    lr(s, vb, m) = log10(ST.simulate([vb], ["V_blood"], s, m, 21.0, [182.0])[1][end] / 41500)
    @test lr("fast_allom", 1400.0, 5.0) ≈ -6.0055 atol = 1e-3   # QSPAtlas records -6.0 / -4.7 as published
    @test lr("fast_allom", 1400.0, 0.5) ≈ -4.7461 atol = 1e-3
    @test lr("med_growth", 5.0e6, 5.0) ≈ -1.301768 atol = 1e-5   # QSPAtlas notes.md: -1.301768 with its fix
    @test lr("med_growth", 5.0e6, 0.5) ≈ -0.400621 atol = 1e-5
    # Fig 4 gamma x0.2 bar, where RO^gamma is not smooth at 0: QSPAtlas (callback dosing, no guard) gives
    # 0.28414675580611276 with V_blood 5 L and medium growth.
    g = log10(ST.simulate([5.0e6, 2.28 * 0.2], ["V_blood", "gamma"], "med_growth", 5.0, 21.0, [182.0])[1][end] / 41500)
    @test g ≈ 0.28414675580611276 atol = 1e-8
end

@testset "ForwardDiff Jacobian against finite differences" begin
    fields = ["V_blood", "L0", "SLtg", "KdegPD1"]
    theta = [5.0e6, 0.0036, 6.247e-7, 0.00246]
    ts = [0.0, 91.0, 182.0]
    J = ST.jacobian(theta, fields, "med_growth", 5.0, 21.0, ts)
    for k in eachindex(theta)
        h = 1e-6
        up, dn = copy(theta), copy(theta)
        up[k] *= exp(h); dn[k] *= exp(-h)
        fd = (vcat(ST.simulate(up, fields, "med_growth", 5.0, 21.0, ts)...) .-
              vcat(ST.simulate(dn, fields, "med_growth", 5.0, 21.0, ts)...)) ./ (2h) ./ theta[k]
        @test maximum(abs.(J[:, k] .- fd)) / maximum(abs.(fd)) < 1e-5
    end
end
