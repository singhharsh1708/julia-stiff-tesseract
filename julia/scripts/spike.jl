# Step 1: the model alone in Julia. Run with
#   julia --project=julia julia/scripts/spike.jl
t_load = @elapsed begin
    push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
    using StiffTesseract
    const ST = StiffTesseract
end
println("load (using StiffTesseract): ", round(t_load; digits = 1), " s")

log_ratio(tv) = log10(tv[end] / 41500.0)

# 1. Reproduce QSPAtlas's numbers (protocol.jl / notes.md): as published and with the V_blood fix.
cases = [
    ("as published: fast_allom, V_blood 1400 uL", "fast_allom", 1400.0, (-6.0, -4.7)),
    ("fix: med_growth, V_blood 5 L", "med_growth", 5.0e6, (-1.302, -0.401)),
]
ts = [182.0]
for (label, scen, vb, expect) in cases
    for (i, mgkg) in enumerate((5.0, 0.5))
        t = @elapsed tv, ro = ST.simulate([vb], ["V_blood"], scen, mgkg, 21.0, ts)
        println(rpad("$label, $mgkg mg/kg:", 52), round(log_ratio(tv); digits = 3),
            " (QSPAtlas ", expect[i], ")  RO(26 wk) ", round(ro[end]; digits = 2), " %   ", round(t; digits = 2), " s")
    end
end

# 2. Timing of a warm solve and of the forward-mode Jacobian over a few parameters.
fields = ["V_blood", "L0", "SLtg", "KdegPD1"]
theta = [5.0e6, 0.0036, 6.247e-7, 0.00246]
tsamp = collect(0.0:14.0:182.0)
t_warm = @elapsed ST.simulate(theta, fields, "med_growth", 5.0, 21.0, tsamp)
t_jac1 = @elapsed J = ST.jacobian(theta, fields, "med_growth", 5.0, 21.0, tsamp)
t_jac2 = @elapsed ST.jacobian(theta, fields, "med_growth", 5.0, 21.0, tsamp)
println("warm simulate: ", round(t_warm; digits = 3), " s; jacobian first ", round(t_jac1; digits = 2),
    " s, warm ", round(t_jac2; digits = 3), " s (", size(J, 1), " outputs x ", size(J, 2), " params)")

# 3. Jacobian against central finite differences in log parameter space.
function fd_column(k; h = 1e-6)
    up = copy(theta); up[k] *= exp(h)
    dn = copy(theta); dn[k] *= exp(-h)
    fu = vcat(ST.simulate(up, fields, "med_growth", 5.0, 21.0, tsamp)...)
    fd = vcat(ST.simulate(dn, fields, "med_growth", 5.0, 21.0, tsamp)...)
    return (fu .- fd) ./ (2h) ./ theta[k]     # d out / d theta_k
end
for k in eachindex(fields)
    fdk = fd_column(k)
    rel = maximum(abs.(J[:, k] .- fdk)) / max(maximum(abs.(fdk)), 1e-300)
    println("d/d", rpad(fields[k], 8), " max rel diff AD vs FD: ", round(rel; sigdigits = 3))
end
