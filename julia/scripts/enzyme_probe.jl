# Does Enzyme (as tesseract-core's julia_recipes call it) differentiate the Lindauer solve?
using StiffTesseract, Enzyme, LinearAlgebra
const ST = StiffTesseract
fields, theta, ts = ["V_blood", "L0"], [5.0e6, 0.0036], [0.0, 91.0, 182.0]
args = ST._args(fields, "med_growth", 5.0, 21.0, ts)
const TP = ST.template()
apply_jl(d) = (; tumour_volume = ST._simulate(TP, d[1], args...)[1])
ct = [0.0, 0.0, 1.0]
f(d) = dot(ct, apply_jl(d).tumour_volume)
J = ST.jacobian(theta, fields, "med_growth", 5.0, 21.0, ts)
println("ForwardDiff reference vjp: ", J[1:3, :]' * ct); flush(stdout)

d = [copy(theta)]; sh = [zeros(2)]
t = @elapsed try
    autodiff(set_runtime_activity(Reverse), Const(f), Active, Duplicated(d, sh))
    println("Enzyme reverse vjp:        ", sh[1])
catch e
    println("Enzyme reverse FAILED: ", first(sprint(showerror, e), 600))
end
println("reverse took ", round(t; digits = 1), " s"); flush(stdout)

d = [copy(theta)]; tang = [[1.0, 0.0]]
t = @elapsed try
    out = autodiff(set_runtime_activity(Forward), Const(apply_jl), Duplicated(d, tang))
    println("Enzyme forward jvp (dV_blood): ", out[1].tumour_volume, "  vs FD-AD ", J[1:3, 1])
catch e
    println("Enzyme forward FAILED: ", first(sprint(showerror, e), 600))
end
println("forward took ", round(t; digits = 1), " s")
