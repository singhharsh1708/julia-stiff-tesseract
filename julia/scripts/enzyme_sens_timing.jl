# Enzyme reverse VJP with SciMLSensitivity loaded (DiffEqBase's Enzyme rule hands the solve to its adjoint):
# first call (compile) and second call in the same process.
using StiffTesseract, Enzyme, SciMLSensitivity, LinearAlgebra
const ST = StiffTesseract
const TP = ST.template()
const ARGS = ST._args(["V_blood", "L0"], "med_growth", 5.0, 21.0, [0.0, 91.0, 182.0])
const CT = [0.0, 0.0, 1.0]
f(d) = dot(CT, ST._simulate(TP, d[1], ARGS...)[1])
ref = ST.jacobian([5.0e6, 0.0036], ["V_blood", "L0"], "med_growth", 5.0, 21.0, [0.0, 91.0, 182.0])[1:3, :]' * CT
println("ForwardDiff: ", ref); flush(stdout)
for i in 1:2
    sh = [zeros(2)]
    t = @elapsed autodiff(set_runtime_activity(Reverse), Const(f), Active, Duplicated([[5.0e6, 0.0036]], sh))
    println("Enzyme call $i: ", round(t; digits = 2), " s  ", sh[1], "  max rel diff ", maximum(abs.(sh[1] .- ref) ./ abs.(ref))); flush(stdout)
end
sh = [zeros(2)]
t = @elapsed autodiff(set_runtime_activity(Reverse), Const(f), Active, Duplicated([[3.0e6, 0.0050]], sh))
println("Enzyme call 3 (new point): ", round(t; digits = 2), " s")
