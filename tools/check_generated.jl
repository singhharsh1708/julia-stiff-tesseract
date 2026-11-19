# Is the committed generated code what tools/generate.jl produces now?
#
#   julia --project=tools tools/generate.jl <dir> && julia --project=tools tools/check_generated.jl <dir>
#
# ModelingToolkit orders the terms of a sum, and the parameters, by hash, which differs between platforms (the
# parameter order on Linux is not the one on macOS), so a regenerated file can differ from the committed one while
# computing the same thing. This compares what matters: the parameter file as text, the state and parameter names,
# and the right-hand side and Jacobian at random points, matched by name. The rest of the code reads both vectors
# by name (StiffTesseract.vectors), so the order itself does not matter.
using Random
import StiffTesseract   # for the NaNMath the generated code calls

fresh_dir = ARGS[1]
committed_dir = joinpath(@__DIR__, "..", "lindauer", "julia", "src", "generated")
module Committed end
module Fresh end
for (mod, dir) in ((Committed, committed_dir), (Fresh, fresh_dir))
    Core.eval(mod, :(const NaNMath = $(StiffTesseract.NaNMath)))
    Base.include(mod, joinpath(dir, "lindauer_rhs.jl"))
end

fail(msg) = (println("stale generated code: ", msg, "; run tools/generate.jl"); exit(1))
read(joinpath(committed_dir, "lindauer_params.jl"), String) == read(joinpath(fresh_dir, "lindauer_params.jl"), String) ||
    fail("lindauer_params.jl differs")
sort(Committed.UNKNOWNS) == sort(Fresh.UNKNOWNS) || fail("state names differ")
sort(Committed.PARAMETERS) == sort(Fresh.PARAMETERS) || fail("parameter names differ")
# position in the committed vectors of each entry of the fresh ones
su = [findfirst(==(k), Committed.UNKNOWNS) for k in Fresh.UNKNOWNS]
sp = [findfirst(==(k), Committed.PARAMETERS) for k in Fresh.PARAMETERS]

Random.seed!(1)
n, m = length(Committed.UNKNOWNS), length(Committed.PARAMETERS)
for _ in 1:50
    u, p, t = exp.(4 .* randn(n)), exp.(4 .* randn(m)), 100 * rand()   # in the committed order
    a, b = zeros(n), zeros(n)
    Committed.lindauer_rhs!(a, u, p, t); Fresh.lindauer_rhs!(b, u[su], p[sp], t)
    all(isfinite, a) || error("non-finite right-hand side at a test point: choose other points")
    isapprox(a[su], b; rtol = 1e-12) || fail("right-hand side differs at a random point")
    A, B = zeros(n, n), zeros(n, n)
    Committed.lindauer_jac!(A, u, p, t); Fresh.lindauer_jac!(B, u[su], p[sp], t)
    all(isfinite, A) || error("non-finite Jacobian at a test point: choose other points")
    isapprox(A[su, su], B; rtol = 1e-12) || fail("Jacobian differs at a random point")
end
println("generated code is current: same parameter file, names, right-hand side and Jacobian")
