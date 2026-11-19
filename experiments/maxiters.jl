# Count the solver's "Interrupted. Larger maxiters is needed." warnings while running `f`: an adjoint whose own
# solve stops at maxiters still returns a gradient, with only this warning. Uses Test's logger (defined when the
# package loads), since a logger type defined here is too new for code Enzyme compiles in an older world.
using Test: TestLogger
using Logging: with_logger, Warn

"Run `f()`; return its value and the number of maxiters warnings it logged (other log records are dropped)."
function count_maxiters(f)
    logger = TestLogger(; min_level = Warn)
    value = with_logger(f, logger)
    return value, count(r -> occursin("maxiters", string(r.message)), logger.logs)
end
