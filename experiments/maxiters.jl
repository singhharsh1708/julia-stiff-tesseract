# Count the solver's "Interrupted. Larger maxiters is needed." warnings while running `f`: an adjoint whose own
# solve stops at maxiters still returns a gradient, with only this warning. Uses Test's logger (defined when the
# package loads), since a logger type defined here is too new for code Enzyme compiles in an older world.
using Test: TestLogger
using Logging: with_logger, Warn

"""
Run `f()`; return its value, the number of maxiters warnings it logged, and the other warnings (their first
200 characters, each once), so a dtmin or instability stop is not lost either.
"""
function count_maxiters(f)
    logger = TestLogger(; min_level = Warn)
    value = with_logger(f, logger)
    messages = [string(r.message) for r in logger.logs]
    n = count(m -> occursin("maxiters", m), messages)
    return value, n, unique(first.(filter(m -> !occursin("maxiters", m), messages), 200))
end
