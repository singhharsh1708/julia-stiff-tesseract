# First pass, 2026-11-18, MacBook (Apple silicon), Julia 1.10.12

Raw logs behind docs/findings.md and docs/log.md, on the ModelingToolkit model. The scripts that wrote them are in
commit ff3a387 (`scripts/step*.py`, `julia/scripts/*.jl`); `pipeline/` holds their current versions.
`raw-step5-*` are `check_gradients` (#300 as submitted, RMS steps) with one injected fault each; `raw-step5b-*`
the same with elementwise steps. Several timings were taken with the machine loaded (load average 10-19).
