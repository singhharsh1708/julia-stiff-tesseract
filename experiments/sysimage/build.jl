# Experiment 2: build a system image from the statements traced while running workload.py.
#   julia --project=experiments/sysimage experiments/sysimage/build.jl <traces.jl> <out.so>
using PackageCompiler
traces, out = ARGS
project = joinpath(@__DIR__, "..", "..", "julia")
t = @elapsed create_sysimage(["StiffTesseract", "PythonCall", "ModelingToolkit", "OrdinaryDiffEq", "ForwardDiff"];
    sysimage_path = out, project = project, precompile_statements_file = traces,
    include_transitive_dependencies = false)
println("sysimage built in ", round(t; digits = 1), " s: ", filesize(out) / 2^20, " MiB")
