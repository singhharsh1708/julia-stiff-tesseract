# Source before running Python against the Julia project: juliacall uses this Julia and this project and
# skips its own package management (the same two variables the jpb/julia template sets).
export PYTHON_JULIACALL_EXE="$(julia -e 'print(joinpath(Sys.BINDIR, "julia"))')"
export PYTHON_JULIACALL_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/julia"
