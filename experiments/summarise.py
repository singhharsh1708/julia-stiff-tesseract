"""Collect the experiment results into Markdown tables (for the CI run summary)."""

import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])


def rows(pattern):
    return [json.loads(p.read_text()) for p in sorted(root.glob(pattern))]


def fmt(x, unit="s"):
    return "-" if x is None else f"{x:.3g} {unit}"


print("## Experiment 1: VJP routes\n")
print("| route | params | first call | second | warm (min of 2) | max rel. err vs ForwardDiff | note |")
print("|---|---|---|---|---|---|---|")
found = {(r["route"], r["nparams"]) for r in rows("**/route-*.json")}
expected = {(rt, n) for rt in ["forwarddiff", "enzyme", "adjoint-default", "adjoint-gauss-enzyme",
                                "adjoint-gauss-reversediff", "adjoint-interp-enzyme"] for n in (2, 8, 16)}
for rt, n in sorted(expected - found):
    print(f"| {rt} | {n} | - | - | - | - | no result (job crashed or timed out) |")
for r in sorted(rows("**/route-*.json"), key=lambda r: (r["route"], r["nparams"])):
    note = "" if r.get("ok") else r.get("error", "")[:120].replace("|", "/").replace("\n", " ")
    print(f"| {r['route']} | {r['nparams']} | {fmt(r.get('first_s'))} | {fmt(r.get('second_s'))} | "
          f"{fmt(r.get('warm_min_s'))} | {fmt(r.get('max_rel_err_vs_forwarddiff'), '')} | {note} |")

print("\n## Experiment 2: system image through juliacall\n")
print("| run | load | first apply | warm apply | first VJP | warm VJP | log10 ratio |")
print("|---|---|---|---|---|---|---|")
for p in sorted(root.glob("**/sysimage-*.json")):
    if p.name == "sysimage-build.json":
        continue
    r = json.loads(p.read_text())
    print(f"| {p.stem} | {fmt(r['load_s'])} | {fmt(r['first_apply_s'])} | {fmt(r['warm_apply_s'])} | "
          f"{fmt(r['first_vjp_s'])} | {fmt(r['warm_vjp_s'])} | {r['log10_ratio']:.3f} |")
for p in root.glob("**/sysimage-build.json"):
    b = json.loads(p.read_text())
    print(f"\nSystem image build: {b['build_s']} s, {b['size_bytes'] / 2**20:.0f} MiB")
