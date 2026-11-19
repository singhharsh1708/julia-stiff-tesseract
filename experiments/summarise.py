"""Collect the experiment results into Markdown (the CI run summary and results/<run>/README.md)."""

import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
ROUTES = ["forwarddiff", "enzyme", "adjoint-gauss-enzyme", "adjoint-gauss-reversediff", "adjoint-interp-enzyme"]


def load(pattern):
    seen, out = set(), []
    for p in sorted(root.glob(pattern)):
        if p.name not in seen:          # each file once, however many artifact folders hold it
            seen.add(p.name)
            out.append((p, json.loads(p.read_text())))
    return out


def s(x, unit=" s"):
    return "-" if x is None else f"{x:.3g}{unit}"


print("## VJP routes (generated model)\n")
print("| route | params | first call | warm (min of 2) | max rel. err vs ForwardDiff | note |")
print("|---|---|---|---|---|---|")
routes = {(r["route"], r["nparams"]): r for _, r in load("**/route-*.json")}
for rt in ROUTES:
    for n in (2, 8, 16):
        r = routes.get((rt, n))
        if r is None:
            print(f"| {rt} | {n} | - | - | - | no result |")
            continue
        note = "" if r.get("ok") else r.get("error", "")[:100].replace("|", "/").replace("\n", " ")
        print(f"| {rt} | {n} | {s(r.get('first_s'))} | {s(r.get('warm_min_s'))} | "
              f"{s(r.get('max_rel_err_vs_forwarddiff'), '')} | {note} |")

print("\n## Where EnzymeVJP goes wrong (GaussAdjoint, 2 parameters, vs ForwardDiff on the same function)\n")
print("| variant | EnzymeVJP first / second | EnzymeVJP max rel. err | ReverseDiffVJP first / second | ReverseDiffVJP max rel. err |")
print("|---|---|---|---|---|")
for _, r in load("**/enzymevjp-*.json"):
    cells = []
    for k in ("enzymevjp", "reversediffvjp"):
        v = r.get(k, {})
        if v.get("ok"):
            cells += [f"{s(v['first_s'])} / {s(v['second_s'])}", s(v["max_rel_err"], "")]
        else:
            cells += ["failed: " + v.get("error", "no result")[:80].replace("|", "/").replace("\n", " "), "-"]
    print(f"| {r['variant']} | " + " | ".join(cells) + " |")

print("\n## First call through juliacall, with and without a system image\n")
print("| run | load | first apply | warm apply | first VJP | warm VJP |")
print("|---|---|---|---|---|---|")
for p, r in load("**/sysimage-*.json"):
    if "load_s" in r:
        print(f"| {p.stem} | {s(r['load_s'])} | {s(r['first_apply_s'])} | {s(r['warm_apply_s'])} | "
              f"{s(r['first_vjp_s'])} | {s(r['warm_vjp_s'])} |")
for _, b in load("**/sysimage-build.json"):
    print(f"\nImage build: {b['build_s']} s, {b['size_bytes'] / 2**20:.0f} MiB, peak memory {b.get('max_rss_kb', 0) / 2**20:.1f} GiB")

print("\n## Container\n")
for _, r in load("**/container-*.json"):
    print("- " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))

print("\n## Figure 4 calibration\n")
for _, r in load("**/fig4.json"):
    for label in ("as_published", "qspatlas_fix"):
        if label in r:
            print(f"- {label}: rms over all {r['targets']} targets {r[label]['rms_all']:.3f}, "
                  f"over the 60 without the 6 known misfits {r[label]['rms_good']:.4f}")
    if "fit" in r:
        f = r["fit"]
        print(f"- fit ({r['value_and_grad_evals']} gradient evaluations, {r['calibration_s']:.0f} s): rms all "
              f"{f['rms_all']:.3f}, without misfits {f['rms_good']:.4f}; " +
              ", ".join(f"{k} {v:.4g}" for k, v in f["values"].items()))

print("\n## check_gradients step rules\n")
for p, r in load("**/gradcheck-*.json"):
    print(f"### {p.stem}\n")
    for rule, cases in r.items():
        cells = []
        for case, v in cases.items():
            if "passed" in v:
                cells.append(f"{case}: {'agree' if v['passed'] else 'flagged'}")
            else:
                cells.append(f"{case}: {'ok' if v.get('returncode') == 0 else 'error'}")
        print(f"- {rule}: " + "; ".join(cells))
    print()
