"""Build a self-contained HTML report from out/*.json and out/*.png."""
import base64
import glob
import html
import json
import os
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "out"
dest = sys.argv[2] if len(sys.argv) > 2 else "report.html"


def img(name, alt):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        return f'<p class="missing">({html.escape(name)} not produced)</p>'
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f'<figure class="plate"><img src="data:image/png;base64,{b}" alt="{html.escape(alt)}"></figure>'


def load(tag):
    p = os.path.join(out, tag + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


runs = {t: load(t) for t in ["hopf_4steps_256", "hopf_10steps_256", "hopf_10steps_512",
                              "lorenz_4steps_40", "lorenz_10steps_40"]}
runs = {k: v for k, v in runs.items() if v}
scaling = json.load(open(os.path.join(out, "scaling.json"))) if os.path.exists(os.path.join(out, "scaling.json")) else []

BIF_T = {"Hopf2D": 0.5, "Lorenz3D": (1.0 - 0.2) / (6.0 - 0.2)}


def fmt_poly(p):
    return html.escape(p.replace("t^2", "t²").replace("t^3", "t³"))


def run_row(tag, r):
    d = r["detection"]
    tb = BIF_T[r["system"]]
    det = f'{d["t_detect"]:.2f} s' if d else "—"
    err = f'{d["t_detect"] - tb:+.2f} s' if d else "—"
    kind = d["kind"] if d else "not detected"
    ev = [e for e in r["events"] if e["kind"] != "continuation"]
    eq = next((e["morse_equation"] for e in ev if e["morse_equation"]), "")
    ok = "✓" in eq
    total = r["timings"].get("total", sum(v for k, v in r["timings"].items() if k != "cmgraph_conley_only"))
    return (f'<tr><td>{html.escape(r["system"])}</td><td class="num">{r["n_steps"]} × {r["step_ms"]:.0f} ms</td>'
            f'<td class="num">{"×".join(map(str, r["grid"]))}</td><td class="num">{det}</td>'
            f'<td class="num">{err}</td><td>{html.escape(kind)}</td>'
            f'<td>{"<span class=pill-ok>holds</span>" if ok else ("—" if not eq else "<span class=pill-bad>fails</span>")}</td>'
            f'<td class="num">{total:.0f} s</td></tr>')


explorer = {}
for tag, r in runs.items():
    explorer[tag] = {
        "label": f'{r["system"]} · {r["n_steps"]} steps · {"×".join(map(str, r["grid"]))}',
        "spaces": [{"label": s["label"],
                    "blocks": [{"conley": b["conley"].replace("t^2", "t²"), "cells": b["cells"],
                                "track": b["track"], "trivial": b["trivial"], "level": b["level"],
                                "cm": b["cmgraph_conley"], "cmstatus": b["cmgraph_status"]}
                               for b in s["blocks"]]} for s in r["spaces"]],
        "steps": r["steps"],
        "events": [e for e in r["events"] if e["kind"] != "continuation"],
    }

scal_rows = "".join(
    f'<tr><td>{html.escape(s["group"])}</td><td class="num">{s["grid"]}<sup>{2 if "Hopf" in s["group"] else 3}</sup></td>'
    f'<td class="num">{s["cells"]:,}</td><td class="num">{s["timings"].get("enclosure", 0):.1f}</td>'
    f'<td class="num">{s["timings"].get("cmgraph_morse_graph", 0):.1f}</td>'
    f'<td class="num">{s["timings"].get("pair_homology", 0) + s["timings"].get("index_pairs", 0):.1f}</td>'
    f'<td class="num">{s["timings"].get("zigzag_persistence", 0):.1f}</td><td class="num">{s["total"]:.1f}</td>'
    f'<td class="num">{s["peak_mb"]:.0f}</td>'
    f'<td>{("t=" + format(s["detection"]["t"], ".2f") + " (" + s["detection"]["kind"] + ")") if s["detection"] else "—"}</td></tr>'
    for s in scaling)

TEMPLATE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html")).read()
page = (TEMPLATE
        .replace("{{RUN_ROWS}}", "".join(run_row(t, r) for t, r in runs.items()))
        .replace("{{IMG_COMPARISON}}", img("hopf_comparison.png", "Barcodes for 4 and 10 steps"))
        .replace("{{IMG_CMG4}}", img("hopf_4steps_256_cmg_zigzag.png", "CMG zigzag, 4 steps"))
        .replace("{{IMG_CMG10}}", img("hopf_10steps_256_cmg_zigzag.png", "CMG zigzag, 10 steps"))
        .replace("{{IMG_PORT4}}", img("hopf_4steps_256_portraits.png", "Morse sets per step, 4 steps"))
        .replace("{{IMG_PORT10}}", img("hopf_10steps_256_portraits.png", "Morse sets per step, 10 steps"))
        .replace("{{IMG_LZ_CMG}}", img("lorenz_4steps_40_cmg_zigzag.png", "Lorenz CMG zigzag"))
        .replace("{{IMG_LZ_PORT}}", img("lorenz_4steps_40_portraits.png", "Lorenz Morse sets (x–z projection)"))
        .replace("{{IMG_LZ_BAR}}", img("lorenz_comparison.png", "Lorenz barcodes"))
        .replace("{{IMG_SCALING}}", img("scaling.png", "Runtime by stage versus grid size"))
        .replace("{{SCALING_ROWS}}", scal_rows)
        .replace("{{EXPLORER_JSON}}", json.dumps(explorer, ensure_ascii=False).replace("</", "<\\/")))
open(dest, "w").write(page)
print("wrote", dest, f"{len(page) / 1e6:.1f} MB")
