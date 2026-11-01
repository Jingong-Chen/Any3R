# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Tables of the any-camera benchmark from evaluate.py results.

    python eval/make_table.py --results results/any3r                  # Markdown, main + 13-camera tables
    python eval/make_table.py --results results/any3r --compare-paper  # with the paper's values beside them
    python eval/make_table.py --paper                                  # the paper's Any3R values alone
    python eval/make_table.py --results results/any3r --format latex

Main table (the Any3R row of the paper's comparison tables): Indoor = mean of Matterport3D and
Stanford2D3DS, Outdoor = PanoCity, for the eight camera models of the main tables; Acc (m) is the
mean over the domain's datasets of the cell mean over scenes of the per-scene Acc, AUC@15 / AUC@30
(%) the mean over the same datasets of the cell's AUC over all its view pairs. The 13-camera table
gives the same three values per dataset.
"""
import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from benchlib import protocol as P                                      # noqa: E402

PAPER = os.path.join(HERE, "benchmark", "paper_any3r.json")
SHORT = {"pinhole": "Pin.", "brown_conrady": "B-C", "equidistant": "Equi.", "erp_crop": "ERP-C", "mei": "MEI",
         "kb4": "KB4", "double_sphere": "DS", "erp": "ERP", "fov": "FOV", "stereographic": "Stereo.",
         "equisolid": "Equisol.", "eucm": "EUCM", "cylindrical": "Cyl."}
METRICS = (("Acc", "Acc (m)"), ("AUC@15", "AUC@15 (%)"), ("AUC@30", "AUC@30 (%)"))


def load_results(path):
    """{(ds, camera): {"Acc", "AUC@15", "AUC@30"}} from an evaluate.py results directory."""
    out = {}
    for ds in P.DATASETS:
        for kind in P.CAMERA_KINDS:
            f = os.path.join(path, ds, f"{kind}.json")
            if os.path.exists(f):
                with open(f, encoding="utf-8") as fh:
                    s = json.load(fh)["summary"]
                out[(ds, kind)] = {k: s[k] for k, _ in METRICS}
    return out


def load_paper(path=PAPER):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {(c["ds"], c["camera"]): {k: c["cell"][k] for k, _ in METRICS} for c in d["cells"].values()}


def fmt(metric, v):
    if v is None:
        return "--"
    return f"{v:.3f}" if metric == "Acc" else f"{100 * v:.1f}"


def domain_values(cells, cams):
    """{(domain, camera): {metric: value or None}}: None when a dataset of the domain is missing."""
    out = {}
    for dn, dss in P.DOMAINS:
        for cam in cams:
            vals = [cells.get((ds, cam)) for ds in dss]
            out[(dn, cam)] = ({k: st.fmean(v[k] for v in vals) for k, _ in METRICS} if all(vals)
                              else {k: None for k, _ in METRICS})
    return out


def _latex(text):
    return text.replace("%", r"\%").replace("°", r"$^\circ$")


def table(rows, header, fmt_kind):
    """rows: [[cells...]]; header: [names...]."""
    if fmt_kind == "latex":
        lines = [r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\toprule",
                 " & ".join(map(_latex, header)) + r" \\", r"\midrule"]
        lines += [" & ".join(map(_latex, r)) + r" \\" for r in rows]
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + [":-:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def cell_text(metric, v, ref):
    if ref is None:
        return fmt(metric, v)
    if v is None:
        return f"-- ({fmt(metric, ref)})"
    return f"{fmt(metric, v)} ({fmt(metric, ref)})"


def main_table(cells, ref, fmt_kind):
    cams = P.MAIN_CAMERAS
    D = domain_values(cells, cams)
    R = domain_values(ref, cams) if ref is not None else None
    out = []
    for dn, _ in P.DOMAINS:
        header = [dn] + [f"{SHORT[c]} {P.CAMERA_FOV[c]:.0f}°" for c in cams]
        rows = []
        for m, label in METRICS:
            rows.append([label] + [cell_text(m, D[(dn, c)][m], R[(dn, c)][m] if R else None) for c in cams])
        out.append(table(rows, header, fmt_kind))
    return "\n\n".join(out)


def full_table(cells, ref, fmt_kind):
    out = []
    for ds in P.DATASETS:
        header = [f"{P.DATASET_NAMES[ds]}: camera (FoV)"] + [label for _, label in METRICS]
        rows = []
        for cam in P.CAMERA_KINDS:
            v = cells.get((ds, cam))
            r = ref.get((ds, cam)) if ref is not None else None
            rows.append([f"{P.CAMERA_LABELS[cam]} ({P.CAMERA_FOV[cam]:.0f}°)"] +
                        [cell_text(m, v[m] if v else None, r[m] if r else None) for m, _ in METRICS])
        out.append(table(rows, header, fmt_kind))
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=None, help="evaluate.py results directory")
    ap.add_argument("--paper", nargs="?", const=PAPER, default=None,
                    help="print the paper's values (benchmark/paper_any3r.json) instead of results")
    ap.add_argument("--compare-paper", action="store_true", help="add the paper's value in parentheses")
    ap.add_argument("--format", choices=("markdown", "latex"), default="markdown")
    a = ap.parse_args()
    if a.paper and not a.results:
        cells, ref = load_paper(a.paper), None
        title = "Any3R, paper values"
    elif a.results:
        cells = load_results(a.results)
        ref = load_paper(a.paper or PAPER) if a.compare_paper else None
        title = f"Any3R, {a.results}" + (" (paper in parentheses)" if ref else "")
        n = len(cells)
        if n < len(P.DATASETS) * len(P.CAMERA_KINDS):
            print(f"note: {n} of {len(P.DATASETS) * len(P.CAMERA_KINDS)} cells are complete; missing ones print --\n")
    else:
        sys.exit("give --results and/or --paper")
    print(f"## Main table ({title})\n")
    print(main_table(cells, ref, a.format))
    print(f"\n## All 13 camera models per dataset ({title})\n")
    print(full_table(cells, ref, a.format))


if __name__ == "__main__":
    main()
