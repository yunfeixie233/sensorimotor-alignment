"""Step 5 — double-check that every JEPA number written into the §F paper
edits traces exactly to the computed CSVs."""
import csv
import re

import os as _os
DIR = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "record", "fig_appendix_paradigm_imagenet", "data",
)
TEX = "/lambda/nfs/vla/vla_idea/writing/cknna_vla"   # dev-only: paper LaTeX source on the dev machine


def h7(path):
    return {r["model"]: round(float(r["cknna_k10"]), 4)
            for r in csv.DictReader(open(path)) if int(r["horizon"]) == 7}


s1 = h7(f"{DIR}/results_cknna_standalone_vit_jepa.csv")
real = h7(f"{DIR}/results_cknna_jepa_realclip.csv")
tiled = h7(f"{DIR}/results_cknna_jepa_tiled4890.csv")

NAME2KEY = {
    "I-JEPA ViT-H/14": "ijepa-vith14", "V-JEPA ViT-L/16": "vjepa-v1-vitl",
    "V-JEPA ViT-H/16": "vjepa-v1-vith", "V-JEPA 2 ViT-L": "vjepa2-vitl",
    "V-JEPA 2 ViT-H": "vjepa2-vith", "V-JEPA 2 ViT-g": "vjepa2-vitg",
    "V-JEPA 2.1 ViT-L": "vjepa2-1-vitl", "V-JEPA 2.1 ViT-g": "vjepa2-1-vitg",
}


def num(cell):
    m = re.search(r"[\d.]+", cell)
    return float(m.group()) if m else None


print("=== main §F table (paradigm-data-size.tex) S2 vs results_cknna_standalone_vit_jepa.csv ===")
ok = True
for line in open(f"{TEX}/tables/paradigm-data-size.tex"):
    if "&" not in line:
        continue
    cells = [c.strip() for c in line.split("&")]
    if cells[0] in NAME2KEY:
        tex = num(cells[5])
        csvv = s1[NAME2KEY[cells[0]] + "-standalone"]
        m = abs(tex - csvv) < 1e-4
        ok &= m
        print(f"  {cells[0]:18s} tex={tex:.4f}  csv={csvv:.4f}  {'OK' if m else 'MISMATCH'}")
print(f"  --> main table all match: {ok}")

print()
print("=== Setting-2 sub-table (paradigm-jepa-realclip.tex) tiled/real vs CSVs ===")
ok2 = True
for line in open(f"{TEX}/tables/paradigm-jepa-realclip.tex"):
    if "&" not in line or "Encoder" in line:
        continue
    cells = [c.strip() for c in line.split("&")]
    if cells[0] in NAME2KEY:
        k = NAME2KEY[cells[0]]
        tt, tr = num(cells[1]), num(cells[2])
        ct, cr = tiled[k + "-tiled"], real[k + "-realclip"]
        m = abs(tt - ct) < 1e-4 and abs(tr - cr) < 1e-4
        ok2 &= m
        print(f"  {cells[0]:18s} tiled tex={tt:.4f}/csv={ct:.4f}  real tex={tr:.4f}/csv={cr:.4f}  "
              f"{'OK' if m else 'MISMATCH'}")
print(f"  --> sub-table all match: {ok2}")

print()
print("=== Finding-1 scatter cohort: I-JEPA point ===")
for line in open(f"{TEX}/figures/paradigm_imagenet_scatter.py"):
    if "I-JEPA" in line and "JEPA" in line and "0.0" in line:
        print(f"  scatter line: {line.strip()}")
        sv = num(line.split('"JEPA"')[1])
        print(f"  scatter I-JEPA S2={sv:.4f}  csv={s1['ijepa-vith14-standalone']:.4f}  "
              f"{'OK' if abs(sv - s1['ijepa-vith14-standalone']) < 1e-4 else 'MISMATCH'}")
