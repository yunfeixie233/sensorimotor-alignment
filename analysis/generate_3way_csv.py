"""Generate CSV with 3-way CKNNA results (imgtext/img/txt) at N=5K.

Reads the JSON results from 3way_cknna_results/ and the original CSV for
success rates, then produces a combined CSV.

Usage:
    python generate_3way_csv.py
"""

import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.environ.get("RECORD_DIR", "/home/ubuntu/verl/starVLA/cknna/record")
DATA_DIR = os.path.join(RECORD_DIR, "..", "cknna_data")
RESULTS_DIR = os.path.join(DATA_DIR, "3way_cknna_results")
ORIG_CSV = os.path.join(RECORD_DIR, "cknna_action_proprio_simplerenv.csv")
OUT_CSV = os.path.join(RECORD_DIR, "cknna_3way_5k.csv")

MODELS_3WAY = [
    "Qwen-GR00T-Bridge",
    "Qwen-GR00T-Bridge-RT-1",
    "Qwen3VL-GR00T-Bridge-RT-1",
    "Qwen-FAST-Bridge-RT-1",
    "Qwen-OFT-Bridge-RT-1",
    "Qwen3VL-OFT-Bridge-RT-1",
    "spatialvla-sft-bridge",
    "pi0-lerobot-bridge",
    "openvla-7b-bridge",
    "openvla-7b-bridge-ft-200k",
    "cogact-small-bridge",
    "cogact-base-bridge",
    "cogact-large-bridge",
    "groot-n15-bridge",
    "groot-n16-bridge",
]

CSV_TO_DIR = {
    "Qwen2.5-GR00T-Bridge": "Qwen-GR00T-Bridge",
    "Qwen2.5-GR00T-Bridge-RT-1": "Qwen-GR00T-Bridge-RT-1",
    "Qwen3-GR00T-Bridge-RT-1": "Qwen3VL-GR00T-Bridge-RT-1",
    "Qwen2.5-FAST-Bridge-RT-1": "Qwen-FAST-Bridge-RT-1",
    "Qwen2.5-OFT-Bridge-RT-1": "Qwen-OFT-Bridge-RT-1",
    "Qwen3-OFT-Bridge-RT-1": "Qwen3VL-OFT-Bridge-RT-1",
    "spatialvla-sft-bridge": "spatialvla-sft-bridge",
    "pi0-lerobot-bridge": "pi0-lerobot-bridge",
    "openvla-7b-bridge": "openvla-7b-bridge",
    "openvla-7b-bridge-ft-200k": "openvla-7b-bridge-ft-200k",
    "CogACT-Small": "cogact-small-bridge",
    "CogACT-Base": "cogact-base-bridge",
    "CogACT-Large": "cogact-large-bridge",
    "GR00T-N1.5-Bridge": "groot-n15-bridge",
    "GR00T-N1.6-Bridge": "groot-n16-bridge",
    "RT-1-X": "rt1x-bridge",
    "Octo-base": "octo-base-bridge",
}


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    orig_rows = {}
    with open(ORIG_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig_rows[row["Model"]] = row

    proprio = {}
    action = {}
    for variant in ["feats_A", "feats_A_img", "feats_A_txt"]:
        p_data = load_json(os.path.join(RESULTS_DIR, f"cknna_proprio_{variant}.json"))
        proprio[variant] = p_data.get("models", {})
        a_data = load_json(os.path.join(RESULTS_DIR, f"cknna_action_{variant}.json"))
        action[variant] = a_data.get("models", {})

    rt1x_octo = load_json(os.path.join(RESULTS_DIR, "cknna_proprio_rt1x_octo.json"))
    rt1x_octo_models = rt1x_octo.get("models", {})

    rt1x_octo_action = {}
    for fn in ["cknna_action_rt1x_octo_rt1x-bridge.json",
               "cknna_action_rt1x_octo_octo-base-bridge.json"]:
        d = load_json(os.path.join(RESULTS_DIR, fn))
        rt1x_octo_action.update(d.get("models", {}))

    fieldnames = [
        "Model", "dir_name", "WidowX_avg",
        "CKNNA_proprio_imgtext_k10", "CKNNA_proprio_img_k10", "CKNNA_proprio_txt_k10",
        "CKNNA_action_imgtext_k10", "CKNNA_action_img_k10", "CKNNA_action_txt_k10",
        "MutualKNN_proprio_imgtext_k10", "MutualKNN_proprio_img_k10", "MutualKNN_proprio_txt_k10",
        "MutualKNN_action_imgtext_k10", "MutualKNN_action_img_k10", "MutualKNN_action_txt_k10",
    ]

    rows = []
    for csv_name, dir_name in CSV_TO_DIR.items():
        orig = orig_rows.get(csv_name, {})
        success = orig.get("WidowX_avg_entire", "")

        row = {
            "Model": csv_name,
            "dir_name": dir_name,
            "WidowX_avg": success,
        }

        for suffix, variant in [("imgtext", "feats_A"), ("img", "feats_A_img"), ("txt", "feats_A_txt")]:
            p_entry = proprio.get(variant, {}).get(dir_name, {})
            a_entry = action.get(variant, {}).get(dir_name, {})

            if not p_entry and dir_name in rt1x_octo_models and suffix == "imgtext":
                p_entry = rt1x_octo_models[dir_name]
            if not a_entry and dir_name in rt1x_octo_action and suffix == "imgtext":
                a_entry = rt1x_octo_action[dir_name]

            row[f"CKNNA_proprio_{suffix}_k10"] = p_entry.get("cknna_k10", "")
            row[f"MutualKNN_proprio_{suffix}_k10"] = p_entry.get("mutual_knn_k10", "")
            row[f"CKNNA_action_{suffix}_k10"] = a_entry.get("cknna_k10", "")
            row[f"MutualKNN_action_{suffix}_k10"] = a_entry.get("mutual_knn_k10", "")

        rows.append(row)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV} ({len(rows)} models)")


if __name__ == "__main__":
    main()
