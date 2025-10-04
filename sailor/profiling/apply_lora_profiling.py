import json, sys

def apply_overlay(base_path, overlay_path, out_path):
    with open(base_path) as f: base = json.load(f)
    with open(overlay_path) as f: overlay = json.load(f)

    for sid, extra in overlay["stage_delta_ms"].items():
        sid_i = int(sid)
        if "time_ms" in base["stages"][sid_i]:
            base["stages"][sid_i]["time_ms"] += float(extra)
        else:
            base["stages"][sid_i]["fwd_ms"] += float(extra) / 2.0
            base["stages"][sid_i]["bwd_ms"] += float(extra) / 2.0

    base["variant"] = overlay.get("name", "adapter-variant")
    with open(out_path, "w") as f: json.dump(base, f, indent=2)

if __name__ == "__main__":
    apply_overlay(sys.argv[1], sys.argv[2], sys.argv[3])
