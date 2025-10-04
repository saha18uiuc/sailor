# sailor/profiling/overlay_loader.py
import json, os
from typing import Dict, List, Optional, Union

StageTimes = Union[List[float], Dict[int, float]]

def _to_int_float_dict(d: Dict) -> Dict[int, float]:
    return {int(k): float(v) for k, v in d.items()}

def load_overlay_from_path(path: str) -> Dict:
    with open(path, "r") as f:
        obj = json.load(f)
    if "stage_delta_ms" not in obj or not isinstance(obj["stage_delta_ms"], dict):
        raise ValueError(f"[overlay_loader] overlay missing 'stage_delta_ms': {path}")
    return {
        "name": obj.get("name", "overlay"),
        "stage_delta_ms": _to_int_float_dict(obj["stage_delta_ms"]),
    }

def _resolve_path(repo_relative_or_abs: str) -> str:
    """If path is not absolute, resolve relative to the repo's sailor/ directory."""
    if os.path.isabs(repo_relative_or_abs):
        return repo_relative_or_abs
    # This file sits at sailor/profiling/, so we need to go up one level to sailor/
    here = os.path.dirname(__file__)
    sailor_dir = os.path.abspath(os.path.join(here, ".."))
    candidate = os.path.abspath(os.path.join(sailor_dir, repo_relative_or_abs))
    # We fallback to CWD relative if not found
    return candidate if os.path.exists(candidate) else os.path.abspath(repo_relative_or_abs)

def load_overlay_from_env() -> Optional[Dict]:
    """
    Read overlay JSON path from env (prefer LORA_OVERLAY_JSON, fallback SAILOR_ADAPTER_OVERLAY).
    Returns parsed overlay dict or None if unset.
    """
    env_path = os.getenv("LORA_OVERLAY_JSON") or os.getenv("SAILOR_ADAPTER_OVERLAY")
    if not env_path:
        return None
    path = _resolve_path(env_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"[overlay_loader] Overlay not found at: {path}")
    return load_overlay_from_path(path)

def apply_stage_overlay(stage_times_ms: StageTimes, overlay: Optional[Dict]) -> StageTimes:
    """
    stage_times_ms: per-stage compute time (fwd+bwd) in ms, BEFORE pipeline scheduling.
    overlay: {"stage_delta_ms": {0: 0.12, 1: 0.34, ...}}
    Returns a new object with deltas added; preserves input type (list or dict).
    """
    if overlay is None:
        return stage_times_ms

    deltas: Dict[int, float] = overlay["stage_delta_ms"]

    # dict[int->float] case
    if isinstance(stage_times_ms, dict):
        out = dict(stage_times_ms)
        for s, d in deltas.items():
            out[s] = float(out.get(s, 0.0)) + float(d)
        return out

    # list[float] case
    out = list(stage_times_ms)
    for s, d in deltas.items():
        if 0 <= s < len(out):
            out[s] = float(out[s]) + float(d)
    return out