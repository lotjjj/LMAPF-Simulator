from __future__ import annotations

import json
from typing import Any, Optional, Sequence


def parse_planner_args(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("planner_args must be a JSON object")
        return obj
    except Exception:
        pass

    def parse_scalar(value: str) -> Any:
        value = value.strip()
        lowered = value.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        if lowered in ("none", "null"):
            return None
        try:
            return int(value)
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            pass
        return value.strip("\"' ")

    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        stripped = stripped[1:-1].strip()

    parsed: dict[str, Any] = {}
    if not stripped:
        return parsed

    parts = [part.strip() for part in stripped.replace(";", ",").split(",") if part.strip()]
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            raise ValueError("planner_args must be JSON or key=value pairs")
        parsed[key.strip().strip("\"' ")] = parse_scalar(value)
    return parsed


def normalize_map_sizes(map_sizes: Optional[Sequence[str]]) -> list[str]:
    from LMAPFEnv.envs import PRESET_MAPS

    if not map_sizes:
        return list(PRESET_MAPS.keys())
    for map_size in map_sizes:
        if map_size not in PRESET_MAPS:
            raise ValueError(
                f"Unknown map_size '{map_size}', expected one of {list(PRESET_MAPS.keys())}"
            )
    return list(map_sizes)
