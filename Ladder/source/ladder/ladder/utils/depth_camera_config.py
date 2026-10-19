"""Keep Warp camera timing explicit and compatible with saved visual policies."""

import math
from pathlib import Path

import yaml


def configure_depth_camera_timing(env_cfg, *, checkpoint=None, update_hz=None):
    """Use new-task defaults, checkpoint timing, or an explicit CLI override.

    Older runs did not save ``update_period`` and sampled once per policy step.
    Read only primitive YAML data: IsaacLab dumps can contain Python tuple tags,
    but loading camera metadata must never construct Python objects.
    """
    if update_hz is not None:
        update_hz = float(update_hz)
        if not math.isfinite(update_hz) or update_hz < 0.0:
            raise ValueError("depth_update_hz must be finite and nonnegative (0 means policy rate)")

    depth_groups = {}
    for name in ("transformer_vision_depth", "visual_depth"):
        group = getattr(env_cfg.observations, name, None)
        term = getattr(group, "image", None)
        if term is None:
            continue
        func = term.func
        func_name = func.rsplit(":", 1)[-1] if isinstance(func, str) else getattr(func, "__name__", "")
        # RTX observations share the visual_depth group name but their term
        # does not accept update_period: its sensor owns the capture clock.
        if func_name == "warp_visual_depth":
            depth_groups[name] = term.params
    if not depth_groups:
        return

    saved_observations = {}
    if checkpoint is not None and update_hz is None:
        config_path = Path(checkpoint).expanduser().resolve().parent / "params" / "env.yaml"
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as stream:
                saved_cfg = yaml.load(stream, Loader=yaml.BaseLoader)
            saved_observations = saved_cfg.get("observations", {})
        else:
            print(
                f"[WARN] No camera timing metadata at {config_path}; assuming legacy policy-rate "
                "depth. Use --depth_update_hz to override if this checkpoint used another rate.",
                flush=True,
            )

    for name, params in depth_groups.items():
        source = "task config"
        if update_hz is not None:
            period = 0.0 if update_hz == 0.0 else 1.0 / update_hz
            source = "CLI override"
        elif checkpoint is not None:
            saved_params = saved_observations.get(name, {}).get("image", {}).get("params", {})
            period = float(saved_params.get("update_period", 0.0))
            source = "checkpoint config" if "update_period" in saved_params else "legacy checkpoint"
        else:
            period = float(params.get("update_period", 0.0))
        if not math.isfinite(period) or period < 0.0:
            raise ValueError(f"Invalid {name} camera update_period: {period}")
        params["update_period"] = period
        rate = "policy rate" if period == 0.0 else f"{1.0 / period:g} Hz (policy-step sampled)"
        print(f"[INFO] {name}: {rate}; update_period={period:g} s; source={source}", flush=True)
