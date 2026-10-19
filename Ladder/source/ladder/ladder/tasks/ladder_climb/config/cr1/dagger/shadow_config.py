"""Configuration-only controls for same-state teacher/student shadow evaluation.

The normal condition reproduces ``play.py --disable_randomization``.  The clean
condition additionally removes the sensor, motor and delay randomization that
that CLI flag leaves enabled.  Neither condition changes contacts, termination
thresholds, action scaling/rate limiting, or the training task definition.
"""

from __future__ import annotations


_PLAY_DISABLED_EVENTS = (
    "robot_material",
    "robot_mass",
    "joint_position_offset",
    "torso_com",
    "camera_extrinsics",
    "root_velocity_push",
)
_MOTOR_EVENTS = ("motor_strength", "motor_strength_interval")
_DEPTH_NOISE = (
    "gaussian_std",
    "pixel_dropout_probability",
    "rung_focused_mask_probability",
)
_CAMERA_RANGES = (
    "translation_range",
    "rotation_range",
    "horizontal_fov_randomization_range_deg",
    "vertical_fov_randomization_range_deg",
)
_RESET_FIELDS = (
    "fixed_reference_frame",
    "pose_range",
    "fixed_start_pose_range",
    "velocity_range",
    "joint_position_range",
    "ladder_inclination_delta_range",
    "ladder_spacing_delta_range",
)


def _json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if hasattr(value, "__dict__"):
        return {
            key: _json_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def configure_shadow_eval(env_cfg, reference_frame: int, seed: int, clean: bool) -> dict:
    """Mutate an instantiated pure-visual task config and return its audit manifest.

    ``seed`` is consumed by IsaacLab during environment construction.  The caller
    must not subsequently apply play/fast-mode overrides.  A clean condition has
    no configured stochastic input or dynamics perturbation, but is not a promise
    of bitwise deterministic GPU physics or of a physically trivial task.
    """
    if reference_frame < 0 or seed < 0:
        raise ValueError("Shadow evaluation requires a non-negative fixed frame and seed")

    changes = []

    def set_value(container, key, value, path):
        previous = container[key] if isinstance(container, dict) else getattr(container, key)
        before, after = _json_value(previous), _json_value(value)
        if isinstance(container, dict):
            container[key] = value
        else:
            setattr(container, key, value)
        if before != after:
            changes.append({"path": path, "before": before, "after": after})

    def disable_event(name):
        if hasattr(env_cfg.events, name):
            set_value(env_cfg.events, name, None, f"events.{name}")

    set_value(env_cfg, "seed", seed, "seed")
    motion = env_cfg.commands.motion
    set_value(motion, "fixed_reference_frame", reference_frame, "commands.motion.fixed_reference_frame")
    if hasattr(motion, "debug_vis"):
        set_value(motion, "debug_vis", False, "commands.motion.debug_vis")
    for name, scene_cfg in vars(env_cfg.scene).items():
        if hasattr(scene_cfg, "debug_vis"):
            set_value(scene_cfg, "debug_vis", False, f"scene.{name}.debug_vis")

    # Match play.py's existing flag exactly before adding the clean condition.
    for name in _PLAY_DISABLED_EVENTS:
        disable_event(name)
    zero_axes = {name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")}
    for name, value in (
        ("pose_range", dict(zero_axes)),
        ("fixed_start_pose_range", {}),
        ("velocity_range", dict(zero_axes)),
        ("joint_position_range", (0.0, 0.0)),
        ("ladder_inclination_delta_range", (0.0, 0.0)),
        ("ladder_spacing_delta_range", (0.0, 0.0)),
    ):
        set_value(motion, name, value, f"commands.motion.{name}")

    proprio_term = env_cfg.observations.transformer_vision_inputs.proprio_history
    depth_term = env_cfg.observations.transformer_vision_depth.image
    # Do not mutate dicts shared with another separately constructed condition.
    proprio_term.params = dict(proprio_term.params)
    depth_term.params = dict(depth_term.params)
    action = env_cfg.actions.joint_pos
    if clean:
        for name in _MOTOR_EVENTS:
            disable_event(name)
        for name, group in vars(env_cfg.observations).items():
            if hasattr(group, "enable_corruption"):
                set_value(group, "enable_corruption", False, f"observations.{name}.enable_corruption")
        set_value(
            proprio_term.params, "add_proprio_noise", False,
            "observations.transformer_vision_inputs.proprio_history.params.add_proprio_noise",
        )
        for name in _DEPTH_NOISE:
            set_value(depth_term.params, name, 0.0, f"observations.transformer_vision_depth.image.params.{name}")
        for name in _CAMERA_RANGES:
            set_value(depth_term.params, name, (0.0, 0.0), f"observations.transformer_vision_depth.image.params.{name}")
        for name in ("min_control_delay", "max_control_delay"):
            set_value(action, name, 0, f"actions.joint_pos.{name}")

    active_events = {
        name: _json_value(event)
        for name, event in vars(env_cfg.events).items()
        if not name.startswith("_") and event is not None
    }
    manager_noise = {}
    for name, group in vars(env_cfg.observations).items():
        if getattr(group, "enable_corruption", False):
            for term_name, term in vars(group).items():
                if getattr(term, "noise", None) is not None:
                    manager_noise[f"{name}.{term_name}"] = _json_value(term.noise)
    remaining = []
    if proprio_term.params["add_proprio_noise"]:
        remaining.append("Custom proprioceptive uniform sensor noise")
    remaining.extend(f"Warp depth {name}" for name in _DEPTH_NOISE if depth_term.params[name] > 0)
    remaining.extend(
        f"Warp camera {name}" for name in _CAMERA_RANGES
        if any(value != 0 for value in depth_term.params[name])
    )
    remaining.extend(f"Actuator randomization events.{name}" for name in _MOTOR_EVENTS if name in active_events)
    if action.max_control_delay > 0:
        remaining.append(f"Control delay sampled in [{action.min_control_delay}, {action.max_control_delay}] physics steps")
    remaining.extend(f"Observation manager noise: {name}" for name in manager_noise)

    return {
        "condition": "clean" if clean else "normal_play_disable_randomization",
        "seed": seed,
        "reference_frame": reference_frame,
        "changes": changes,
        "remaining_configured_perturbations": remaining,
        "effective": {
            "reset": {name: _json_value(getattr(motion, name)) for name in _RESET_FIELDS},
            "proprio_params": _json_value(proprio_term.params),
            "depth_params": _json_value(depth_term.params),
            "active_events": active_events,
            "active_manager_observation_noise": manager_noise,
            "action": _json_value(action),
        },
        "unchanged": [
            "Motion-command reset implementation, valid-frame checks, and reference progression",
            "Scene collision geometry, gravity, contacts, and termination thresholds",
            "Action scale/default offsets, clipping, and target-velocity rate limit",
            "Depth image dimensions, clipping/normalization, and history initialization/update",
            "Any non-random reset or semantic/material annotation events (see active_events)",
        ],
        "caveat": "Configured stochastic perturbations are separate from GPU physics numerical nondeterminism.",
    }
