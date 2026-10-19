"""Simulator-free checks of the shadow diagnostic's explicit config overrides."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as Cfg

import pytest


_PATH = Path(__file__).parents[1] / "source/ladder/ladder/tasks/ladder_climb/config/cr1/dagger/shadow_config.py"
_SPEC = importlib.util.spec_from_file_location("ladder_shadow_config_unit", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
configure_shadow_eval = _MODULE.configure_shadow_eval


def _config():
    event = lambda: Cfg(func="mock:randomize", mode="reset", params={"range": (0.8, 1.2)})
    return Cfg(
        seed=None,
        scene=Cfg(contact_sensor=Cfg(debug_vis=True)),
        commands=Cfg(motion=Cfg(
            debug_vis=True, fixed_reference_frame=None,
            pose_range={"x": (-0.1, 0.1)}, fixed_start_pose_range={"x": (-0.2, 0.2)},
            velocity_range={"x": (-0.1, 0.1)}, joint_position_range=(-0.1, 0.1),
            ladder_inclination_delta_range=(-1.0, 1.0), ladder_spacing_delta_range=(-0.01, 0.01),
        )),
        events=Cfg(
            robot_material=event(), robot_mass=event(), joint_position_offset=event(),
            torso_com=event(), camera_extrinsics=event(), root_velocity_push=event(),
            motor_strength=event(), motor_strength_interval=event(),
            reset_scene=Cfg(func="mock:reset", mode="reset", params={}),
            annotate_materials=Cfg(func="mock:annotate", mode="startup", params={}),
        ),
        observations=Cfg(
            transformer_vision_inputs=Cfg(
                enable_corruption=False,
                proprio_history=Cfg(params={"add_proprio_noise": True, "num_proprio_frames": 9}),
            ),
            transformer_vision_depth=Cfg(
                enable_corruption=False,
                image=Cfg(params={
                    "width": 80, "height": 60, "min_depth": 0.1, "max_depth": 2.0,
                    "gaussian_std": 0.005, "pixel_dropout_probability": 0.05,
                    "rung_focused_mask_probability": 0.1,
                    "translation_range": (-0.01, 0.01), "rotation_range": (-0.026, 0.026),
                    "horizontal_fov_randomization_range_deg": (-1.0, 1.0),
                    "vertical_fov_randomization_range_deg": (-1.0, 1.0),
                }),
            ),
            teacher=Cfg(enable_corruption=True, joint_pos=Cfg(noise=Cfg(n_min=-0.1, n_max=0.1))),
        ),
        actions=Cfg(joint_pos=Cfg(min_control_delay=0, max_control_delay=2, max_target_velocity=6.0, scale=0.25)),
        rewards=Cfg(), terminations=Cfg(required_contact=Cfg(consecutive_steps=30)),
    )


def test_normal_retains_all_play_remaining_randomization():
    cfg = _config()
    report = configure_shadow_eval(cfg, reference_frame=0, seed=42, clean=False)
    assert cfg.seed == 42 and cfg.commands.motion.fixed_reference_frame == 0
    assert cfg.events.robot_material is None and cfg.events.root_velocity_push is None
    assert cfg.events.motor_strength is not None and cfg.events.motor_strength_interval is not None
    assert cfg.observations.transformer_vision_inputs.proprio_history.params["add_proprio_noise"]
    assert cfg.observations.transformer_vision_depth.image.params["translation_range"] == (-0.01, 0.01)
    assert cfg.actions.joint_pos.max_control_delay == 2
    assert not cfg.commands.motion.debug_vis and not cfg.scene.contact_sensor.debug_vis
    assert report["remaining_configured_perturbations"]
    assert json.loads(json.dumps(report))["condition"] == "normal_play_disable_randomization"


def test_clean_removes_perturbations_without_changing_task_or_rate_limit():
    cfg = _config()
    reset_event, semantic_event = cfg.events.reset_scene, cfg.events.annotate_materials
    terminations, rewards = cfg.terminations, cfg.rewards
    report = configure_shadow_eval(cfg, reference_frame=1500, seed=3, clean=True)
    assert cfg.events.motor_strength is None and cfg.events.motor_strength_interval is None
    assert cfg.events.reset_scene is reset_event and cfg.events.annotate_materials is semantic_event
    assert cfg.terminations is terminations and cfg.rewards is rewards
    assert cfg.commands.motion.fixed_reference_frame == 1500
    assert cfg.commands.motion.fixed_start_pose_range == {}
    assert not cfg.observations.transformer_vision_inputs.proprio_history.params["add_proprio_noise"]
    assert not cfg.observations.teacher.enable_corruption
    assert cfg.actions.joint_pos.min_control_delay == cfg.actions.joint_pos.max_control_delay == 0
    assert cfg.actions.joint_pos.max_target_velocity == 6.0 and cfg.actions.joint_pos.scale == 0.25
    depth = cfg.observations.transformer_vision_depth.image.params
    assert (depth["width"], depth["height"], depth["min_depth"], depth["max_depth"]) == (80, 60, 0.1, 2.0)
    assert report["remaining_configured_perturbations"] == []
    assert report["effective"]["active_manager_observation_noise"] == {}
    assert json.loads(json.dumps(report))["condition"] == "clean"


def test_manifest_has_values_before_and_after_and_is_idempotent():
    cfg = _config()
    report = configure_shadow_eval(cfg, reference_frame=0, seed=42, clean=True)
    delay = next(item for item in report["changes"] if item["path"] == "actions.joint_pos.max_control_delay")
    assert delay["before"] == 2 and delay["after"] == 0
    assert configure_shadow_eval(cfg, reference_frame=0, seed=42, clean=True)["changes"] == []


@pytest.mark.parametrize("frame,seed", [(-1, 42), (0, -1)])
def test_non_diagnostic_random_seed_or_frame_rejected(frame, seed):
    with pytest.raises(ValueError, match="non-negative"):
        configure_shadow_eval(_config(), reference_frame=frame, seed=seed, clean=False)
