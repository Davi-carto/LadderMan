"""Camera-rate configuration and legacy checkpoint compatibility without Isaac Sim."""

from types import SimpleNamespace as Cfg

import pytest
import yaml

from ladder.utils.depth_camera_config import configure_depth_camera_timing


class warp_visual_depth:
    """Name-compatible stand-in for the actual ManagerTermBase class."""


def _config(period=1.0 / 30.0):
    return Cfg(observations=Cfg(
        transformer_vision_depth=Cfg(image=Cfg(func=warp_visual_depth, params={"update_period": period})),
    ))


def _period(cfg):
    return cfg.observations.transformer_vision_depth.image.params["update_period"]


def _checkpoint(tmp_path, params):
    config_dir = tmp_path / "params"
    config_dir.mkdir()
    payload = {"observations": {"transformer_vision_depth": {"image": {"params": params}}}}
    (config_dir / "env.yaml").write_text(yaml.dump(payload), encoding="utf-8")
    return tmp_path / "model_100.pt"


def test_fresh_run_keeps_configured_30_hz():
    cfg = _config()
    configure_depth_camera_timing(cfg)
    assert _period(cfg) == 1.0 / 30.0


def test_resume_and_play_restore_saved_rate_with_python_tuple_tags(tmp_path):
    checkpoint = _checkpoint(tmp_path, {"update_period": 0.1, "translation_range": (-0.01, 0.01)})
    cfg = _config()
    configure_depth_camera_timing(cfg, checkpoint=checkpoint)
    assert _period(cfg) == 0.1


def test_checkpoint_without_new_field_retains_legacy_policy_rate(tmp_path):
    checkpoint = _checkpoint(tmp_path, {"width": 80, "height": 60})
    cfg = _config()
    configure_depth_camera_timing(cfg, checkpoint=checkpoint)
    assert _period(cfg) == 0.0


def test_checkpoint_without_config_warns_and_retains_legacy_rate(tmp_path, capsys):
    cfg = _config()
    configure_depth_camera_timing(cfg, checkpoint=tmp_path / "model.pt")
    assert _period(cfg) == 0.0
    assert "assuming legacy policy-rate" in capsys.readouterr().out


def test_explicit_override_takes_priority_over_checkpoint(tmp_path):
    checkpoint = _checkpoint(tmp_path, {"update_period": 0.1})
    cfg = _config()
    configure_depth_camera_timing(cfg, checkpoint=checkpoint, update_hz=30.0)
    assert _period(cfg) == 1.0 / 30.0


def test_zero_hz_override_requests_policy_rate():
    cfg = _config()
    configure_depth_camera_timing(cfg, update_hz=0.0)
    assert _period(cfg) == 0.0


@pytest.mark.parametrize("rate", [-1.0, float("nan"), float("inf")])
def test_invalid_override_rejected(rate):
    with pytest.raises(ValueError, match="depth_update_hz"):
        configure_depth_camera_timing(_config(), update_hz=rate)


def test_nonvisual_env_is_unchanged():
    cfg = Cfg(observations=Cfg(policy=Cfg()))
    configure_depth_camera_timing(cfg)
    assert list(vars(cfg.observations)) == ["policy"]


def test_warp_visual_group_uses_the_same_timing_setting():
    cfg = Cfg(observations=Cfg(visual_depth=Cfg(image=Cfg(
        func="ladder.tasks.ladder_climb.mdp.observations:warp_visual_depth",
        params={"update_period": 0.1},
    ))))
    configure_depth_camera_timing(cfg, update_hz=30.0)
    assert cfg.observations.visual_depth.image.params["update_period"] == 1.0 / 30.0


def test_rtx_depth_term_does_not_receive_warp_timing_parameter():
    params = {"sensor_name": "depth_camera", "min_depth": 0.1}
    cfg = Cfg(observations=Cfg(visual_depth=Cfg(image=Cfg(
        func="ladder.tasks.ladder_climb.mdp.observations:visual_depth", params=params,
    ))))
    configure_depth_camera_timing(cfg, update_hz=30.0)
    assert params == {"sensor_name": "depth_camera", "min_depth": 0.1}
