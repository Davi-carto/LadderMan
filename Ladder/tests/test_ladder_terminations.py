import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _load_terminations_module():
    commands = ModuleType("ladder.tasks.ladder_climb.mdp.commands")
    commands.LadderMotionCommand = object
    managers = ModuleType("isaaclab.managers")
    managers.ManagerTermBase = object
    math_utils = ModuleType("isaaclab.utils.math")
    math_utils.quat_error_magnitude = lambda *args, **kwargs: None
    math_utils.subtract_frame_transforms = lambda *args, **kwargs: None

    path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "ladder"
        / "ladder"
        / "tasks"
        / "ladder_climb"
        / "mdp"
        / "terminations.py"
    )
    spec = importlib.util.spec_from_file_location("ladder_terminations_under_test", path)
    module = importlib.util.module_from_spec(spec)
    stubs = {stub.__name__: stub for stub in (commands, managers, math_utils)}
    previous = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec.loader.exec_module(module)
    finally:
        for name, previous_module in previous.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module
    return module


_terminations = _load_terminations_module()
motion_complete = _terminations.motion_complete
stable_anchor_timeout = _terminations.stable_anchor_timeout


def _env(
    time_steps,
    *,
    total_frames=5,
    advance_reference=True,
    episode_steps=None,
    stop_holding=None,
):
    if episode_steps is None:
        episode_steps = [1] * len(time_steps)
    command = SimpleNamespace(
        time_steps=torch.tensor(time_steps),
        motion=SimpleNamespace(time_step_total=total_frames),
        cfg=SimpleNamespace(advance_reference=advance_reference),
    )
    if stop_holding is not None:
        command.stop_holding = torch.tensor(stop_holding)
    return SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command),
        episode_length_buf=torch.tensor(episode_steps),
    )


def test_motion_complete_only_times_out_at_final_frame():
    result = motion_complete(_env([0, 3, 4]), command_name="motion")

    assert result.tolist() == [False, False, True]


def test_motion_complete_is_disabled_when_reference_is_held():
    result = motion_complete(_env([4], advance_reference=False), command_name="motion")

    assert result.tolist() == [False]


def test_motion_complete_does_not_fire_during_reset():
    result = motion_complete(_env([4], episode_steps=[0]), command_name="motion")

    assert result.tolist() == [False]


def test_motion_complete_does_not_fire_during_zero_command_hold():
    result = motion_complete(
        _env([4], stop_holding=[True]), command_name="motion"
    )

    assert result.tolist() == [False]


def test_stable_anchor_timeout_exposes_command_failure_flag():
    command = SimpleNamespace(stable_anchor_timeout_failed=torch.tensor([False, True]))
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command)
    )
    assert stable_anchor_timeout(env, "motion").tolist() == [False, True]
