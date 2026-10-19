"""Check action delivery timing without launching Isaac Sim."""

import ast
from collections.abc import Sequence
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
BUFFER_ROOTS = [ROOT / "isaaclab_v2.1.0/source/isaaclab/isaaclab/utils/buffers"]
BUFFER_ROOTS += list(Path(sys.prefix).glob(
    "lib/python*/site-packages/isaaclab/source/isaaclab/isaaclab/utils/buffers"
))


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture(params=BUFFER_ROOTS, ids=lambda path: "reference" if ROOT in path.parents else "installed")
def action_class(request):
    # Use the real IsaacLab buffers and complete production action class. Only
    # the simulator-backed parent/action asset is replaced with a small test double.
    class JointPositionAction:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.device, self.num_envs = env.device, env.num_envs
            self._asset = env.robot
            self._joint_names = ["hip", "knee", "ankle"]
            self._joint_ids = slice(None)
            self._raw_actions = torch.zeros_like(env.robot.data.joint_pos)
            self._processed_actions = self._raw_actions.clone()

        @property
        def raw_actions(self):
            return self._raw_actions

        @property
        def processed_actions(self):
            return self._processed_actions

        def process_actions(self, actions):
            self._raw_actions[:] = actions
            self._processed_actions = actions * self.cfg.scale + self.cfg.offset

        def reset(self, env_ids=None):
            self._raw_actions[env_ids] = 0.0

    namespace = dict(torch=torch, Sequence=Sequence, JointPositionAction=JointPositionAction,
                     RateLimitedJointPositionActionCfg=SimpleNamespace, ManagerBasedEnv=object)
    for filename, name in (("circular_buffer.py", "CircularBuffer"), ("delay_buffer.py", "DelayBuffer")):
        load_definitions(request.param / filename, {name}, namespace)
    load_definitions(
        ROOT / "Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/actions.py",
        {"_resolve_joint_values", "RateLimitedJointPositionAction"}, namespace,
    )
    return namespace["RateLimitedJointPositionAction"]


def make_action(action_class, minimum=0, maximum=5, num_envs=6, physics_dt=0.001):
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.full((num_envs, 3), 0.3)))

    def set_target(target, *, joint_ids):
        robot.applied = target.clone()

    robot.set_joint_position_target = set_target
    env = SimpleNamespace(step_dt=0.02, physics_dt=physics_dt, num_envs=num_envs, device="cpu", robot=robot)
    cfg = SimpleNamespace(max_target_velocity=6.0, min_control_delay=minimum,
                          max_control_delay=maximum, scale=0.5, offset=0.3)
    action = action_class(cfg, env)
    action.reset()
    return action, env


def test_arrival_at_each_integer_millisecond_not_at_policy_ticks(action_class):
    action, env = make_action(action_class)
    lags = torch.arange(6, dtype=torch.int32)
    action._action_delay_buffer.set_time_lag(lags)
    old_target = env.robot.data.joint_pos.clone()
    for policy_step in range(3):
        raw = torch.full((6, 3), 0.08 * (policy_step + 1))
        action.process_actions(raw)
        torch.testing.assert_close(action.raw_actions, raw)
        new_target = raw * 0.5 + 0.3
        for substep in range(20):
            action.apply_actions()
            expected = torch.where((substep < lags)[:, None], old_target, new_target)
            torch.testing.assert_close(env.robot.applied, expected)
        old_target = new_target
        torch.testing.assert_close(action._action_delay_buffer.time_lags, lags)


def test_two_point_delay_switches_after_one_5ms_substep_not_one_policy_step(action_class):
    action, env = make_action(action_class, maximum=1, num_envs=2, physics_dt=0.005)
    action._action_delay_buffer.set_time_lag(torch.tensor([0, 1], dtype=torch.int32))
    previous = env.robot.data.joint_pos.clone()
    for policy_step in range(3):
        raw = torch.full((2, 3), 0.08 * (policy_step + 1))
        action.process_actions(raw)
        new = raw * 0.5 + 0.3
        for substep in range(4):
            action.apply_actions()
            expected = new.clone()
            if substep == 0:
                expected[1] = previous[1]
            torch.testing.assert_close(env.robot.applied, expected)
        previous = new


def test_rate_limit_stays_per_policy_step_with_or_without_delay(action_class):
    for maximum in (0, 5):
        action, env = make_action(action_class, minimum=maximum, maximum=maximum)
        assert (action._action_delay_buffer is None) == (maximum == 0)
        for step in range(3):
            action.process_actions(torch.ones(6, 3) * 10)
            target = torch.full((6, 3), 0.3 + (step + 1) * 0.12)
            torch.testing.assert_close(action.processed_actions, target)
            for substep in range(20):
                action.apply_actions()
                # The 6-rad/s limit must not be applied 20 extra times.
                torch.testing.assert_close(action.processed_actions, target)
                if substep >= maximum:
                    torch.testing.assert_close(env.robot.applied, target)


def test_partial_reset_clears_only_selected_histories_and_holds_new_pose(action_class):
    action, env = make_action(action_class, minimum=5, maximum=5)
    histories = [[] for _ in range(6)]
    reset_pose = env.robot.data.joint_pos.clone()
    for tick in range(100):
        if tick in (40, 80):
            ids = torch.tensor([1, 3]) if tick == 40 else torch.arange(6)
            action.reset(ids)
            # The reference command writes its pose after action.reset().
            env.robot.data.joint_pos[ids] = 0.2 + tick * 0.001
            reset_pose[ids] = env.robot.data.joint_pos[ids]
            for index in ids.tolist():
                histories[index] = []
        if tick % 20 == 0:
            action.process_actions(torch.full((6, 3), tick * 0.002))
        for index in range(6):
            histories[index].append(action.processed_actions[index].clone())
        expected = torch.stack([
            history[-6] if len(history) > 5 else reset_pose[index]
            for index, history in enumerate(histories)
        ])
        action.apply_actions()
        torch.testing.assert_close(env.robot.applied, expected)


@pytest.mark.parametrize("physics_dt,maximum", [(0.001, 5), (0.005, 1)])
def test_reset_samples_full_range_and_does_not_resample_during_episode(action_class, physics_dt, maximum):
    torch.manual_seed(42)
    action, env = make_action(action_class, num_envs=4096, physics_dt=physics_dt, maximum=maximum)
    before = action._action_delay_buffer.time_lags.clone()
    assert set(before.tolist()) == set(range(maximum + 1))
    action.process_actions(torch.zeros(4096, 3))
    for _ in range(20):
        action.apply_actions()
    torch.testing.assert_close(action._action_delay_buffer.time_lags, before)
    action.reset(torch.arange(2000))
    after = action._action_delay_buffer.time_lags
    torch.testing.assert_close(after[2000:], before[2000:])
    assert torch.any(after[:2000] != before[:2000])
    assert action._physics_steps_since_reset[:2000].tolist() == [0] * 2000
    assert action._physics_steps_since_reset[2000:].tolist() == [20] * 2096


@pytest.mark.parametrize("minimum,maximum", [(-1, 5), (5, 2)])
def test_invalid_delay_bounds_rejected(action_class, minimum, maximum):
    with pytest.raises(ValueError, match="Control delay"):
        make_action(action_class, minimum, maximum)


@pytest.mark.parametrize("lag", [None, 0, 1, 5])
@pytest.mark.parametrize("physics_dt", [0.001, 0.005])
def test_play_prints_physics_time_not_policy_time(capsys, lag, physics_dt):
    namespace = {}
    load_definitions(ROOT / "Ladder/scripts/rsl_rl/play.py", {"_print_control_delay"}, namespace)
    buffer = None if lag is None else SimpleNamespace(time_lags=[lag])
    env = SimpleNamespace(
        physics_dt=physics_dt, step_dt=0.02,
        action_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(_action_delay_buffer=buffer)),
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(time_steps=[1500])),
    )
    namespace["_print_control_delay"](env, label="TEST")
    output = capsys.readouterr().out
    assert f"control_delay_physics_steps={lag or 0}" in output
    assert f"control_delay_ms={(lag or 0) * physics_dt * 1000:.1f}" in output
