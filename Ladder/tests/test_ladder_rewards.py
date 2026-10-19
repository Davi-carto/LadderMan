import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _load_rewards_module():
    commands = ModuleType("ladder.tasks.ladder_climb.mdp.commands")
    commands.LadderMotionCommand = object
    managers = ModuleType("isaaclab.managers")
    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

    managers.ManagerTermBase = ManagerTermBase
    managers.RewardTermCfg = object
    managers.SceneEntityCfg = object
    sensors = ModuleType("isaaclab.sensors")
    sensors.ContactSensor = object
    math_utils = ModuleType("isaaclab.utils.math")
    math_utils.quat_error_magnitude = lambda q1, q2: torch.linalg.vector_norm(
        q1 - q2, dim=-1
    )

    path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "ladder"
        / "ladder"
        / "tasks"
        / "ladder_climb"
        / "mdp"
        / "rewards.py"
    )
    spec = importlib.util.spec_from_file_location("ladder_rewards_under_test", path)
    module = importlib.util.module_from_spec(spec)
    stubs = {stub.__name__: stub for stub in (commands, managers, sensors, math_utils)}
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


_rewards = _load_rewards_module()


@pytest.mark.parametrize("joint_ids, expected", [(slice(None), [12., 0.]), ([1], [6., 0.])])
def test_joint_mechanical_power_uses_applied_torque_abs_per_joint_and_no_dt(joint_ids, expected):
    data = SimpleNamespace(
        applied_torque=torch.tensor([[2., -3., 0.], [4., -5., 6.]]),
        computed_torque=torch.full((2, 3), 1000.),  # Must not use unclipped PD torque.
        joint_vel=torch.tensor([[3., 2., 5.], [0., 0., 0.]]),
    )
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, step_dt=.02)
    cfg = SimpleNamespace(name="robot", joint_ids=joint_ids)
    actual = _rewards.joint_mechanical_power_l1(env, cfg)
    assert actual.shape == (2,)
    torch.testing.assert_close(actual, torch.tensor(expected))
    # Return watts, not joules; only RewardManager applies the control dt.
    env.step_dt = .01
    torch.testing.assert_close(_rewards.joint_mechanical_power_l1(env, cfg), actual)


motion_joint_position_error_exp = _rewards.motion_joint_position_error_exp
motion_joint_velocity_error_deadband_huber = (
    _rewards.motion_joint_velocity_error_deadband_huber
)
action_second_difference_l2 = _rewards.action_second_difference_l2
ladder_phase_conditioned_limb_tracking_exp = (
    _rewards.ladder_phase_conditioned_limb_tracking_exp
)
stable_support_phase_conditioned_limb_tracking_exp = (
    _rewards.stable_support_phase_conditioned_limb_tracking_exp
)
next_stable_target_distance_progress = (
    _rewards.next_stable_target_distance_progress
)
stable_anchor_completion_bonus = _rewards.stable_anchor_completion_bonus
limb_swing_contact_penalty = _rewards.limb_swing_contact_penalty
limb_downward_contact_penalty = _rewards.limb_downward_contact_penalty
limb_upward_contact_force_reward = _rewards.limb_upward_contact_force_reward
ladder_surface_contact_penalty = _rewards.ladder_surface_contact_penalty
filtered_contact_penalty = _rewards.filtered_contact_penalty
filtered_contact_force_excess_penalty = _rewards.filtered_contact_force_excess_penalty
step_surface_orientation_contact_penalty = _rewards.step_surface_orientation_contact_penalty
missing_required_step_contact_penalty = _rewards.missing_required_step_contact_penalty
whole_robot_rail_force_max = _rewards.whole_robot_rail_force_max
whole_robot_rail_binary_contact_penalty = (
    _rewards.whole_robot_rail_binary_contact_penalty
)
whole_robot_rail_force_excess_penalty = (
    _rewards.whole_robot_rail_force_excess_penalty
)


class _CommandManager:
    def __init__(self, command):
        self._command = command

    def get_term(self, name):
        assert name == "motion"
        return self._command


def _contact_env(contact_indicator, forces_w_history):
    command = SimpleNamespace(
        limb_names=("left_hand", "right_hand", "left_foot", "right_foot"),
        contact_indicator=torch.tensor(contact_indicator, dtype=torch.float32),
    )
    forces_w_history = torch.tensor(forces_w_history, dtype=torch.float32)
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            net_forces_w=forces_w_history[:, 0],
            net_forces_w_history=forces_w_history,
        )
    )
    return SimpleNamespace(
        command_manager=_CommandManager(command),
        scene=SimpleNamespace(sensors={"contact_forces": sensor}),
    )


def test_motion_joint_position_error_exp_uses_selected_joint_squared_norm():
    command = SimpleNamespace(
        robot_joint_pos=torch.tensor([[1.0, 2.0, 4.0]]),
        joint_pos=torch.tensor([[0.0, 2.0, 2.0]]),
    )
    env = SimpleNamespace(command_manager=_CommandManager(command))
    asset_cfg = SimpleNamespace(joint_ids=[0, 2])

    reward = motion_joint_position_error_exp(env, "motion", temperature=0.5, asset_cfg=asset_cfg)

    expected = torch.exp(torch.tensor([-10.0]))
    assert torch.allclose(reward, expected)


def test_motion_joint_position_error_exp_rejects_nonpositive_temperature():
    command = SimpleNamespace(
        robot_joint_pos=torch.zeros((1, 1)),
        joint_pos=torch.zeros((1, 1)),
    )
    env = SimpleNamespace(command_manager=_CommandManager(command))
    asset_cfg = SimpleNamespace(joint_ids=[0])

    with pytest.raises(ValueError, match="temperature must be positive"):
        motion_joint_position_error_exp(env, "motion", temperature=0.0, asset_cfg=asset_cfg)


def test_motion_joint_velocity_error_deadband_huber_is_robust_and_averaged():
    command = SimpleNamespace(
        robot_joint_vel=torch.tensor([[0.0, 0.2, 1.2]]),
        joint_vel=torch.zeros((1, 3)),
    )
    env = SimpleNamespace(command_manager=_CommandManager(command))
    asset_cfg = SimpleNamespace(joint_ids=[0, 1, 2])

    loss = motion_joint_velocity_error_deadband_huber(
        env,
        "motion",
        deadband=0.15,
        delta=0.5,
        asset_cfg=asset_cfg,
    )

    expected = torch.tensor([(0.0 + 0.5 * 0.05**2 + 0.5 * (1.05 - 0.25)) / 3.0])
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize(
    ("deadband", "delta", "message"),
    [(-0.1, 0.5, "deadband must be nonnegative"), (0.1, 0.0, "delta must be positive")],
)
def test_motion_joint_velocity_error_deadband_huber_rejects_invalid_parameters(
    deadband, delta, message
):
    command = SimpleNamespace(
        robot_joint_vel=torch.zeros((1, 1)),
        joint_vel=torch.zeros((1, 1)),
    )
    env = SimpleNamespace(command_manager=_CommandManager(command))

    with pytest.raises(ValueError, match=message):
        motion_joint_velocity_error_deadband_huber(
            env,
            "motion",
            deadband=deadband,
            delta=delta,
            asset_cfg=SimpleNamespace(joint_ids=[0]),
        )


def test_action_second_difference_l2_skips_first_two_steps_and_selects_joints():
    action_manager = SimpleNamespace(action=torch.zeros((2, 3)))
    env = SimpleNamespace(
        action_manager=action_manager,
        num_envs=2,
        device=torch.device("cpu"),
    )
    term = action_second_difference_l2(None, env)
    asset_cfg = SimpleNamespace(joint_ids=[0, 2])

    assert torch.equal(term(env, asset_cfg), torch.zeros(2))
    action_manager.action.copy_(torch.tensor([[1.0, 100.0, 2.0], [2.0, 100.0, 1.0]]))
    assert torch.equal(term(env, asset_cfg), torch.zeros(2))
    action_manager.action.copy_(torch.tensor([[3.0, 200.0, 3.0], [3.0, 200.0, 4.0]]))

    # Selected second differences are [1, -1] and [-1, 2].  The large
    # unselected middle-joint values must not contribute.
    assert torch.equal(term(env, asset_cfg), torch.tensor([2.0, 5.0]))


def test_action_second_difference_l2_reset_clears_selected_environment_history():
    action_manager = SimpleNamespace(action=torch.zeros((2, 1)))
    env = SimpleNamespace(
        action_manager=action_manager,
        num_envs=2,
        device=torch.device("cpu"),
    )
    term = action_second_difference_l2(None, env)
    asset_cfg = SimpleNamespace(joint_ids=[0])

    term(env, asset_cfg)
    action_manager.action.fill_(1.0)
    term(env, asset_cfg)
    term.reset(torch.tensor([0]))
    action_manager.action.copy_(torch.tensor([[3.0], [3.0]]))

    loss = term(env, asset_cfg)
    assert loss[0] == 0.0
    assert loss[1] == 1.0


def test_phase_conditioned_limb_tracking_switches_between_target_and_reference():
    command = SimpleNamespace(
        cfg=SimpleNamespace(
            body_names=["left_sphere_hand_link", "right_sphere_hand_link"]
        ),
        limb_names=("left_hand", "right_hand"),
        contact_indicator=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        robot_limb_pos_w=torch.tensor(
            [
                [[0.1, 0.0, 0.0], [9.0, 0.0, 0.0]],
                [[9.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            ]
        ),
        target_contact_pos_w=torch.zeros((2, 2, 3)),
        robot_body_pos_w=torch.tensor(
            [
                [[9.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [9.0, 0.0, 0.0]],
            ]
        ),
        body_pos_relative_w=torch.zeros((2, 2, 3)),
    )
    env = SimpleNamespace(command_manager=_CommandManager(command))

    reward = ladder_phase_conditioned_limb_tracking_exp(
        env,
        "motion",
        ["left_hand", "right_hand"],
        ["left_sphere_hand_link", "right_sphere_hand_link"],
        target_temperature=0.01,
        reference_temperature=0.03,
    )

    expected = torch.tensor(
        [
            0.5
            * (
                torch.exp(torch.tensor(-1.0))
                + torch.exp(torch.tensor(-4.0 / 3.0))
            ),
            0.5
            * (
                torch.exp(torch.tensor(-1.0 / 3.0))
                + torch.exp(torch.tensor(-4.0))
            ),
        ]
    )
    assert torch.allclose(reward, expected)


def test_phase_conditioned_limb_tracking_requires_one_reference_body_per_limb():
    command = SimpleNamespace(limb_names=("left_hand", "right_hand"))
    env = SimpleNamespace(command_manager=_CommandManager(command))

    with pytest.raises(ValueError, match="one body for every limb"):
        ladder_phase_conditioned_limb_tracking_exp(
            env,
            "motion",
            ["left_hand", "right_hand"],
            ["left_sphere_hand_link"],
            target_temperature=0.01,
            reference_temperature=0.03,
        )


def test_stable_support_limb_tracking_uses_ground_aware_contract():
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=["left_ankle_x_link"]),
        limb_names=("left_foot",),
        # The legacy contract says swing, while stable support says ground.
        contact_indicator=torch.zeros((1, 1)),
        stable_support_contact_indicator=torch.ones((1, 1)),
        stable_support_rung_id=torch.zeros((1, 1), dtype=torch.long),
        robot_limb_pos_w=torch.tensor([[[0.1, 0.0, 0.0]]]),
        stable_support_target_pos_w=torch.zeros((1, 1, 3)),
        robot_body_pos_w=torch.tensor([[[9.0, 0.0, 0.0]]]),
        body_pos_relative_w=torch.zeros((1, 1, 3)),
    )
    env = SimpleNamespace(command_manager=_CommandManager(command))

    reward = stable_support_phase_conditioned_limb_tracking_exp(
        env,
        "motion",
        ["left_foot"],
        ["left_ankle_x_link"],
        target_temperature=0.01,
        reference_temperature=0.03,
    )

    assert torch.allclose(reward, torch.exp(torch.tensor([-1.0])))


def test_next_stable_target_distance_progress_resets_on_target_change():
    command = SimpleNamespace(
        next_stable_limb_target_valid=torch.tensor(
            [[True, False], [True, False]]
        ),
        robot_limb_pos_w=torch.tensor(
            [[[0.10, 0.0, 0.0], [0.0, 0.0, 0.0]],
             [[0.10, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
        next_stable_limb_target_pos_w=torch.zeros((2, 2, 3)),
        next_stable_target_frame=torch.tensor([10, 10]),
    )
    env = SimpleNamespace(
        command_manager=_CommandManager(command),
        num_envs=2,
        device=torch.device("cpu"),
        step_dt=0.02,
    )
    term = next_stable_target_distance_progress(None, env)

    assert torch.equal(term(env, "motion"), torch.zeros(2))
    command.robot_limb_pos_w[:, 0, 0] = torch.tensor([0.08, 0.14])
    assert torch.allclose(
        term(env, "motion"), torch.tensor([1.0, -2.0])
    )

    command.next_stable_target_frame[0] = 20
    command.robot_limb_pos_w[:, 0, 0] = torch.tensor([0.30, 0.14])
    assert torch.equal(term(env, "motion"), torch.zeros(2))

    term.reset(torch.tensor([1]))
    command.robot_limb_pos_w[:, 0, 0] = torch.tensor([0.28, 0.10])
    assert torch.allclose(
        term(env, "motion"), torch.tensor([1.0, 0.0])
    )


def test_stable_anchor_completion_bonus_is_one_shot_and_rate_independent():
    command = SimpleNamespace(
        stable_anchor_completion_event=torch.tensor([True, False])
    )
    env = SimpleNamespace(
        command_manager=_CommandManager(command),
        step_dt=0.02,
    )

    reward = stable_anchor_completion_bonus(env, "motion")

    # RewardManager multiplies by step_dt, producing exactly [1, 0].
    assert torch.equal(reward * env.step_dt, torch.tensor([1.0, 0.0]))


def test_limb_swing_contact_penalty_only_counts_inactive_contact():
    env = _contact_env(
        contact_indicator=[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        forces_w_history=[
            [[[6.0, 0.0, 0.0], [20.0, 0.0, 0.0]]],
            [[[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
    )
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])

    penalty = limb_swing_contact_penalty(
        env, "motion", ["left_hand", "right_hand"], sensor_cfg, force_threshold=5.0
    )

    assert torch.allclose(penalty, torch.tensor([0.5, 0.0]))


def test_limb_downward_contact_penalty_detects_negative_world_z_force():
    env = _contact_env(
        contact_indicator=[[0.0, 0.0, 0.0, 0.0]],
        forces_w_history=[
            [
                [[0.0, 0.0, -6.0], [0.0, 0.0, 30.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, -4.0]],
            ]
        ],
    )
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])

    penalty = limb_downward_contact_penalty(env, sensor_cfg, force_threshold=5.0)

    assert torch.allclose(penalty, torch.tensor([0.5]))


def test_limb_upward_contact_force_reward_is_phase_gated_and_saturated():
    env = _contact_env(
        contact_indicator=[
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        forces_w_history=[
            [
                [[0.0, 0.0, 5.0], [0.0, 0.0, 50.0]],
                [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
            ],
            [
                [[0.0, 0.0, 27.5], [0.0, 0.0, 100.0]],
                [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
            ],
            [
                [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
                [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
            ],
        ],
    )
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])

    reward = limb_upward_contact_force_reward(
        env,
        "motion",
        ["left_hand", "right_hand"],
        sensor_cfg,
        force_threshold=5.0,
        force_saturation=50.0,
    )

    assert torch.allclose(reward, torch.tensor([0.5, 0.5, 0.0]))


def test_split_upward_force_rewards_equal_combined_reward():
    env = _contact_env(
        contact_indicator=[
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        forces_w_history=[
            [[[0.0, 0.0, 27.5], [0.0, 0.0, 50.0]]],
            [[[0.0, 0.0, 27.5], [0.0, 0.0, 100.0]]],
            [[[0.0, 0.0, 100.0], [0.0, 0.0, 27.5]]],
        ],
    )
    combined_sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
    left_sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0])
    right_sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[1])

    combined = limb_upward_contact_force_reward(
        env,
        "motion",
        ["left_hand", "right_hand"],
        combined_sensor_cfg,
        force_threshold=5.0,
        force_saturation=50.0,
    )
    left = limb_upward_contact_force_reward(
        env,
        "motion",
        ["left_hand"],
        left_sensor_cfg,
        force_threshold=5.0,
        force_saturation=50.0,
        normalization_limb_names=["left_hand", "right_hand"],
    )
    right = limb_upward_contact_force_reward(
        env,
        "motion",
        ["right_hand"],
        right_sensor_cfg,
        force_threshold=5.0,
        force_saturation=50.0,
        normalization_limb_names=["left_hand", "right_hand"],
    )

    assert torch.allclose(left + right, combined)


def test_limb_upward_force_reward_ignores_unfiltered_side_contact():
    command = SimpleNamespace(
        limb_names=("left_hand", "right_hand"),
        contact_indicator=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            # The total force includes all colliders and represents a side-board hit.
            net_forces_w=torch.tensor([[[0.0, 0.0, 50.0]], [[0.0, 0.0, 50.0]]]),
            force_matrix_w=torch.tensor(
                [
                    [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                    [[[0.0, 0.0, 10.0], [0.0, 0.0, 17.5]]],
                ]
            ),
        )
    )
    env = SimpleNamespace(
        command_manager=_CommandManager(command),
        scene=SimpleNamespace(sensors={"right_hand_step_contacts": sensor}),
    )

    reward = limb_upward_contact_force_reward(
        env,
        "motion",
        ["right_hand"],
        SimpleNamespace(name="right_hand_step_contacts", body_ids=[0]),
        force_threshold=5.0,
        force_saturation=50.0,
        normalization_limb_names=["left_hand", "right_hand"],
        filtered_contacts_only=True,
    )

    assert torch.allclose(reward, torch.tensor([0.0, 0.5]))


def test_limb_upward_force_reward_uses_only_indicator_target_rung():
    command = SimpleNamespace(
        limb_names=("right_hand",),
        contact_indicator=torch.ones((2, 1), dtype=torch.float32),
        contact_rung_id=torch.tensor([[2], [2]]),
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            net_forces_w=torch.zeros((2, 1, 3)),
            force_matrix_w=torch.tensor(
                [
                    [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 27.5], [0.0, 0.0, 50.0]]],
                    [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 50.0]]],
                ]
            ),
        )
    )
    env = SimpleNamespace(
        command_manager=_CommandManager(command),
        scene=SimpleNamespace(sensors={"right_hand_step_contacts": sensor}),
    )

    reward = limb_upward_contact_force_reward(
        env,
        "motion",
        ["right_hand"],
        SimpleNamespace(name="right_hand_step_contacts", body_ids=[0]),
        force_threshold=5.0,
        force_saturation=50.0,
        filtered_contacts_only=True,
        target_rung_only=True,
    )

    assert torch.allclose(reward, torch.tensor([0.5, 0.0]))


def test_filtered_contact_force_excess_penalty_is_surface_specific_and_quadratic():
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=torch.tensor(
                [
                    [[[0.0, 0.0, 30.0], [40.0, 0.0, 0.0], [0.0, 0.0, -50.0]]],
                    [[[0.0, 0.0, 70.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                ]
            )
        )
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"step_contacts": sensor}))
    sensor_cfg = SimpleNamespace(name="step_contacts", body_ids=[0])

    top = filtered_contact_force_excess_penalty(
        env,
        sensor_cfg,
        force_soft=50.0,
        force_scale=20.0,
        surface="top",
    )
    front = filtered_contact_force_excess_penalty(
        env,
        sensor_cfg,
        force_soft=20.0,
        force_scale=20.0,
        surface="front",
    )

    assert torch.allclose(top, torch.tensor([0.0, 1.0]))
    assert torch.allclose(front, torch.tensor([1.0, 0.0]))


def test_whole_robot_rail_penalties_aggregate_all_body_sensors_per_rail():
    sensor_a = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=torch.tensor(
                [
                    [[[30.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                    [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                ]
            )
        )
    )
    sensor_b = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=torch.tensor(
                [
                    [[[10.0, 0.0, 0.0], [0.0, 0.0, 105.0]]],
                    [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                ]
            )
        )
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"rail_a": sensor_a, "rail_b": sensor_b})
    )
    sensor_names = ("rail_a", "rail_b")

    force_max = whole_robot_rail_force_max(env, sensor_names, expected_rail_count=2)
    left_binary = whole_robot_rail_binary_contact_penalty(
        env, sensor_names, rail_index=0, force_threshold=5.0
    )
    right_excess = whole_robot_rail_force_excess_penalty(
        env,
        sensor_names,
        rail_index=1,
        force_soft=5.0,
        force_scale=50.0,
        max_excess=3.0,
    )

    assert torch.allclose(force_max, torch.tensor([[30.0, 105.0], [0.0, 0.0]]))
    assert torch.allclose(left_binary, torch.tensor([1.0, 0.0]))
    assert torch.allclose(right_excess, torch.tensor([4.0, 0.0]))


def test_filtered_contact_force_excess_penalty_filters_rung_scope_and_caps_outlier():
    command = SimpleNamespace(
        limb_names=("right_hand",),
        contact_indicator=torch.tensor([[1.0], [1.0]]),
        contact_rung_id=torch.tensor([[1], [1]]),
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=torch.tensor(
                [
                    [[[0.0, 0.0, 220.0], [0.0, 0.0, 280.0], [0.0, 0.0, 0.0]]],
                    [[[0.0, 0.0, 1000.0], [0.0, 0.0, 100.0], [0.0, 0.0, 0.0]]],
                ]
            )
        )
    )
    env = SimpleNamespace(
        command_manager=_CommandManager(command),
        scene=SimpleNamespace(sensors={"step_contacts": sensor}),
    )
    sensor_cfg = SimpleNamespace(name="step_contacts", body_ids=[0])

    target = filtered_contact_force_excess_penalty(
        env,
        sensor_cfg,
        force_soft=180.0,
        force_scale=100.0,
        surface="top",
        max_excess=3.0,
        command_name="motion",
        limb_name="right_hand",
        rung_scope="target",
    )
    non_target = filtered_contact_force_excess_penalty(
        env,
        sensor_cfg,
        force_soft=20.0,
        force_scale=50.0,
        surface="top",
        max_excess=3.0,
        command_name="motion",
        limb_name="right_hand",
        rung_scope="non_target",
    )

    assert torch.allclose(target, torch.tensor([1.0, 0.0]))
    assert torch.allclose(non_target, torch.tensor([9.0, 9.0]))


def test_ladder_surface_contact_penalty_allows_step_contact_only_at_required_target():
    command = SimpleNamespace(
        limb_names=("left_hand",),
        contact_indicator=torch.tensor([[1.0], [1.0], [0.0]]),
        robot_limb_pos_w=torch.tensor(
            [
                [[0.00, 0.0, 0.0]],
                [[0.08, 0.0, 0.0]],
                [[0.00, 0.0, 0.0]],
            ]
        ),
        target_contact_pos_w=torch.zeros((3, 1, 3)),
    )
    force_matrix_w = torch.tensor(
        [
            [[[[6.0, 0.0, 0.0]]]],
            [[[[6.0, 0.0, 0.0]]]],
            [[[[6.0, 0.0, 0.0]]]],
        ]
    ).squeeze(1)
    sensor = SimpleNamespace(data=SimpleNamespace(force_matrix_w=force_matrix_w))
    env = SimpleNamespace(
        command_manager=_CommandManager(command),
        scene=SimpleNamespace(sensors={"step_contacts": sensor}),
    )
    sensor_cfg = SimpleNamespace(name="step_contacts", body_ids=[0])

    penalty = ladder_surface_contact_penalty(
        env,
        "motion",
        "left_hand",
        sensor_cfg,
        force_threshold=5.0,
        allow_target_contact=True,
        target_distance_threshold=0.06,
    )

    assert torch.allclose(penalty, torch.tensor([0.0, 1.0, 1.0]))


def test_ladder_surface_contact_penalty_always_counts_side_board_contact():
    command = SimpleNamespace(
        limb_names=("right_hand",),
        contact_indicator=torch.tensor([[1.0]]),
        robot_limb_pos_w=torch.zeros((1, 1, 3)),
        target_contact_pos_w=torch.zeros((1, 1, 3)),
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(force_matrix_w=torch.tensor([[[[6.0, 0.0, 0.0]]]]))
    )
    env = SimpleNamespace(
        command_manager=_CommandManager(command),
        scene=SimpleNamespace(sensors={"side_contacts": sensor}),
    )

    penalty = ladder_surface_contact_penalty(
        env,
        "motion",
        "right_hand",
        SimpleNamespace(name="side_contacts", body_ids=[0]),
        force_threshold=5.0,
        allow_target_contact=False,
    )

    assert torch.allclose(penalty, torch.tensor([1.0]))


def test_filtered_contact_penalty_counts_only_forces_above_threshold():
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=torch.tensor(
                [
                    [[[[0.0, 0.0, 4.0]]]],
                    [[[[0.0, 0.0, 6.0]]]],
                ]
            ).squeeze(1)
        )
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"ground_contacts": sensor}))

    penalty = filtered_contact_penalty(
        env,
        SimpleNamespace(name="ground_contacts", body_ids=[0]),
        force_threshold=5.0,
    )

    assert torch.allclose(penalty, torch.tensor([0.0, 1.0]))


def test_step_surface_orientation_penalties_allow_top_and_split_front_from_underside():
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=torch.tensor(
                [
                    [[[0.0, 0.0, 10.0]]],
                    [[[10.0, 0.0, 0.0]]],
                    [[[0.0, 0.0, -10.0]]],
                    [[[6.0, 0.0, 8.0]]],
                    [[[8.0, 0.0, 4.0]]],
                ]
            )
        )
    )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"step_contacts": sensor}))
    sensor_cfg = SimpleNamespace(name="step_contacts", body_ids=[0])

    front = step_surface_orientation_contact_penalty(
        env,
        sensor_cfg,
        force_threshold=5.0,
        min_top_normal_fraction=0.5,
        surface="front",
    )
    underside = step_surface_orientation_contact_penalty(
        env,
        sensor_cfg,
        force_threshold=5.0,
        min_top_normal_fraction=0.5,
        surface="underside",
    )

    assert torch.allclose(front, torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(underside, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))


def test_missing_required_step_contact_waits_for_phase_and_reset_grace():
    contact_history = torch.ones((20, 1), dtype=torch.float32)
    command = SimpleNamespace(
        limb_names=("right_hand",),
        contact_indicator=torch.ones((5, 1), dtype=torch.float32),
        time_steps=torch.tensor([10, 10, 3, 10, 10]),
        motion=SimpleNamespace(contact_indicator=contact_history),
        robot_limb_pos_w=torch.tensor(
            [
                [[0.04, 0.0, 0.0]],
                [[0.04, 0.0, 0.0]],
                [[0.04, 0.0, 0.0]],
                [[0.04, 0.0, 0.0]],
                [[0.09, 0.0, 0.0]],
            ]
        ),
        target_contact_pos_w=torch.zeros((5, 1, 3)),
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=torch.tensor(
                [
                    [[[0.0, 0.0, 0.0]]],
                    [[[0.0, 0.0, 6.0]]],
                    [[[0.0, 0.0, 0.0]]],
                    [[[0.0, 0.0, 0.0]]],
                    [[[0.0, 0.0, 0.0]]],
                ]
            )
        )
    )
    env = SimpleNamespace(
        command_manager=_CommandManager(command),
        episode_length_buf=torch.tensor([20, 20, 20, 5, 20]),
        scene=SimpleNamespace(sensors={"step_contacts": sensor}),
    )

    penalty = missing_required_step_contact_penalty(
        env,
        "motion",
        "right_hand",
        SimpleNamespace(name="step_contacts", body_ids=[0]),
        force_threshold=5.0,
        target_distance_threshold=0.08,
        grace_steps=5,
    )

    assert torch.allclose(penalty, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))

    penalty_without_distance_gate = missing_required_step_contact_penalty(
        env,
        "motion",
        "right_hand",
        SimpleNamespace(name="step_contacts", body_ids=[0]),
        force_threshold=5.0,
        target_distance_threshold=0.08,
        grace_steps=5,
        require_near_target=False,
    )

    assert torch.allclose(
        penalty_without_distance_gate,
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0]),
    )


def test_missing_required_step_contact_can_require_target_rung_force():
    command = SimpleNamespace(
        limb_names=("right_hand",),
        contact_indicator=torch.ones((2, 1), dtype=torch.float32),
        contact_rung_id=torch.ones((2, 1), dtype=torch.long),
        time_steps=torch.zeros(2, dtype=torch.long),
        robot_limb_pos_w=torch.zeros((2, 1, 3)),
        target_contact_pos_w=torch.zeros((2, 1, 3)),
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=torch.tensor(
                [
                    [[[0.0, 0.0, 0.0], [0.0, 0.0, 6.0], [0.0, 0.0, 0.0]]],
                    [[[0.0, 0.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                ]
            )
        )
    )
    env = SimpleNamespace(
        command_manager=_CommandManager(command),
        episode_length_buf=torch.ones(2, dtype=torch.long),
        scene=SimpleNamespace(sensors={"step_contacts": sensor}),
    )

    penalty = missing_required_step_contact_penalty(
        env,
        "motion",
        "right_hand",
        SimpleNamespace(name="step_contacts", body_ids=[0]),
        force_threshold=5.0,
        target_distance_threshold=0.08,
        grace_steps=0,
        target_rung_only=True,
    )

    assert torch.allclose(penalty, torch.tensor([0.0, 1.0]))
