from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "isaaclab.sim", reason="Isaac Sim runtime is required for Ladder MDP imports"
)

from ladder.robots.cr1 import CR1_ACTION_SCALE
from ladder.tasks.ladder_climb.mdp import events, observations


def test_joint_position_noise_is_applied_after_field_normalization(monkeypatch):
    joint_names = tuple(CR1_ACTION_SCALE)
    joint_count = len(joint_names)
    robot = SimpleNamespace(
        joint_names=joint_names,
        data=SimpleNamespace(
            root_ang_vel_b=torch.zeros(1, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            joint_pos=torch.zeros(1, joint_count),
            default_joint_pos=torch.zeros(1, joint_count),
            joint_vel=torch.zeros(1, joint_count),
        ),
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=SimpleNamespace(articulations={"robot": robot}),
        action_manager=SimpleNamespace(action=torch.zeros(1, joint_count)),
    )
    monkeypatch.setattr(
        observations,
        "_uniform_sensor_noise",
        lambda value, magnitude: torch.full_like(value, magnitude),
    )

    frame = observations.cr1_transformer_vision_proprio_frame(
        env,
        add_noise=True,
        root_angular_velocity_noise=0.0,
        orientation_noise_rad=0.0,
        joint_position_noise_normalized=0.01,
        joint_velocity_noise=0.0,
    )

    joint_position = frame[:, 9 : 9 + joint_count]
    torch.testing.assert_close(joint_position, torch.full_like(joint_position, 0.01))


def test_no_command_proprio_frame_keeps_only_a_zero_compatibility_pad():
    joint_names = tuple(CR1_ACTION_SCALE)
    joint_count = len(joint_names)
    robot = SimpleNamespace(
        joint_names=joint_names,
        data=SimpleNamespace(
            root_ang_vel_b=torch.zeros(2, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            joint_pos=torch.zeros(2, joint_count),
            default_joint_pos=torch.zeros(2, joint_count),
            joint_vel=torch.zeros(2, joint_count),
        ),
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene=SimpleNamespace(articulations={"robot": robot}),
        action_manager=SimpleNamespace(action=torch.zeros(2, joint_count)),
    )

    frame = observations.cr1_transformer_vision_proprio_frame(
        env,
        include_climb_command=False,
    )

    assert frame.shape == (2, 85)
    torch.testing.assert_close(frame[:, -1], torch.zeros(2))


def test_motor_strength_sample_is_shared_by_stiffness_and_damping(monkeypatch):
    class FakeImplicitActuator:
        def __init__(self):
            self.joint_indices = slice(None)
            self.stiffness = torch.full((2, 3), 80.0)
            self.damping = torch.full((2, 3), 4.0)

    class FakeAsset:
        def __init__(self):
            self.device = "cpu"
            self.num_joints = 3
            self.actuators = {"all_joints": FakeImplicitActuator()}
            self.data = SimpleNamespace(
                default_joint_stiffness=torch.full((2, 3), 80.0),
                default_joint_damping=torch.full((2, 3), 4.0),
            )
            self.written_stiffness = None
            self.written_damping = None

        def write_joint_stiffness_to_sim(self, value, joint_ids, env_ids):
            self.written_stiffness = value.clone()

        def write_joint_damping_to_sim(self, value, joint_ids, env_ids):
            self.written_damping = value.clone()

    asset = FakeAsset()

    class FakeScene:
        num_envs = 2

        def __getitem__(self, key):
            return asset

    env = SimpleNamespace(scene=FakeScene())
    samples = iter((1.2, 1.1, 0.9))
    monkeypatch.setattr(events, "ImplicitActuator", FakeImplicitActuator)
    monkeypatch.setattr(
        events,
        "sample_uniform",
        lambda lower, upper, shape, device: torch.full(shape, next(samples), device=device),
    )

    events.randomize_actuator_strength_and_gains(
        env,
        torch.tensor([1]),
        SimpleNamespace(name="robot", joint_ids=slice(None)),
        motor_strength_range=(0.8, 1.2),
        stiffness_factor_range=(0.9, 1.1),
        damping_factor_range=(0.9, 1.1),
    )

    actuator = asset.actuators["all_joints"]
    torch.testing.assert_close(actuator.stiffness[0], torch.full((3,), 80.0))
    torch.testing.assert_close(actuator.damping[0], torch.full((3,), 4.0))
    torch.testing.assert_close(actuator.stiffness[1], torch.full((3,), 80.0 * 1.2 * 1.1))
    torch.testing.assert_close(actuator.damping[1], torch.full((3,), 4.0 * 1.2 * 0.9))
    torch.testing.assert_close(asset.written_stiffness, actuator.stiffness[1:2])
    torch.testing.assert_close(asset.written_damping, actuator.damping[1:2])
