from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab.envs", reason="requires the IsaacLab runtime")

from ladder.robots.cr1 import CR1_ACTION_SCALE, CR1_ACTUATED_JOINT_NAMES
from ladder.tasks.ladder_climb.mdp.observations import cr1_ssr_proprio_frame


class _Scene(dict):
    @property
    def articulations(self):
        return self


def test_cr1_ssr_proprio_frame_includes_joint_velocity_in_fixed_order():
    batch_size = 2
    joint_pos = torch.arange(batch_size * 25, dtype=torch.float32).reshape(batch_size, 25) * 0.01
    default_joint_pos = torch.full_like(joint_pos, 0.05)
    joint_vel = joint_pos + 1.0
    actions = torch.linspace(-1.0, 1.0, batch_size * 25).reshape(batch_size, 25)
    robot = SimpleNamespace(
        joint_names=list(CR1_ACTUATED_JOINT_NAMES),
        data=SimpleNamespace(
            root_ang_vel_b=torch.full((batch_size, 3), 1.0),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch_size, 1),
            joint_pos=joint_pos,
            default_joint_pos=default_joint_pos,
            joint_vel=joint_vel,
        )
    )
    env = SimpleNamespace(
        num_envs=batch_size,
        scene=_Scene(robot=robot),
        action_manager=SimpleNamespace(action=actions),
        device="cpu",
    )

    frame = cr1_ssr_proprio_frame(env)
    joint_scale = torch.tensor([CR1_ACTION_SCALE[name] for name in CR1_ACTUATED_JOINT_NAMES])

    assert frame.shape == (batch_size, 85)
    torch.testing.assert_close(frame[:, :3], robot.data.root_ang_vel_b * 0.2)
    torch.testing.assert_close(frame[:, 9:34], (joint_pos - default_joint_pos) / joint_scale)
    torch.testing.assert_close(frame[:, 34:59], joint_vel * 0.1)
    torch.testing.assert_close(frame[:, 59:84], actions)
    torch.testing.assert_close(frame[:, 84], torch.ones(batch_size))
