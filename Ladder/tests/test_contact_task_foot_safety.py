"""Foot-only surface/pitch penalties run through the actual command methods."""

import math

import pytest
import torch

from test_contact_task_amp import _contact_task_command, _advance_contact_step
from ladder.utils.contact_task import SupportContactPhases


def _foot_task():
    c = _contact_task_command(body_goal=True, foot_safety=True)
    c.sources[0] = torch.tensor([1, 1, 1, 1])
    c.targets[0] = torch.tensor([1, 1, 2, 1])
    c.actual[:] = c.source_ids
    c.contact_phases.reset(torch.arange(2), c.source_ids, c.target_ids, c.source_ids)
    return c


def _toe_up_quat(degrees):
    angle = math.radians(degrees) / 2
    return torch.tensor([math.cos(angle), 0., -math.sin(angle), 0.])


@pytest.mark.parametrize("force,expected", [
    ((20., 0., 0.), (1., 0.)),
    ((-20., 0., 0.), (1., 0.)),
    ((0., 20., 0.), (1., 0.)),
    ((0., -20., 0.), (1., 0.)),
    ((0., 0., -20.), (0., 1.)),
    ((0., 0., 20.), (0., 0.)),
    ((4.9, 0., 0.), (0., 0.)),
])
def test_only_feet_are_penalized_and_target_rung_is_not_exempt(force, expected):
    c = _foot_task()
    # Hand side/bottom contacts cannot enter either new penalty.
    c._stable_anchor_limb_contact_sensors[0].data.force_matrix_w[..., 0] = 1000.
    c._stable_anchor_limb_contact_sensors[1].data.force_matrix_w[..., 2] = -1000.
    # Put the offending force on left foot's target rung (ID 2, column 1).
    c._stable_anchor_limb_contact_sensors[2].data.force_matrix_w[:, 0, 1] = torch.tensor(force)
    actual_before = c.actual.clone()
    c._refresh_foot_rewards(c.actual, torch.ones(2, dtype=torch.bool))
    for key, value in zip(("foot_front_contact", "foot_underside_contact"), expected):
        torch.testing.assert_close(c._reward_values[key], torch.full((2,), value))
    torch.testing.assert_close(c.actual, actual_before)


def test_surface_penalty_is_bounded_dt_scaled_and_counts_both_feet():
    c = _foot_task()
    for sensor in c._stable_anchor_limb_contact_sensors[2:]:
        sensor.data.force_matrix_w[:, 0, 0, 0] = 20.
        sensor.data.force_matrix_w[1, 0, 0, 0] = 10000.
    c._refresh_foot_rewards(c.actual, torch.ones(2, dtype=torch.bool))
    # Each foot is counted once even if force or the number of hit rungs grows.
    torch.testing.assert_close(c._reward_values["foot_front_contact"], torch.tensor([2., 2.]))
    torch.testing.assert_close(c._reward_values["foot_front_contact"] * -3. * .02,
                               torch.tensor([-.12, -.12]))
    for _ in range(4):
        c._refresh_foot_rewards(c.actual, torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(c.metrics["foot_left_front_contact_time_s"], torch.tensor([.1, .1]))
    torch.testing.assert_close(c.metrics["foot_left_front_force_peak_n"], torch.tensor([20., 10000.]))
    # Suppress only the resetting environment; keep the other one measurable.
    c._refresh_foot_rewards(c.actual, torch.tensor([False, True]))
    torch.testing.assert_close(c._reward_values["foot_front_contact"], torch.tensor([0., 2.]))
    torch.testing.assert_close(c.metrics["foot_left_front_contact_time_s"], torch.tensor([.1, .12]))


@pytest.mark.parametrize("toe_up,score", [
    (0., 0.), (10., 0.), (15., 0.), (30., .25), (45., 1.), (80., 1.),
    (-40., 0.), (-55., 0.), (-60., (5. / 30.) ** 2),
])
def test_swing_pitch_is_asymmetric_and_preserves_reference_toe_down(toe_up, score):
    c = _foot_task()
    c.contact_phases.phase[:, 2] = SupportContactPhases.MOVE
    c.actual[:, 2] = -1
    c.robot.data.body_quat_w[:, 2] = _toe_up_quat(toe_up)
    c._refresh_foot_rewards(c.actual, torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(c._reward_values["foot_swing_pose"], torch.full((2,), score), atol=1e-5, rtol=0)
    torch.testing.assert_close(c.foot_reward_diagnostics["toe_up_deg"][:, 0],
                               torch.full((2,), toe_up), atol=1e-4, rtol=0)


@pytest.mark.parametrize("phase,actual,acquired,expected", [
    (SupportContactPhases.PREPARE, 1, False, 0.),
    (SupportContactPhases.PREPARE, -1, False, .25),  # early physical lift
    (SupportContactPhases.RELEASE, 1, False, .25),  # not restricted to air time
    (SupportContactPhases.MOVE, -1, False, .25),
    (SupportContactPhases.ACQUIRE, 2, False, .25),
    (SupportContactPhases.ACQUIRE, 2, True, 0.),   # established support
    (SupportContactPhases.ACQUIRE, -1, True, .25),
])
def test_pose_penalty_covers_release_lift_and_landing_not_held_support(phase, actual, acquired, expected):
    c = _foot_task()
    c.contact_phases.phase[:, 2] = phase
    c.contact_phases.acquired[:, 2] = acquired
    c.actual[:, 2] = actual
    c.robot.data.body_quat_w[:] = _toe_up_quat(30.)  # other limbs must not contribute
    c._refresh_foot_rewards(c.actual, torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(c._reward_values["foot_swing_pose"], torch.full((2,), expected), atol=1e-5, rtol=0)


def test_pose_penalty_is_yaw_invariant_and_rejects_inverted_or_sideways_feet():
    c = _foot_task()
    c.actual[:, 2] = -1
    q = _toe_up_quat(30.)
    c.robot.data.body_quat_w[0, 2] = q
    # Pre-multiplication by 180-degree world yaw: [0, 0, 0, 1] * q.
    c.robot.data.body_quat_w[1, 2] = torch.tensor([0., -q[2], 0., q[0]])
    c._refresh_foot_rewards(c.actual, torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(c._reward_values["foot_swing_pose"], torch.tensor([.25, .25]), atol=1e-5, rtol=0)
    c.robot.data.body_quat_w[0, 2] = torch.tensor([0., 1., 0., 0.])  # upside down
    c.robot.data.body_quat_w[1, 2] = torch.tensor([2.**-.5, 2.**-.5, 0., 0.])
    c._refresh_foot_rewards(c.actual, torch.ones(2, dtype=torch.bool))
    assert torch.isfinite(c._reward_values["foot_swing_pose"]).all()
    torch.testing.assert_close(c._reward_values["foot_swing_pose"], torch.ones(2))


def test_real_refresh_penalizes_terminal_collision_and_does_not_advance_metrics_twice():
    c = _foot_task()
    c.danger[:] = True
    c.cfg.task_collision_failure_confirmation_steps = 1
    c._stable_anchor_limb_contact_sensors[2].data.force_matrix_w[:, 0, 1, 2] = -30.
    _advance_contact_step(c)
    assert c.failed.all()
    torch.testing.assert_close(c._reward_values["foot_underside_contact"], torch.ones(2))
    exposure = c.metrics["foot_left_underside_contact_time_s"].clone()
    c.refresh()
    torch.testing.assert_close(c.metrics["foot_left_underside_contact_time_s"], exposure)


def test_baseline_behavior_and_observation_do_not_depend_on_foot_safety_switch():
    baseline = _contact_task_command(body_goal=True)
    guarded = _contact_task_command(body_goal=True, foot_safety=True)
    for c in (baseline, guarded):
        c.actual[:] = c.target_ids
        c.limb_pos.copy_(c.goal_pos_w)
        for _ in range(13):
            _advance_contact_step(c)
    assert baseline.completed.all() and guarded.completed.all()
    for key, value in baseline._reward_values.items():
        torch.testing.assert_close(guarded._reward_values[key], value)
    assert all(guarded._reward_values[k].eq(0).all() for k in
               ("foot_front_contact", "foot_underside_contact", "foot_swing_pose"))


def test_reference_swing_poses_fit_the_free_pitch_and_side_tilt_band():
    import numpy as np
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    paths = list((repo / "Ladder/motions/zdw0824_capture1").glob("**/*visual_up_down_stop_50hz_motion.npz"))
    if not paths:
        pytest.skip("Local motion unavailable")
    c = _foot_task()
    c.actual[:, 2] = -1
    with np.load(paths[0]) as motion:
        names = motion["body_names"].tolist()
        for index, name in enumerate(("left_ankle_x_link", "right_ankle_x_link")):
            mask = motion["stable_support_rung_id"][:, index + 2] < 0
            quats = torch.tensor(motion["body_quat_w"][mask, names.index(name)])
            # Check extremal samples of each angle, including the actual hard
            # transitions, without relying on a single standing calibration.
            from ladder.rsl_rl.amp import quat_apply_inverse
            up = torch.zeros(quats.shape[0], 3); up[:, 2] = 1
            local = quat_apply_inverse(quats, up)
            angles = torch.rad2deg(torch.asin(local[:, :2].clamp(-1, 1)))
            candidates = torch.cat((angles.argmin(0), angles.argmax(0)))
            for q in quats[candidates]:
                c.robot.data.body_quat_w[:, 2] = q
                c._refresh_foot_rewards(c.actual, torch.ones(2, dtype=torch.bool))
                assert c._reward_values["foot_swing_pose"].eq(0).all()
