"""Phase-bounded reference rewards, including actual command refresh timing."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ladder.utils.contact_task import SupportContactPhases
from ladder.utils.reference_guidance import (
    ContactTaskReferenceGuidance, PhaseReferenceClock, REFERENCE_TERMS,
    reference_reward_candidates, select_reference_frame,
)
from test_contact_task_amp import _contact_task_command, _advance_contact_step


ROOT = Path(__file__).resolve().parents[2]
ANCHORS = [0, 285, 373, 456, 514, 607, 686, 765, 824, 916, 996, 1077, 1142,
           1237, 1500, 1837, 1935, 2001, 2092, 2165, 2287, 2357, 2434, 2510,
           2597, 2664, 2729, 2794, 3060]


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _synthetic():
    ids = torch.tensor([[5, 6, 1, 1]] * 10 + [[5, 6, -1, 1]] * 10 + [[5, 6, 2, 1]] * 11)
    clock = PhaseReferenceClock(torch.tensor([0, 30]), ids, 2)
    phases = SupportContactPhases(2, "cpu")
    phases.reset(torch.arange(2), ids[0].repeat(2, 1), ids[-1].repeat(2, 1), ids[0].repeat(2, 1))
    return clock, phases


def test_release_leads_physical_detachment_but_cannot_run_to_landing():
    clock, phases = _synthetic()
    phases.phase[:, 2] = phases.RELEASE
    for _ in range(40):
        clock.advance(torch.zeros(2, dtype=torch.long), phases, torch.ones(2, dtype=torch.bool), 1.)
    # The target can lift five frames before actual release; never reaches landing at 20.
    torch.testing.assert_close(clock.frame, torch.full((2,), 15.))
    frames, valid = clock.candidates(torch.zeros(2, dtype=torch.long), phases)
    assert frames[valid].max() == 15
    phases.phase[:, 2] = phases.MOVE
    clock.advance(torch.zeros(2, dtype=torch.long), phases, torch.ones(2, dtype=torch.bool), 1.)
    assert clock.frame.eq(16).all()


def test_prepare_holds_source_and_acquired_support_cannot_match_flight():
    clock, phases = _synthetic()
    task = torch.zeros(2, dtype=torch.long)
    for _ in range(20):
        clock.advance(task, phases, torch.ones(2, dtype=torch.bool), 1.)
    assert clock.frame.eq(9).all()
    phases.phase[:, 2] = phases.ACQUIRE
    phases.acquired[:, 2] = True
    clock.advance(task, phases, torch.ones(2, dtype=torch.bool), 1.)
    assert clock.frame.eq(20).all()
    frames, valid = clock.candidates(task, phases)
    assert frames[valid].min() == 20


def test_three_frame_matching_window_does_not_change_release_lead():
    clock, phases = _synthetic()
    phases.phase[:, 2] = phases.MOVE
    clock.frame[:] = 15
    frames, valid = clock.candidates(torch.zeros(2, dtype=torch.long), phases)
    assert frames[0].tolist() == [12, 13, 14, 15, 16, 17, 18]
    assert valid.all()
    assert clock.lead == 5


def test_one_frame_for_all_rewards_and_empty_window_is_zero():
    frames = torch.tensor([[9, 10], [20, 21]])
    rewards = torch.tensor([[[.9, .1], [.4, .8]], [[1., 1.], [1., 1.]]])
    errors = 1 - rewards
    selected, values, err, valid = select_reference_frame(
        frames, torch.tensor([[True, True], [False, False]]), rewards, errors, torch.ones(2))
    assert selected.tolist() == [10, -1]
    torch.testing.assert_close(values, torch.tensor([[.4, .8], [0., 0.]]))
    assert valid.tolist() == [True, False]
    assert err[1].eq(0).all()


def test_matching_frame_does_not_rewind_clock_and_partial_reset_is_isolated():
    clock, phases = _synthetic()
    clock.frame[:] = 12
    phases.phase[:, 2] = phases.RELEASE
    before = clock.frame.clone()
    frames, valid = clock.candidates(torch.zeros(2, dtype=torch.long), phases)
    rewards = torch.zeros(*frames.shape, 1)
    rewards[:, 0] = 1
    select_reference_frame(frames, valid, rewards, rewards, torch.ones(1))
    torch.testing.assert_close(clock.frame, before)
    clock.reset(torch.tensor([0]), torch.tensor([27]))
    assert clock.frame.tolist() == [27, 12]


def test_teacher_kernel_sum_mean_and_swing_mask():
    body = torch.tensor([[[.01, .04, .09, .16]], [[0., 0., 0., 0.]]])
    joint = torch.tensor([[[.1, .2, .3, .4]], [[0., 0., 0., 0.]]])
    rewards, errors = reference_reward_candidates(body, joint, torch.zeros(2, 1),
        [0, 1], [2, 3], [0, 1], [2, 3], [2, 3], torch.tensor([[True, False], [False, False]]))
    torch.testing.assert_close(rewards[0, 0, 0], torch.exp(torch.tensor(-.025 / .18**2)))
    torch.testing.assert_close(rewards[0, 0, 2], torch.exp(torch.tensor(-.3 / .8)))
    torch.testing.assert_close(rewards[0, 0, 4], torch.exp(torch.tensor(-.09 / .02)))
    assert rewards[1, 0, 4] == 0  # no constant bonus when there is no swing foot
    assert errors[1].eq(0).all()


@pytest.fixture(scope="module")
def motion():
    paths = list((ROOT / "Ladder/motions/zdw0824_capture1").glob("**/*visual_up_down_stop_50hz_motion.npz"))
    if not paths:
        pytest.skip("Local motion unavailable")
    with np.load(paths[0]) as data:
        return {k: data[k].copy() for k in data.files}


def test_all_real_random_reset_frames_have_compatible_reference(motion):
    ids = torch.tensor(motion["stable_support_rung_id"], dtype=torch.long)
    anchors = torch.tensor(ANCHORS)
    frames = torch.arange(anchors[-1])
    task = torch.searchsorted(anchors[1:], frames, right=True)
    phases = SupportContactPhases(len(frames), "cpu")
    phases.reset(frames, ids[anchors[task]], ids[anchors[task + 1]], ids[frames])
    clock = PhaseReferenceClock(anchors, ids, len(frames))
    clock.reset(frames, frames)
    candidates, valid = clock.candidates(task, phases)
    assert valid.any(-1).all()
    assert valid[:, clock.window].all()  # random reset's exact reference frame is allowed
    assert (candidates[valid] >= 0).all()


def test_all_28_real_transitions_can_progress_including_sequential_hand_release(motion):
    ids = torch.tensor(motion["stable_support_rung_id"], dtype=torch.long)
    anchors = torch.tensor(ANCHORS)
    task = torch.arange(28)
    clock = PhaseReferenceClock(anchors, ids, 28)
    clock.reset(task, anchors[:-1])
    phases = SupportContactPhases(28, "cpu")
    phases.reset(task, ids[anchors[:-1]], ids[anchors[1:]], ids[anchors[:-1]])
    missing = torch.zeros(28, 4, dtype=torch.long)
    for _ in range(400):
        actual = ids[clock.frame.long()]
        missing = torch.where(actual < 0, missing + 1, 0)
        phases.step(actual, torch.ones(28, 4, dtype=torch.bool), torch.ones(28, 4, dtype=torch.bool),
                    torch.ones(28, dtype=torch.bool), torch.ones(28, dtype=torch.bool), missing)
        previous = clock.frame.clone()
        clock.advance(task, phases, torch.ones(28, dtype=torch.bool), 1.)
        assert (clock.frame >= previous).all()
        assert (clock.frame <= anchors[1:]).all()
    torch.testing.assert_close(clock.frame, anchors[1:].float())


def _guided_command(*, pause_decay=False):
    c = _contact_task_command(body_goal=True, foot_safety=True)
    c._geometry_randomization_enabled = False
    c.task_anchors = torch.tensor([0, 30, 60])
    ids = torch.tensor([[5, 6, 1, 1]] * 10 + [[5, 6, -1, 1]] * 10
                       + [[5, 6, 2, 1]] * 20 + [[5, 6, 2, -1]] * 10 + [[5, 6, 2, 2]] * 11)
    c.motion.stable_support_rung_id = ids
    c.motion.stable_reset_segment_ids = torch.tensor([0] * 30 + [1] * 30 + [1])
    c.sources, c.targets = ids[c.task_anchors[:-1]], ids[c.task_anchors[1:]]
    c.task_sources = c.sources
    c.cfg.body_names = ["body", "left_ankle_x_link", "right_ankle_x_link"]
    c.cfg.guidance_lower_body_names = tuple(c.cfg.body_names[1:])
    c.cfg.guidance_upper_body_names = ("body",)
    c.cfg.guidance_lower_joint_pattern = ".*_knee_joint"
    c.cfg.guidance_upper_joint_pattern = "waist_y_joint"
    c.cfg.guidance_window_frames = 3
    c.cfg.guidance_lead_frames = 5
    c.cfg.guidance_pause_decay_enabled = pause_decay
    c.cfg.guidance_pause_grace_s = .2
    c.cfg.guidance_pause_fade_s = .5
    c.cfg.stable_anchor_failure_after_s = 10.
    c.cfg.task_reference_duration_scale, c.cfg.task_deadline_extra_s = 2., 1.
    c.motion.body_pos_w = torch.zeros(61, 3, 3)
    c.motion.body_pos_w[..., 2] = 1.
    c.motion.body_pos_w[:, 1, 0] = torch.linspace(0, .6, 61)
    c.motion.body_quat_w = torch.tensor([1., 0., 0., 0.]).repeat(61, 3, 1)
    c.motion.joint_pos = torch.zeros(61, 3)
    c.motion.fps = 50.
    c.robot.joint_names = ["right_knee_joint", "waist_y_joint", "left_knee_joint"]
    c.robot_body_pos_w = c.motion.body_pos_w[0].repeat(2, 1, 1)
    c.robot_joint_pos = torch.zeros(2, 3)
    c._env.cfg = SimpleNamespace(rewards=SimpleNamespace(**{
        n: SimpleNamespace(weight=w) for n, w in zip(REFERENCE_TERMS, [.15, .15, .25, .15, .30, .10])}))
    c.reference_guidance = ContactTaskReferenceGuidance(c)
    c._set_task_goals(torch.arange(2))
    c.actual.copy_(c.source_ids)
    return c


def test_runtime_rewards_cached_and_clock_advanced_once_after_reward():
    c = _guided_command()
    old_time = c.time_steps.clone()
    _advance_contact_step(c)
    assert c.reference_guidance.clock.frame.eq(0).all()  # refresh only evaluates
    before = c._reward_values["reference_body_lower"].clone()
    c.refresh()
    torch.testing.assert_close(c._reward_values["reference_body_lower"], before)
    c._update_command()
    assert c.reference_guidance.clock.frame.eq(1).all()
    c._update_command()
    assert c.reference_guidance.clock.frame.eq(1).all()
    torch.testing.assert_close(c.time_steps, old_time)
    assert c.reference_guidance.lower_joints == [0, 2]  # names, not a contiguous slice
    assert c.task_deadline_steps.eq(500).all()


def test_no_compatible_window_masks_all_terms_and_does_not_fail_task():
    c = _guided_command()
    c.reference_guidance.clock.frame[:] = 0
    c.contact_phases.phase[:, 2] = SupportContactPhases.ACQUIRE
    c.contact_phases.acquired[:, 2] = True
    _advance_contact_step(c)
    assert all(c._reward_values[n].eq(0).all() for n in REFERENCE_TERMS)
    assert not c.failed.any()
    assert c.metrics["guidance_conflict_time_s"].gt(0).all()
    c._update_command()
    assert c.reference_guidance.clock.frame.ge(20).all()  # align forward to actual phase


def test_reward_values_dt_scaling_and_reset_do_not_leak_to_other_env():
    c = _guided_command()
    c.reference_guidance.evaluate(c, torch.tensor([True, False]))
    assert all(c._reward_values[n][1] == 0 for n in REFERENCE_TERMS)
    rate = sum(c._reward_values[n] * getattr(c._env.cfg.rewards, n).weight for n in REFERENCE_TERMS)
    torch.testing.assert_close(c.metrics["guidance_weighted_reward_integral"], rate * .02)
    c.reference_guidance.clock.frame[:] = 12
    c.time_steps[0] = 25
    c._set_task_goals(torch.tensor([0]))
    assert c.reference_guidance.clock.frame.tolist() == [25, 12]
    assert c.reference_guidance.diagnostics["valid"][0] == 0


def test_alignment_is_env_translation_invariant_and_zero_at_matching_pose():
    c = _guided_command()
    c._env.scene.env_origins[1] = torch.tensor([3., 4., 5.])
    c.robot_anchor_pos_w[1] += c._env.scene.env_origins[1]
    c.robot_body_pos_w[1] += c._env.scene.env_origins[1]
    c.reference_guidance.evaluate(c, torch.ones(2, dtype=torch.bool))
    for name in REFERENCE_TERMS:
        torch.testing.assert_close(c._reward_values[name][0], c._reward_values[name][1])
    for name in REFERENCE_TERMS[:4]:
        torch.testing.assert_close(c._reward_values[name], torch.ones(2))


def test_reference_holding_does_not_reset_or_extend_goal_deadline():
    c = _guided_command()
    c.contact_phases.phase[:, 2] = SupportContactPhases.RELEASE
    for _ in range(499):
        _advance_contact_step(c)
        c._update_command()
        assert not c.failed.any()
    assert c.reference_guidance.clock.frame.eq(15).all()
    _advance_contact_step(c)
    assert c.failed.all()
    assert c.metrics["task_timeout"].eq(1).all()
    assert c.task_steps.eq(500).all()


def test_goal_success_pays_old_reference_then_resets_new_clock_in_place():
    c = _guided_command()
    c.actual.copy_(c.target_ids)
    c.limb_pos.copy_(c.goal_pos_w)
    for _ in range(13):
        _advance_contact_step(c)
    assert c.completed.all()
    values = {n: c._reward_values[n].clone() for n in REFERENCE_TERMS}
    pose = c.robot_body_pos_w.clone()
    c._update_command()
    assert c.task_index.eq(1).all()
    assert c.reference_guidance.clock.frame.eq(30).all()
    assert c.task_steps.eq(0).all()
    torch.testing.assert_close(c.robot_body_pos_w, pose)
    for n in REFERENCE_TERMS:
        torch.testing.assert_close(c._reward_values[n], values[n])


def test_handoff_release_updates_reference_without_another_reset_wait():
    c = _guided_command(pause_decay=True)
    c.actual.copy_(c.target_ids)
    c.limb_pos.copy_(c.goal_pos_w)
    for _ in range(13):
        _advance_contact_step(c)
    c._update_command()
    assert c.task_index.eq(1).all() and c._reset_steps.eq(13).all()
    # In the new goal the right foot can prepare/release immediately; the
    # first goal's physical reset has already settled.
    for _ in range(4):
        _advance_contact_step(c)
        c._update_command()
    assert c.contact_phases.phase[:, 3].eq(SupportContactPhases.RELEASE).all()
    c.actual[:, 3] = -1
    for _ in range(4):
        _advance_contact_step(c)
        c._update_command()
    assert c.task_steps.eq(8).all()
    assert c.contact_phases.phase[:, 3].eq(SupportContactPhases.MOVE).all()
    assert c.reference_guidance.clock.frame.ge(40).all()  # reference release starts at40
    _advance_contact_step(c)
    assert c.reference_guidance.diagnostics["valid"].all()
    assert c.reference_guidance.diagnostics["matched_frame"].ge(40).all()
    assert c._reward_values["reference_swing_foot"].gt(0).all()


def test_phase_intersection_conflict_never_synthesizes_different_limb_frames():
    clock, phases = _synthetic()
    # A contrived simultaneous transition: one acquired limb requires >=20,
    # another preparing limb forbids >9. No shared whole-body pose exists.
    phases.source[:, 3], phases.target[:, 3] = 1, 2
    clock.upper[:, 3, phases.PREPARE] = 9
    phases.phase[:, 2] = phases.ACQUIRE
    phases.acquired[:, 2] = True
    clock.frame[:] = 15
    _, valid = clock.candidates(torch.zeros(2, dtype=torch.long), phases)
    assert not valid.any()
    before = clock.frame.clone()
    clock.advance(torch.zeros(2, dtype=torch.long), phases, torch.ones(2, dtype=torch.bool), 1.)
    torch.testing.assert_close(clock.frame, before)


def test_new_variant_config_is_isolated_and_has_exact_weights():
    path = ROOT / "Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/contact_task_env_cfg.py"
    source = ast.parse(path.read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "CR1ReferenceGuidedContactSequenceAMPEnvCfg")
    # Static contract check complements (does not replace) actual config import.
    assignments = {ast.unparse(n.targets[0]): n.value for n in ast.walk(cls) if isinstance(n, ast.Assign)}
    assert ast.literal_eval(assignments["self.episode_length_s"]) == 320.
    weights = [ast.literal_eval(assignments[f"self.rewards.{n}"].args[1]) for n in REFERENCE_TERMS]
    assert weights == [.15, .15, .25, .15, .30, .10]
    durations = torch.diff(torch.tensor(ANCHORS)) / 50
    assert torch.maximum(torch.tensor(10.), 2 * durations + 1).sum() < 320


@pytest.mark.parametrize("dt,steps,expected", [
    (.02, 0, 1.), (.02, 10, 1.), (.02, 11, .96), (.02, 20, .6),
    (.02, 34, .04), (.02, 35, 0.), (.02, 500, 0.),
    (.01, 20, 1.), (.01, 45, .5), (.01, 70, 0.),
])
def test_pause_fade_scales_six_rates_after_selection_and_dt_once(dt, steps, expected):
    c = _guided_command(pause_decay=True)
    c._env.step_dt = dt
    g = c.reference_guidance
    g.pause_steps[0] = steps
    g.evaluate(c, torch.ones(2, dtype=torch.bool))
    # Equal states, only env0 has waited. Matching/error diagnostics unaffected.
    assert g.diagnostics["matched_frame"][0] == g.diagnostics["matched_frame"][1]
    for name in REFERENCE_TERMS:
        torch.testing.assert_close(c._reward_values[name][0], c._reward_values[name][1] * expected)
        assert c.metrics[f"guidance_{name}_error"][0] == c.metrics[f"guidance_{name}_error"][1]
    torch.testing.assert_close(g.diagnostics["reward_scale"], torch.tensor([expected, 1.]))
    torch.testing.assert_close(c.metrics["guidance_weighted_reward_integral"],
                               g.diagnostics["raw_weighted_rate"] * torch.tensor([expected, 1.]) * dt)
    torch.testing.assert_close(c.metrics["guidance_raw_weighted_reward_integral"],
                               g.diagnostics["raw_weighted_rate"] * dt)


def test_paused_old_variant_keeps_original_reward():
    c = _guided_command()
    g = c.reference_guidance
    g.pause_steps[:] = 500
    g.evaluate(c, torch.ones(2, dtype=torch.bool))
    assert g.diagnostics["reward_scale"].eq(1).all()
    torch.testing.assert_close(g.diagnostics["weighted_rate"], g.diagnostics["raw_weighted_rate"])


def test_pause_fade_holds_zero_without_shortening_exploration_deadline():
    c = _guided_command(pause_decay=True)
    c.contact_phases.phase[:, 2] = SupportContactPhases.RELEASE
    for _ in range(100):
        _advance_contact_step(c)
        c._update_command()
    assert c.reference_guidance.clock.frame.eq(15).all()
    before = c.metrics["guidance_weighted_reward_integral"].clone()
    for _ in range(399):
        _advance_contact_step(c)
        c._update_command()
        assert not c.failed.any()
        assert all(c._reward_values[n].eq(0).all() for n in REFERENCE_TERMS)
        assert c._reward_values["phase_contact_error"].gt(0).all()
    torch.testing.assert_close(c.metrics["guidance_weighted_reward_integral"], before)
    _advance_contact_step(c)
    assert c.failed.all() and c.task_steps.eq(500).all()


def test_contact_flicker_does_not_renew_but_confirmed_release_advances_and_renews():
    c = _guided_command(pause_decay=True)
    g = c.reference_guidance
    c.contact_phases.phase[:, 2] = SupportContactPhases.RELEASE
    for _ in range(60):
        _advance_contact_step(c)
        c._update_command()
    assert g.diagnostics["reward_scale"].eq(0).all()
    # Less than four samples of lost contact must not refresh the grace.
    for actual in (-1, -1, 1):
        c.actual[:, 2] = actual
        _advance_contact_step(c)
        c._update_command()
        assert g.diagnostics["reward_scale"].eq(0).all()
        assert g.clock.frame.eq(15).all()
    c.actual[:, 2] = -1
    for _ in range(4):
        _advance_contact_step(c)
        c._update_command()
    assert g.clock.frame.gt(15).all()
    assert g.pause_steps.eq(0).all()
    _advance_contact_step(c)
    assert g.diagnostics["reward_scale"].eq(1).all()


def test_unsafe_pause_recovery_does_not_refresh_grace_and_advance_is_idempotent():
    c = _guided_command(pause_decay=True)
    g = c.reference_guidance
    c.contact_phases.phase[:, 2] = SupportContactPhases.RELEASE
    g.clock.frame[:] = 15
    c.task_steps[:] = 1
    c._env.episode_length_buf[:] = 1
    for _ in range(40):
        c._env.common_step_counter += 1
        g.advance(c, torch.zeros(2, dtype=torch.bool))
    assert g.pause_steps.eq(40).all()
    g.advance(c, torch.ones(2, dtype=torch.bool))
    assert g.pause_steps.eq(40).all()  # same physics step, no double aging
    c._env.common_step_counter += 1
    g.advance(c, torch.ones(2, dtype=torch.bool))
    g.evaluate(c, torch.ones(2, dtype=torch.bool))
    assert g.pause_steps.eq(41).all()
    assert g.diagnostics["reward_scale"].eq(0).all()


def test_pause_age_clears_only_reset_env_and_goal_handoff():
    c = _guided_command(pause_decay=True)
    g = c.reference_guidance
    g.pause_steps[:] = 100
    c.time_steps[0] = 25
    c._set_task_goals(torch.tensor([0]))
    assert g.pause_steps.tolist() == [0, 100]
    c.actual.copy_(c.target_ids)
    c.limb_pos.copy_(c.goal_pos_w)
    for _ in range(13):
        _advance_contact_step(c)
    assert c.completed.all()
    c._update_command()
    assert c.task_index.eq(1).all()
    assert g.pause_steps.eq(0).all()
    assert c.task_steps.eq(0).all()


def test_pause_decay_variant_changes_only_one_environment_field():
    path = ROOT / "Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/contact_task_env_cfg.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
               and n.name == "CR1PauseDecayReferenceContactSequenceAMPEnvCfg")
    assert ast.unparse(cls.bases[0]) == "CR1ReferenceGuidedContactSequenceAMPEnvCfg"
    assignments = {ast.unparse(n.targets[0]): ast.literal_eval(n.value)
                   for n in ast.walk(cls) if isinstance(n, ast.Assign)}
    assert assignments == {"self.commands.motion.guidance_pause_decay_enabled": True}


def test_uniform_reference_weight_scaling_preserves_shared_frame_and_pause_decay():
    baseline = _guided_command(pause_decay=True)
    doubled = _guided_command(pause_decay=True)
    for name in REFERENCE_TERMS:
        getattr(doubled._env.cfg.rewards, name).weight *= 2.0
    doubled.reference_guidance = ContactTaskReferenceGuidance(doubled)
    for pause_steps in (0, 10, 20, 35, 50):
        for command in (baseline, doubled):
            command.reference_guidance.pause_steps[:] = pause_steps
            command.reference_guidance.evaluate(command, torch.ones(2, dtype=torch.bool))
        before, after = baseline.reference_guidance, doubled.reference_guidance
        torch.testing.assert_close(after.diagnostics["matched_frame"], before.diagnostics["matched_frame"])
        torch.testing.assert_close(after.diagnostics["reward_scale"], before.diagnostics["reward_scale"])
        torch.testing.assert_close(after.diagnostics["weighted_rate"], before.diagnostics["weighted_rate"] * 2)
        for name in REFERENCE_TERMS:
            torch.testing.assert_close(doubled._reward_values[name], baseline._reward_values[name])
    assert doubled.reference_guidance.diagnostics["reward_scale"].eq(0).all()
