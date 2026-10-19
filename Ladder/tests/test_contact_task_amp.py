"""Task boundary, SSR parity and PPO+AMP regression tests without Isaac Sim."""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tensordict import TensorDict

from ladder.utils.contact_task import SupportContactPhases, support_task_plan, support_task_completion, signed_goal_progress
from ladder.utils.contact_force import filtered_surface_force_max
from ladder.rsl_rl.amp import (AMPHistory, AMPDiscriminator, AMPRunningMeanStd,
                               LadderAMPReferenceDataset, build_amp_style_observation)
from ladder.rsl_rl.algorithms.end_to_end_visual_amp_ppo import EndToEndVisualAMPPPO
from ladder.rsl_rl.runners.end_to_end_visual_amp_wrapper import AMP_STYLE_KEY
from ladder.rsl_rl.modules.transformer_vision_actor import CURRENT_PROPRIO_KEY, PROPRIO_HISTORY_KEY, DEPTH_HISTORY_KEY
from ladder.rsl_rl.modules.end_to_end_visual_actor import (
    ORACLE_CONTACT_GOAL_KEY, CONTACT_TASK_DATA_CONTRACT, BODY_GOAL_CONTACT_TASK_DATA_CONTRACT,
)


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_completion_never_waives_new_contact_or_wrong_rung():
    source = torch.tensor([[4, 4, 0, 0]]).expand(4, -1)
    target = torch.tensor([[4, 5, 0, 0]]).expand(4, -1)
    actual = torch.tensor([[4, 4, 0, 0], [-1, 5, 0, 0], [3, 5, 0, 0], [4, 5, 0, 0]])
    near = slow = torch.ones(4, 4, dtype=torch.bool)
    completed, _, _ = support_task_completion(source, target, source != target, actual, near, slow,
        torch.ones(4, dtype=torch.bool), torch.full((4,), 101), torch.ones(4, 4, dtype=torch.long),
        relaxed_after_steps=100)
    assert completed.tolist() == [False, True, False, True]
    completed, _, _ = support_task_completion(source, target, source != target, actual, near, slow,
        torch.ones(4, dtype=torch.bool), torch.full((4,), 101), torch.full((4, 4), 3), relaxed_after_steps=100)
    assert not completed[1]  # sustained old-support loss is not a sensor flicker


def test_release_task_requires_actual_release_and_progress_is_signed():
    source, target = torch.tensor([[4, 4, 0, 0]]), torch.tensor([[-1, -1, 0, 0]])
    good, _, maintained = support_task_completion(source, target, source != target, source,
        torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool), torch.tensor([True]),
        torch.tensor([200]), torch.zeros(1, 4, dtype=torch.long), relaxed_after_steps=100)
    assert not good.item()
    assert maintained.tolist() == [[False, False, True, True]]
    a, b = torch.tensor([[0.2, 1., 1., 1.]]), torch.tensor([[0.25, 1., 1., 1.]])
    mask = torch.tensor([[True, False, False, False]])
    assert signed_goal_progress(a, b, mask, mask).item() < 0
    torch.testing.assert_close(signed_goal_progress(a, b, mask, mask), -signed_goal_progress(b, a, mask, mask))


def test_progress_does_not_reward_a_closed_approach_retreat_cycle():
    mask = torch.tensor([[True, False, False, False]])
    distances = [torch.full((1, 4), value) for value in (1., .9, .8, .7, 1.)]
    total = sum(signed_goal_progress(a, b, mask, mask) for a, b in zip(distances, distances[1:]))
    torch.testing.assert_close(total, torch.zeros(1))


def _contact_task_command(*, body_goal=False, foot_safety=False):
    # Importing the full Isaac Sim command requires an Omniverse runtime.
    # Exercise its actual refresh/advance methods with deterministic sensors.
    path = Path(__file__).resolve().parents[1] / "source/ladder/ladder/tasks/ladder_climb/mdp/contact_task.py"
    command_class = next(node for node in ast.parse(path.read_text()).body
                         if isinstance(node, ast.ClassDef) and node.name == "ContactTaskCommand")
    methods = [node for node in command_class.body if isinstance(node, ast.FunctionDef)
               and node.name in ("refresh", "_update_command", "_set_task_goals", "_adaptive_sampling",
                                 "_refresh_foot_rewards")]
    from ladder.rsl_rl.amp import quat_apply_inverse
    namespace = dict(torch=torch, quat_apply_inverse=quat_apply_inverse,
                     SupportContactPhases=SupportContactPhases,
                     filtered_surface_force_max=filtered_surface_force_max,
                     support_task_completion=support_task_completion, signed_goal_progress=signed_goal_progress)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)

    class Command:
        refresh = namespace["refresh"]
        _update_command = namespace["_update_command"]
        _set_task_goals = namespace["_set_task_goals"]
        _adaptive_sampling = namespace["_adaptive_sampling"]
        _refresh_foot_rewards = namespace["_refresh_foot_rewards"]

        @property
        def source_ids(self):
            return self.sources[self.task_index]

        @property
        def target_ids(self):
            return self.targets[self.task_index]

        @property
        def active_limbs(self):
            return self.source_ids != self.target_ids

        @property
        def robot_limb_pos_w(self):
            return self.limb_pos

        @property
        def actual_stable_support_rung_id(self):
            return self.actual

        @property
        def current_expected_support_ids(self):
            return (self.motion.stable_support_rung_id[self.time_steps]
                    if self.contact_phases is None else self.contact_phases.expected)

        def _stable_support_goal_positions_for_frames(self, frames):
            return frames.float()[:, None, None].expand(-1, 4, 3) * .01

        def _stable_anchor_has_dangerous_contact(self):
            return self.danger

    c = Command()
    c._env = SimpleNamespace(common_step_counter=0, episode_length_buf=torch.zeros(2, dtype=torch.long),
                             step_dt=.02, scene=SimpleNamespace(env_origins=torch.zeros(2, 3)))
    c.cfg = SimpleNamespace(single_task_episode=False, task_limb_speed_limit=.25,
        task_min_upright=.5, task_min_root_height=.4,
        stable_anchor_root_linear_speed_threshold=.5, stable_anchor_root_angular_speed_threshold=1.5,
        stable_anchor_relaxed_after_s=2., task_settle_steps=10,
        stable_anchor_confirmation_steps=4, stable_anchor_failure_after_s=5.5,
        task_reference_duration_scale=1.25, task_deadline_extra_s=.5,
        task_failure_confirmation_steps=10, task_collision_failure_confirmation_steps=20,
        fixed_reference_frame=None, contact_task_body_goal_enabled=body_goal,
        task_body_goal_tolerance_m=.1, contact_task_foot_safety_enabled=foot_safety,
        task_foot_surface_force_threshold_n=5., task_foot_min_top_normal_fraction=.5,
        task_foot_swing_pitch_range_deg=(-55., 15.), task_foot_swing_side_tilt_limit_deg=20.,
        task_foot_swing_error_scale_deg=30.)
    c.task_anchors = torch.tensor([0, 10, 20])
    support_by_frame = torch.tensor([[-1, -1, 0, 0]] * 10 + [[1, 2, 0, 0]] * 10)
    c.motion = SimpleNamespace(stable_reset_segment_ids=torch.tensor([0] * 10 + [1] * 10),
                               stable_support_rung_id=support_by_frame)
    c.motion.body_pos_w = torch.zeros(21, 1, 3)
    c.motion.body_pos_w[:, 0, 2] = 1.
    c.motion.body_pos_w[10, 0, 0] = 1.
    c.motion.body_pos_w[20, 0, 0] = 2.
    c.motion_anchor_body_index = 0
    c.sources = torch.tensor([[1, 1, 0, 0], [1, 2, 0, 0]])
    c.targets = torch.tensor([[1, 2, 0, 0], [2, 2, 0, 0]])
    c.task_sources = c.sources
    c.task_index = torch.zeros(2, dtype=torch.long)
    c.time_steps = torch.zeros(2, dtype=torch.long)
    c.task_steps = torch.zeros(2, dtype=torch.long)
    c._reset_steps = torch.zeros(2, dtype=torch.long)
    c.confirmation = torch.zeros(2, dtype=torch.long)
    c.missing_steps = torch.zeros(2, 4, dtype=torch.long)
    c.unsafe_steps = torch.zeros(2, dtype=torch.long)
    c.danger_steps = torch.zeros(2, dtype=torch.long)
    c.task_deadline_steps = torch.zeros(2, dtype=torch.long)
    c.completed = torch.zeros(2, dtype=torch.bool)
    c.failed = torch.zeros(2, dtype=torch.bool)
    c.previous_distance = torch.zeros(2, 4)
    c.goal_pos_w = torch.zeros(2, 4, 3)
    c.body_goal_pos_w = torch.zeros(2, 3)
    c.previous_body_distance = torch.zeros(2)
    c.contact_phases = SupportContactPhases(2, "cpu") if body_goal else None
    c.limb_pos = torch.zeros(2, 4, 3)
    c.actual = c.sources[0].repeat(2, 1)
    c.danger = torch.zeros(2, dtype=torch.bool)
    c.num_envs = 2
    c.device = "cpu"
    c._fixed_start_episode = torch.zeros(2, dtype=torch.bool)
    c._secondary_fixed_start_episode = torch.zeros(2, dtype=torch.bool)
    c._uniform_start_episode = torch.zeros(2, dtype=torch.bool)
    c._current_reset_valid_fraction = torch.zeros(2)
    c.robot_anchor_pos_w = torch.tensor([[0., 0., 1.]]).repeat(2, 1)
    c.robot_anchor_quat_w = torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)
    c.robot_anchor_lin_vel_w = torch.zeros(2, 3)
    c.robot_anchor_ang_vel_w = torch.zeros(2, 3)
    c.robot = SimpleNamespace(data=SimpleNamespace(body_lin_vel_w=torch.zeros(2, 4, 3)))
    c.robot.data.body_quat_w = torch.tensor([1., 0., 0., 0.]).repeat(2, 4, 1)
    c.limb_body_indexes = [0, 1, 2, 3]
    c._limb_target_distance_thresholds = torch.full((4,), .2)
    c._control_fps = 50
    c._last_refresh_step = -1
    c._reward_values = {}
    c.metrics = {name: torch.zeros(2) for name in (
        "task_index", "task_success", "task_timeout", "task_fall", "task_collision",
        "task_completed_count", "target_distance_m", "target_contact_fraction",
        "task_elapsed_s", "task_deadline_s", "task_unsafe", "reset_frame",
        "sequence_start_task_index", "sequence_remaining_tasks",
        "sequence_completion_fraction", "sequence_success",
        "task_00_successes", "task_00_failures", "task_01_successes", "task_01_failures",
        "task_00_attempts", "task_01_attempts")}
    if body_goal:
        for name in ("contact_error", "release_events", "contact_events", "changes",
                     "new_support_loss", *SupportContactPhases.NAMES):
            c.metrics[f"phase_{name}"] = torch.zeros(2)
        for name in SupportContactPhases.NAMES:
            c.metrics[f"phase_{name}_time_s"] = torch.zeros(2)
        for name in ("body_target_distance_m", "body_goal_progress_m", "body_completion_error_m"):
            c.metrics[name] = torch.zeros(2)
    if foot_safety:
        c.foot_reward_diagnostics = {}
        c._stable_anchor_limb_contact_sensors = [
            SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(2, 1, 7, 3)))
            for _ in range(4)]
        for foot in ("left", "right"):
            for name in ("front_contact_time_s", "underside_contact_time_s",
                         "front_force_peak_n", "underside_force_peak_n", "swing_time_s",
                         "swing_toe_up_max_deg", "swing_side_tilt_max_deg",
                         "swing_toe_up_excess_time_s", "swing_pose_error_integral_s"):
                c.metrics[f"foot_{foot}_{name}"] = torch.zeros(2)
    c._set_task_goals(torch.arange(2))
    return c


def test_support_match_rewards_exact_rung_ground_and_release_ids():
    c = _contact_task_command()
    c._env.episode_length_buf[:] = 1

    def refresh_once():
        c._env.common_step_counter += 1
        c._last_refresh_step = -1
        c.refresh()
        return c._reward_values["support_match"].clone()

    c.actual[0] = c.targets[0]
    torch.testing.assert_close(refresh_once()[0], torch.tensor(1.0))

    c.actual[0] = torch.tensor([1, 7, 0, 0])
    torch.testing.assert_close(refresh_once()[0], torch.tensor(0.75))

    c.targets[0] = torch.tensor([-1, -1, 0, 0])
    c.actual[0] = torch.tensor([-1, -1, 0, 0])
    torch.testing.assert_close(refresh_once()[0], torch.tensor(1.0))
    c.actual[0, 0] = 1
    torch.testing.assert_close(refresh_once()[0], torch.tensor(0.75))


def test_event_command_confirms_once_and_does_not_advance_on_failure():
    c = _contact_task_command()
    c.actual[0] = c.target_ids[0]
    c.limb_pos.copy_(c.goal_pos_w)
    for step in range(1, 14):
        c._env.common_step_counter += 1
        c._env.episode_length_buf += 1
        c.refresh()
        c.refresh()  # reward and termination managers share one evaluation
        assert c.task_steps.tolist() == [step, step]
        assert c.time_steps.tolist() == [0, 0]
    assert c.completed.tolist() == [True, False]
    assert c._reward_values["completion"].tolist() == [50., 0.]
    c.unsafe_steps[0] = 2
    c.danger_steps[0] = 3
    c._update_command()
    c._update_command()
    assert c.task_index.tolist() == [1, 0]
    assert c.time_steps.tolist() == [10, 0]
    assert c.unsafe_steps[0] == 0 and c.danger_steps[0] == 0
    assert c.metrics["task_01_attempts"].tolist() == [1., 0.]
    c._env.common_step_counter += 1
    c._env.episode_length_buf += 1
    c.refresh()
    assert c._reward_values["completion"].sum() == 0
    assert c._reward_values["progress"][0] == 0
    for _ in range(261):
        c._env.common_step_counter += 1
        c._env.episode_length_buf += 1
        c.refresh()
        c._update_command()
    assert c.failed[1] and c.task_index[1] == 0


def test_all_frame_reset_maps_to_the_next_stable_anchor():
    c = _contact_task_command()
    env_ids = torch.arange(2)
    for frame, expected_task in ((0, 0), (7, 0), (10, 1), (19, 1)):
        c.cfg.fixed_reference_frame = frame
        c._adaptive_sampling(env_ids)
        assert c.time_steps.tolist() == [frame, frame]
        assert c.task_index.tolist() == [expected_task, expected_task]
        c._set_task_goals(env_ids)
        assert c.time_steps.tolist() == [frame, frame]
        assert c.task_deadline_steps.tolist() == [275, 275]
    c.cfg.fixed_reference_frame = 20
    with pytest.raises(ValueError, match="next stable anchor"):
        c._adaptive_sampling(env_ids)
    c.cfg.fixed_reference_frame = None
    sampled = set()
    for _ in range(100):
        c._adaptive_sampling(env_ids)
        sampled.update(c.time_steps.tolist())
        assert torch.equal(c.task_index, c.motion.stable_reset_segment_ids[c.time_steps])
    assert sampled - {0, 10}  # It is not restricted to stable anchors.


def test_sequence_keeps_robot_state_and_terminates_only_at_final_goal():
    path = Path(__file__).resolve().parents[1] / "source/ladder/ladder/tasks/ladder_climb/mdp/contact_task.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "contact_task_done")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    done = namespace["contact_task_done"]
    c = _contact_task_command()
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: c))
    for goal in (0, 1):
        c.actual[0] = c.target_ids[0]
        c.limb_pos[0] = c.goal_pos_w[0]
        before_pos = c.limb_pos.clone()
        # Reset settling only applies to the first goal. Later goals still
        # require four consecutive eligible physical samples.
        for _ in range(13 if goal == 0 else 4):
            c._env.common_step_counter += 1
            c._env.episode_length_buf += 1
            c.refresh()
        assert c.completed.tolist() == [True, False]
        assert done(env, True).tolist() == [goal == 1, False]
        assert not done(env, False).any()
        assert c.metrics["task_completed_count"][0] == goal + 1
        assert c.metrics["sequence_completion_fraction"][0] == (goal + 1) / 2
        assert c.metrics["sequence_remaining_tasks"][0] == 1 - goal
        assert c.metrics["sequence_success"][0] == (goal == 1)
        c._update_command()
        torch.testing.assert_close(c.limb_pos, before_pos)
        assert c.task_index.tolist() == [1, 0]
        assert c._env.episode_length_buf.tolist() == [13 if goal == 0 else 17] * 2
        assert c._reward_values["completion"].tolist() == [50., 0.]
    assert c.metrics["task_00_successes"].tolist() == [1., 0.]
    assert c.metrics["task_01_successes"].tolist() == [1., 0.]


def test_sequence_fraction_counts_only_goals_remaining_after_random_reset():
    c = _contact_task_command()
    c.cfg.fixed_reference_frame = 17
    c._adaptive_sampling(torch.arange(2))
    c.metrics["sequence_start_task_index"][:] = c.task_index.float()
    c._set_task_goals(torch.arange(2))
    c.actual[:] = c.target_ids
    c.limb_pos.copy_(c.goal_pos_w)
    for _ in range(13):
        c._env.common_step_counter += 1
        c._env.episode_length_buf += 1
        c.refresh()
    assert c.metrics["task_completed_count"].eq(1).all()
    assert c.metrics["sequence_completion_fraction"].eq(1).all()
    assert c.metrics["sequence_success"].eq(1).all()


def test_playback_termination_values_are_already_selected_by_environment():
    path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/play_transformer_vision_ppo.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_termination_values")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    env = SimpleNamespace(termination_manager=SimpleNamespace(
        get_active_iterable_terms=lambda env_idx: [("task_success", [env_idx == 3]),
                                                   ("task_failure", [env_idx != 3])]))
    assert namespace["_termination_values"](env, 3) == {"task_success": True, "task_failure": False}
    assert namespace["_termination_values"](env, 2) == {"task_success": False, "task_failure": True}


def test_long_reference_segment_gets_a_longer_deadline():
    c = _contact_task_command()
    c.task_anchors = torch.tensor([0, 285, 622])
    c.time_steps[:] = torch.tensor([0, 285])
    c.task_index[:] = torch.tensor([0, 1])
    c._set_task_goals(torch.arange(2))
    assert c.task_deadline_steps.tolist() == [382, 447]


def test_persistent_collision_but_not_a_brief_touch_ends_a_task():
    c = _contact_task_command()
    c.danger[:] = True
    for step in range(1, 20):
        c._env.common_step_counter = step
        c._env.episode_length_buf += 1
        c.refresh()
        assert not c.failed.any()
    c._env.common_step_counter = 20
    c._env.episode_length_buf += 1
    c.refresh()
    assert c.failed.all() and c.metrics["task_collision"].eq(1).all()
    assert c.metrics["task_timeout"].eq(0).all()


def test_actor_observation_uses_measured_current_and_next_raw_support_ids():
    path = Path(__file__).resolve().parents[1] / "source/ladder/ladder/tasks/ladder_climb/mdp/contact_task.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "contact_task_observation")
    from ladder.rsl_rl.amp import quat_apply_inverse
    namespace = {"torch": torch, "quat_apply_inverse": quat_apply_inverse}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    c = _contact_task_command()
    c.actual[:] = torch.tensor([[-1, 0, 1, 7], [7, 1, 0, -1]])
    c.time_steps[1] = 10
    c.targets[0, 0] = -1  # A release limb still receives its target-frame XYZ.
    c.effective_climb_command = torch.ones(2, 1)
    c.robot.data.body_lin_vel_w = torch.zeros(2, 4, 3)
    env = SimpleNamespace(num_envs=2, command_manager=SimpleNamespace(get_term=lambda _: c))
    obs = namespace["contact_task_observation"](env)
    assert obs.shape == (2, 57)
    torch.testing.assert_close(obs[0, :3], torch.tensor([.1, .1, -.9]))
    torch.testing.assert_close(obs[0, 36:40], torch.tensor([-1., 0., 1., 7.]))
    torch.testing.assert_close(obs[0, 40:44], torch.tensor([-1., -1., 0., 0.]))
    torch.testing.assert_close(obs[1, 40:44], torch.tensor([1., 2., 0., 0.]))
    torch.testing.assert_close(obs[0, 44:48], torch.tensor([-1., 2., 0., 0.]))


def test_reference_window_matches_policy_history_and_excludes_reset():
    history = AMPHistory(2, 71, 5, 2, "cpu")
    history.seed(torch.zeros(2, 71))
    for step in range(1, 9):
        windows, valid = history.append(torch.full((2, 71), float(step)), torch.tensor([False, step == 4]))
    assert valid.tolist() == [True, False]
    assert windows.reshape(2, 5, 71)[0, :, 0].tolist() == [0, 2, 4, 6, 8]
    assert windows.reshape(2, 5, 71)[1].min() >= 4


def test_amp_reference_holds_zero_all_velocities():
    count = 20
    q = torch.zeros(count, 5, 4); q[..., 0] = 1
    pos = torch.randn(count, 5, 3)
    motion = SimpleNamespace(_stable_support_limb_body_indexes=[1, 2, 3, 4],
        body_pos_w=pos, _body_pos_w=pos, body_quat_w=q,
        body_lin_vel_w=torch.ones_like(pos), body_ang_vel_w=torch.ones_like(pos),
        joint_pos=torch.randn(count, 25), joint_vel=torch.ones(count, 25),
        climb_direction=torch.ones(count), stable_motion_anchor_frames=torch.tensor([0, 10, 19]))
    dataset = LadderAMPReferenceDataset(SimpleNamespace(motion=motion, motion_anchor_body_index=0),
                                       anchor_hold_probability=1.)
    samples = dataset.sample(7).reshape(7, 5, 71)
    assert torch.equal(samples[:, 0], samples[:, -1])
    assert samples[:, :, 3:9].count_nonzero() == 0
    assert samples[:, :, 34:59].count_nonzero() == 0


def test_amp_feature_translation_invariance_and_normalization():
    q = torch.tensor([[1., 0, 0, 0]]).expand(4, -1)
    args = dict(anchor_pos_w=torch.randn(4, 3), anchor_quat_w=q,
                anchor_lin_vel_w=torch.randn(4, 3), anchor_ang_vel_w=torch.randn(4, 3),
                joint_pos=torch.randn(4, 25), joint_vel=torch.randn(4, 25), limb_pos_w=torch.randn(4, 4, 3))
    original = build_amp_style_observation(**args)
    args["anchor_pos_w"] += 10
    args["limb_pos_w"] += 10
    torch.testing.assert_close(original, build_amp_style_observation(**args), atol=2e-6, rtol=1e-5)
    norm = AMPRunningMeanStd(71)
    samples = torch.randn(16, 355)
    norm.update_transitions(samples)
    torch.testing.assert_close(norm.mean.float(), samples.reshape(-1, 71).mean(0), atol=1e-5, rtol=1e-4)
    assert norm.normalize_transitions(samples).dtype == torch.float32


def test_discriminator_matches_local_ssr_reward_and_gradient_penalty():
    path = Path("/home/zhangdawei/humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/rslrl-317/rsl_rl/modules/amp_discriminator.py")
    if not path.exists():
        pytest.skip("Local SSR reference not installed")
    tree = ast.parse(path.read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AMP_Discriminator")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    ssr = namespace["AMP_Discriminator"](355)
    ours = AMPDiscriminator(71, num_frames=5)
    ours.trunk.load_state_dict(ssr.linear.state_dict())
    ours.head.load_state_dict(ssr.linear_final.state_dict())
    states = torch.randn(8, 5, 71)
    torch.testing.assert_close(ours.reward(states.flatten(1)), ssr.compute_amp_reward(states))
    torch.testing.assert_close(ours.gradient_penalty(states.flatten(1)), ssr.compute_grad_penalty(states.flatten(1)))


def test_normalizer_matches_ssr_across_rolling_window_wrap():
    path = Path("/home/zhangdawei/humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/rslrl-317/rsl_rl/utils/utils.py")
    if not path.exists():
        pytest.skip("Local SSR reference not installed")
    definitions = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
                   and n.name in ("RunningMeanStd", "Normalizer")]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    ssr = namespace["Normalizer"](71, device="cpu")
    ours = AMPRunningMeanStd(71)
    for step in range(103):
        data = torch.randn(12, 71) + step * 0.01
        ssr.update(data)
        ours.update_transitions(data)
    torch.testing.assert_close(ours.mean, ssr.mean)
    torch.testing.assert_close(ours.variance, ssr.var)
    torch.testing.assert_close(ours.normalize_transitions(data), ssr.normalize(data))


def observations(history_frames=25, depth_frames=2, body_goal=False):
    history = torch.randn(4, history_frames, 85) * 0.1
    return TensorDict({CURRENT_PROPRIO_KEY: history[:, -1], PROPRIO_HISTORY_KEY: history,
        DEPTH_HISTORY_KEY: torch.rand(4, depth_frames, 60, 80), "critic": torch.randn(4, 135 if body_goal else 128),
        ORACLE_CONTACT_GOAL_KEY: torch.rand(4, 64 if body_goal else 57), AMP_STYLE_KEY: torch.randn(4, 71)}, [4])


def algorithm(history_frames=25, depth_frames=2, body_goal=False, regulation_schedule=None):
    cfg = {"num_steps_per_env": 10, "policy": {"use_contact_task": True,
        "use_contact_task_phases": body_goal, "use_contact_task_body_goal": body_goal,
        "actor_hidden_dims": [32], "critic_hidden_dims": [16], "init_noise_std": 1.},
        "algorithm": {"num_learning_epochs": 1, "num_mini_batches": 1, "sequence_micro_batch_envs": 2,
            "schedule": "fixed", "amp_replay_capacity": 64, "amp_discriminator_batch_size": 8,
            "amp_discriminator_updates": 1}}
    cfg["algorithm"]["regulation_reward_schedule"] = regulation_schedule
    env = SimpleNamespace(num_envs=4, num_actions=25, unwrapped=SimpleNamespace(step_dt=0.02),
        sample_amp_expert_transitions=lambda n: torch.randn(n, 355),
        amp_reference_dataset=SimpleNamespace(observation_dim=71, num_frames=5, history_stride=2,
                                              anchor_hold_probability=.2, balance_directions=True))
    if regulation_schedule is not None:
        terms = {name: SimpleNamespace(weight=weight) for name, weight in {
            **regulation_schedule["base_weights"], "joint_mechanical_power": -5e-5,
            "reference_joint_lower": .25, "joint_limit": -10.,
        }.items()}
        env.unwrapped.reward_manager = SimpleNamespace(
            get_term_cfg=terms.__getitem__, set_term_cfg=terms.__setitem__,
        )
    return EndToEndVisualAMPPPO.construct_algorithm(observations(history_frames, depth_frames, body_goal), env, cfg, "cpu")


REGULATION_SCHEDULE = {
    "ramp_updates": 2500, "final_multiplier": 2.5,
    "base_weights": {"action_rate_l2": -.1, "action_second_difference_l2_lower": -.05,
                     "joint_acc_l2_upper": -2.5e-7, "joint_acc_l2_lower": -1e-6,
                     "joint_torque_l2": -1e-5},
}


@pytest.mark.parametrize("elapsed,multiplier", [(0, 1.), (1, 1.0006), (1250, 1.75),
    (2499, 2.4994), (2500, 2.5), (4499, 2.5), (9000, 2.5)])
def test_regulation_ramp_starts_at_resume_and_uses_absolute_weights(elapsed, multiplier):
    baseline = algorithm()
    baseline.num_updates = 10000
    alg = algorithm(regulation_schedule=REGULATION_SCHEDULE)
    alg.load(baseline.save())
    assert alg._regulation_start_update == 10000
    assert alg._regulation_metrics["regulation_multiplier"] == 1.
    alg.num_updates += elapsed
    for _ in range(3):  # Applying again must not multiply the already-scaled weights.
        alg._apply_regulation_schedule()
        assert alg._regulation_metrics["regulation_multiplier"] == pytest.approx(multiplier)
        for name, weight in REGULATION_SCHEDULE["base_weights"].items():
            assert alg._regulation_reward_manager.get_term_cfg(name).weight == pytest.approx(weight * multiplier)
    for name, weight in (("joint_mechanical_power", -5e-5), ("reference_joint_lower", .25), ("joint_limit", -10.)):
        assert alg._regulation_reward_manager.get_term_cfg(name).weight == weight
    torch.testing.assert_close(alg._reward_group_weights(), torch.tensor([1., .2]))
    restored = algorithm(regulation_schedule=REGULATION_SCHEDULE)
    restored.load(alg.save())
    assert restored._regulation_start_update == 10000
    assert restored._regulation_metrics == alg._regulation_metrics


def test_regulation_ramp_is_fixed_within_rollout_and_logged_for_that_rollout():
    alg = algorithm(regulation_schedule=REGULATION_SCHEDULE)
    alg.num_updates = 1250
    obs = observations()
    with torch.inference_mode():
        for step in range(10):
            alg.act(obs)
            assert alg._regulation_metrics["regulation_multiplier"] == 1.75
            alg.process_env_step(obs, torch.ones(4), torch.full((4,), step == 4), {})
        alg.compute_returns(obs)
    result = alg.update()
    assert result["regulation_multiplier"] == 1.75
    assert result["regulation_elapsed_updates"] == 1250
    assert result["regulation_weight_action_rate_l2"] == pytest.approx(-.175)
    with torch.inference_mode():
        alg.act(obs)
    assert alg._regulation_metrics["regulation_multiplier"] == pytest.approx(1.7506)


def test_regulation_checkpoint_rejects_changed_or_disabled_schedule_for_training():
    alg = algorithm(regulation_schedule=REGULATION_SCHEDULE)
    checkpoint = alg.save()
    changed = deepcopy(REGULATION_SCHEDULE)
    changed["ramp_updates"] = 2000
    for schedule in (None, changed):
        restored = algorithm(regulation_schedule=schedule)
        with pytest.raises(RuntimeError, match="Regulation reward schedule differs"):
            restored.load(checkpoint)
    # Actor-only playback remains independent of the reward curriculum.
    algorithm().load(checkpoint, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False})


@pytest.mark.parametrize("all_done", [False, True])
@pytest.mark.parametrize("history_frames,depth_frames", [(25, 2), (10, 3)])
@pytest.mark.parametrize("body_goal", [False, True])
def test_ppo_amp_update_checkpoint_and_reset_only_rollout(all_done, history_frames, depth_frames, body_goal):
    alg = algorithm(history_frames, depth_frames, body_goal)
    assert alg.policy.actor_input_dim == (405 if body_goal else 398)
    assert alg.data_contract == (BODY_GOAL_CONTACT_TASK_DATA_CONTRACT if body_goal else CONTACT_TASK_DATA_CONTRACT)
    assert alg.policy.encoder.history_frames == history_frames
    assert alg.policy.encoder.depth_frames == depth_frames
    assert alg.storage.proprio_history.shape == (10, 4, history_frames, 85)
    assert alg.storage.depth_history.shape == (10, 4, depth_frames, 60, 80)
    before = deepcopy(alg.amp_discriminator.state_dict())
    obs = observations(history_frames, depth_frames, body_goal)
    # Match the actual RSL inference-mode rollout boundary.
    with torch.inference_mode():
        for step in range(10):
            alg.act(obs)
            obs = observations(history_frames, depth_frames, body_goal)
            alg.process_env_step(obs, torch.ones(4), torch.full((4,), all_done), {})
        alg.compute_returns(obs)
    assert alg.storage.rewards.shape == (10, 4, 2)
    assert alg.storage.returns.shape == (10, 4, 2)
    assert alg.storage.advantages.shape == (10, 4, 2)
    torch.testing.assert_close(alg.storage.advantages.mean(dim=(0, 1)), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(alg._reward_group_weights(), torch.tensor([1.0, 0.2]), atol=1e-6, rtol=0)
    result = alg.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    if not all_done:
        assert any(not torch.equal(before[k], v) for k, v in alg.amp_discriminator.state_dict().items())
    restored = algorithm(history_frames, depth_frames, body_goal)
    restored.load(alg.save())
    assert restored.amp_training_updates == 1
    assert restored.amp_normalizer.count == alg.amp_normalizer.count
    with pytest.raises(RuntimeError, match="data contract"):
        restored.load({**alg.save(), "data_contract": "old_oracle_v5"})


def test_real_motion_support_plan_preserves_hand_groups():
    import numpy as np
    repo = Path(__file__).resolve().parents[2]
    paths = list((repo / "Ladder/motions/zdw0824_capture1").glob("**/*visual_up_down_stop_50hz_motion.npz"))
    if not paths:
        pytest.skip("Local motion unavailable")
    with np.load(paths[0]) as motion:
        source, target, active, direction = support_task_plan(
            torch.tensor(motion["stable_support_anchor_frames"]),
            torch.tensor(motion["stable_support_rung_id"]), torch.tensor(motion["climb_direction"]).flatten())
    assert len(source) == 28
    assert (active.sum(-1) == 1).sum() == 24
    assert (active.sum(-1) == 2).sum() == 4
    assert (source[1:] == target[:-1]).all()


def _advance_contact_step(c):
    c._env.common_step_counter += 1
    c._env.episode_length_buf += 1
    c.refresh()


def test_goal_handoff_updates_phases_without_repeating_reset_settling():
    c = _contact_task_command(body_goal=True)
    c.actual[0] = c.target_ids[0]
    c.limb_pos.copy_(c.goal_pos_w)
    for _ in range(13):
        _advance_contact_step(c)
    assert c.completed[0]
    c._update_command()
    assert c.task_index.tolist() == [1, 0]
    assert c.task_steps[0] == 0
    assert c._reset_steps[0] == 13  # goal switch is not a physical reset
    for step in range(1, 5):
        _advance_contact_step(c)
        assert c.contact_phases.prepare_count[0] == step
        assert c.contact_phases.phase[0, 0] == (
            SupportContactPhases.PREPARE if step < 4 else SupportContactPhases.RELEASE)
        assert not c.completed[0]
    c.actual[0, 0] = -1
    for step in range(1, 5):
        _advance_contact_step(c)
        assert c.contact_phases.phase[0, 0] == (
            SupportContactPhases.RELEASE if step < 4 else SupportContactPhases.MOVE)
        assert c._reward_values["release_event"][0] == (0. if step < 4 else 50.)
    assert c.task_steps[0] == 8 < c.cfg.task_settle_steps
    c.limb_pos[0, 0] = c.goal_pos_w[0, 0] + .5  # remain in flight
    _advance_contact_step(c)
    assert c._reward_values["release_event"][0] == 0  # no repeated bonus


def test_reset_settling_is_per_env_and_independent_of_randomized_episode_age():
    c = _contact_task_command(body_goal=True)
    # Env0 continues its episode; env1 has just physically reset, even if RSL
    # randomizes episode_length_buf to stagger initial timeout boundaries.
    c._env.episode_length_buf[:] = torch.tensor([100, 200])
    c._reset_steps[:] = torch.tensor([100, 0])
    c.motion.stable_support_rung_id[0] = c.source_ids[0]
    c._set_task_goals(torch.arange(2))
    for step in range(1, 14):
        _advance_contact_step(c)
        c.refresh()  # managers must not age reset settling twice
        assert c._reset_steps.tolist() == [100 + step, step]
        assert c.contact_phases.prepare_count[1] == max(0, step - 9)
        if step >= 4:
            assert c.contact_phases.phase[0, 1] == SupportContactPhases.RELEASE
        assert c.contact_phases.phase[1, 1] == (
            SupportContactPhases.PREPARE if step < 13 else SupportContactPhases.RELEASE)


def test_handoff_early_flight_is_recognized_but_sensor_flicker_is_not():
    c = _contact_task_command(body_goal=True)
    c._env.episode_length_buf[:] = 100
    c._reset_steps[:] = 100
    c.task_index[:] = 1
    c.time_steps[:] = 10
    c._set_task_goals(torch.arange(2))
    c.actual.copy_(c.source_ids)
    c.actual[:, 0] = -1
    c.limb_pos[:, 0] = c.goal_pos_w[:, 0] + .5
    for step in range(1, 7):
        _advance_contact_step(c)
        assert c.contact_phases.phase[:, 0].eq(
            SupportContactPhases.PREPARE if step < 6 else SupportContactPhases.MOVE).all()
        # Early departure is recognized, not retroactively paid as a safe release.
        assert c._reward_values["release_event"].eq(0).all()
    assert c.task_steps.eq(6).all()


def test_handoff_completion_still_requires_consecutive_safe_target_contacts():
    c = _contact_task_command(body_goal=True)
    c._env.episode_length_buf[:] = 100
    c._reset_steps[:] = 100
    c.task_index[:] = 1
    c.time_steps[:] = 10
    c._set_task_goals(torch.arange(2))
    c.actual.copy_(c.target_ids)
    c.limb_pos.copy_(c.goal_pos_w)
    for step in range(1, 4):
        _advance_contact_step(c)
        assert c.confirmation.eq(step).all() and not c.completed.any()
    c.actual[:, 0] = c.source_ids[:, 0]  # breaks the consecutive target confirmation
    _advance_contact_step(c)
    assert c.confirmation.eq(0).all() and not c.completed.any()
    c.actual.copy_(c.target_ids)
    for step in range(1, 5):
        _advance_contact_step(c)
        assert c.completed.eq(step == 4).all()
    assert c._reward_values["completion"].eq(50).all()
    _advance_contact_step(c)
    assert c._reward_values["completion"].eq(0).all()


def test_body_goal_progress_is_signed_dt_correct_and_has_no_dwell_bonus():
    c = _contact_task_command(body_goal=True)
    _advance_contact_step(c)  # first post-reset physics sample seeds the distance
    rewards = []
    for x in (.2, .4, .4, .2, .0):
        c.robot_anchor_pos_w[:, 0] = x
        _advance_contact_step(c)
        rewards.append(c._reward_values["body_progress"].clone() * c._env.step_dt * .5)
    torch.testing.assert_close(rewards[0], torch.full((2,), .1))
    assert rewards[2].eq(0).all()  # staying close is not a per-step payout
    assert rewards[3].lt(0).all()
    torch.testing.assert_close(sum(rewards), torch.zeros(2), atol=1e-6, rtol=0)
    # Movements entirely within the 10 cm tolerance do not constrain the torso.
    c.robot_anchor_pos_w[:, 0] = .95
    _advance_contact_step(c)
    c.robot_anchor_pos_w[:, 0] = 1.02
    _advance_contact_step(c)
    assert c._reward_values["body_progress"].eq(0).all()


def test_body_goal_is_fixed_until_advance_and_partial_reset_has_no_jump_reward():
    c = _contact_task_command(body_goal=True)
    _advance_contact_step(c)
    old_goal = c.body_goal_pos_w.clone()
    c.time_steps[:] = 7  # a clock cannot change the fixed target
    _advance_contact_step(c)
    torch.testing.assert_close(c.body_goal_pos_w, old_goal)
    c.task_index[0] = 1
    c.time_steps[0] = 10
    c._set_task_goals(torch.tensor([0]))
    assert c.body_goal_pos_w[:, 0].tolist() == [2., 1.]
    _advance_contact_step(c)
    assert c._reward_values["body_progress"].eq(0).all()
    c.robot_anchor_pos_w[0, 0] = .1
    _advance_contact_step(c)
    torch.testing.assert_close(c._reward_values["body_progress"] * .02,
                               torch.tensor([.1, 0.]), atol=1e-6, rtol=0)


def test_body_goal_observation_is_64d_local_and_translation_invariant():
    path = Path(__file__).resolve().parents[1] / "source/ladder/ladder/tasks/ladder_climb/mdp/contact_task.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "contact_task_observation")
    from ladder.rsl_rl.amp import quat_apply_inverse
    namespace = {"torch": torch, "quat_apply_inverse": quat_apply_inverse}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    c = _contact_task_command(body_goal=True)
    c.effective_climb_command = torch.ones(2, 1)
    env = SimpleNamespace(num_envs=2, command_manager=SimpleNamespace(get_term=lambda _: c))
    obs = namespace["contact_task_observation"](env)
    assert obs.shape == (2, 64)
    torch.testing.assert_close(obs[:, -3:], torch.tensor([[1., 0., 0.]]).repeat(2, 1))
    translation = torch.tensor([8., -3., 2.])
    for value in (c.robot_anchor_pos_w, c.body_goal_pos_w, c.limb_pos, c.goal_pos_w):
        value += translation
    torch.testing.assert_close(namespace["contact_task_observation"](env), obs)
    c.robot_anchor_quat_w[:] = torch.tensor([2.**-.5, 0., 0., 2.**-.5])
    rotated = namespace["contact_task_observation"](env)
    torch.testing.assert_close(rotated[:, -3:], torch.tensor([[0., -1., 0.]]).repeat(2, 1),
                               atol=1e-6, rtol=0)


def test_body_goal_has_no_extra_completion_gate_and_reports_completion_error():
    c = _contact_task_command(body_goal=True)
    c.actual[:] = c.target_ids
    c.limb_pos.copy_(c.goal_pos_w)
    for _ in range(13):
        _advance_contact_step(c)
    assert c.completed.all()  # torso is still 1 m from target; this is soft shaping
    torch.testing.assert_close(c.metrics["body_completion_error_m"], torch.ones(2))
    old_pos = c.robot_anchor_pos_w.clone()
    c._update_command()
    assert c.task_index.tolist() == [1, 1]
    torch.testing.assert_close(c.robot_anchor_pos_w, old_pos)
    assert c.body_goal_pos_w[:, 0].tolist() == [2., 2.]
    _advance_contact_step(c)
    assert c._reward_values["body_progress"].eq(0).all()


def test_idle_phased_body_goal_times_out_instead_of_waiting_forever():
    c = _contact_task_command(body_goal=True)
    c.contact_phases.reset(torch.arange(2), c.source_ids, c.target_ids, c.source_ids)
    for _ in range(int(c.task_deadline_steps.max())):
        _advance_contact_step(c)
        c._update_command()
    assert c.failed.all() and c.metrics["task_timeout"].eq(1).all()
    assert c.task_index.eq(0).all() and not c.completed.any()
    assert c.metrics["phase_release_time_s"].gt(0).all()


def test_real_motion_all_28_goals_can_advance_without_reference_clock():
    """Scripted sensors check sequence wiring, NOT physics or learnability."""
    import numpy as np
    repo = Path(__file__).resolve().parents[2]
    paths = list((repo / "Ladder/motions/zdw0824_capture1").glob("**/*visual_up_down_stop_50hz_motion.npz"))
    if not paths:
        pytest.skip("Local motion unavailable")
    c = _contact_task_command(body_goal=True)
    with np.load(paths[0]) as motion:
        c.task_anchors = torch.tensor(motion["stable_support_anchor_frames"])
        c.motion.stable_support_rung_id = torch.tensor(motion["stable_support_rung_id"])
        c.motion.stable_reset_segment_ids = torch.tensor(motion["stable_support_reset_segment_ids"])
        c.motion.body_pos_w = torch.tensor(motion["body_pos_w"], dtype=torch.float32)
        c.motion_anchor_body_index = motion["body_names"].tolist().index("body")
        c.sources, c.targets, _, _ = support_task_plan(
            c.task_anchors, c.motion.stable_support_rung_id,
            torch.tensor(motion["climb_direction"]).flatten())
    c.task_sources = c.sources
    for index in range(len(c.sources)):
        for name in ("attempts", "successes", "failures"):
            c.metrics[f"task_{index:02d}_{name}"] = torch.zeros(2)
    c._set_task_goals(torch.arange(2))
    for index in range(28):
        assert c.task_index.eq(index).all()
        c.actual[:] = c.target_ids
        c.limb_pos.copy_(c.goal_pos_w)
        c.robot_anchor_pos_w.copy_(c.body_goal_pos_w)
        for _ in range(13 if index == 0 else 4):
            _advance_contact_step(c)
        assert c.completed.all() and not c.failed.any()
        assert c.metrics["sequence_success"].eq(index == 27).all()
        assert c.metrics["task_completed_count"].eq(index + 1).all()
        before_pose = c.robot_anchor_pos_w.clone()
        c._update_command()
        torch.testing.assert_close(c.robot_anchor_pos_w, before_pose)
    assert c.metrics["sequence_completion_fraction"].eq(1).all()
