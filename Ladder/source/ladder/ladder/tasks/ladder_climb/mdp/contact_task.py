"""Event-driven limb-placement tasks with optional reward-only reference guidance."""

from __future__ import annotations

import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from .commands import LadderMotionCommand, LadderMotionCommandCfg
from ladder.utils.contact_task import (
    SupportContactPhases, support_task_plan, support_task_completion, signed_goal_progress,
)
from ladder.utils.contact_force import filtered_surface_force_max


class ContactTaskCommand(LadderMotionCommand):
    """Reuse the local asset, contact association and reset code, not its clock."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.task_anchors = self.motion.stable_motion_anchor_frames
        (self.task_sources, self.task_targets, self.task_active,
         self.task_directions) = support_task_plan(
            self.task_anchors, self.motion.stable_support_rung_id, self.motion.climb_direction)
        self.task_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.task_steps = torch.zeros_like(self.task_index)
        # Physical-reset age, independent of goal handoffs and RSL's randomized
        # initial episode_length_buf. Only _resample_command restarts settling.
        self._reset_steps = torch.zeros_like(self.task_index)
        self.confirmation = torch.zeros_like(self.task_index)
        self.missing_steps = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device)
        self.previous_distance = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.body_goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.previous_body_distance = torch.zeros(self.num_envs, device=self.device)
        self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros_like(self.completed)
        self.unsafe_steps = torch.zeros_like(self.task_index)
        self.danger_steps = torch.zeros_like(self.task_index)
        self.task_deadline_steps = torch.zeros_like(self.task_index)
        self._last_refresh_step = -1
        self._reward_values = {}
        self.contact_phases = (SupportContactPhases(
            self.num_envs, self.device, cfg.stable_anchor_confirmation_steps,
            cfg.task_release_grace_steps) if cfg.contact_task_phases_enabled else None)
        self.metrics = {name: torch.zeros(self.num_envs, device=self.device) for name in (
            "task_index", "task_success", "task_timeout", "task_fall", "task_collision",
            "task_completed_count", "target_distance_m", "target_contact_fraction",
            "task_elapsed_s", "task_deadline_s", "task_unsafe", "reset_frame",
            "sequence_start_task_index", "sequence_remaining_tasks",
            "sequence_completion_fraction", "sequence_success",
        )}
        for index in range(len(self.task_sources)):
            for name in ("attempts", "successes", "failures"):
                self.metrics[f"task_{index:02d}_{name}"] = torch.zeros(self.num_envs, device=self.device)
        if self.contact_phases is not None:
            for name in ("contact_error", "release_events", "contact_events",
                         "changes", "new_support_loss", *SupportContactPhases.NAMES):
                self.metrics[f"phase_{name}"] = torch.zeros(self.num_envs, device=self.device)
            for name in SupportContactPhases.NAMES:
                self.metrics[f"phase_{name}_time_s"] = torch.zeros(self.num_envs, device=self.device)
        if cfg.contact_task_body_goal_enabled:
            if self.contact_phases is None:
                raise ValueError("Body-goal contact task requires contact phases")
            if self._geometry_randomization_enabled:
                raise ValueError("Reference body goals currently require nominal ladder geometry")
            for name in ("body_target_distance_m", "body_goal_progress_m", "body_completion_error_m"):
                self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        if cfg.contact_task_foot_safety_enabled:
            if self.contact_phases is None:
                raise ValueError("Swing-foot penalties require contact phases")
            low, high = cfg.task_foot_swing_pitch_range_deg
            if not (-90 < low <= 0 <= high < 90
                    and 0 < cfg.task_foot_swing_side_tilt_limit_deg < 90
                    and cfg.task_foot_swing_error_scale_deg > 0
                    and 0 < cfg.task_foot_min_top_normal_fraction <= 1
                    and cfg.task_foot_surface_force_threshold_n >= 0):
                raise ValueError("Invalid foot-surface or swing-pose penalty limits")
            self.foot_reward_diagnostics = {}
            for foot in ("left", "right"):
                for name in ("front_contact_time_s", "underside_contact_time_s",
                             "front_force_peak_n", "underside_force_peak_n", "swing_time_s",
                             "swing_toe_up_max_deg", "swing_side_tilt_max_deg",
                             "swing_toe_up_excess_time_s", "swing_pose_error_integral_s"):
                    self.metrics[f"foot_{foot}_{name}"] = torch.zeros(self.num_envs, device=self.device)
        self.reference_guidance = None
        if cfg.contact_task_reference_guidance_enabled:
            from ladder.utils.reference_guidance import ContactTaskReferenceGuidance
            self.reference_guidance = ContactTaskReferenceGuidance(self)
        # Preserve terminal values for evaluation after IsaacLab auto-resets.
        self.last_episode_metrics = {name: torch.zeros(self.num_envs, device=self.device) for name in (
            "reset_frame", "sequence_start_task_index", "task_completed_count",
            "sequence_remaining_tasks", "sequence_completion_fraction", "sequence_success",
            "task_index", "task_timeout", "task_fall", "task_collision",
        )}
        if self.contact_phases is not None:
            for name in self.metrics:
                if name.startswith(("phase_", "body_", "foot_", "guidance_")):
                    self.last_episode_metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def source_ids(self):
        return self.task_sources[self.task_index]

    @property
    def target_ids(self):
        return self.task_targets[self.task_index]

    @property
    def active_limbs(self):
        return self.task_active[self.task_index]

    @property
    def current_expected_support_ids(self):
        contact_phases = getattr(self, "contact_phases", None)
        if contact_phases is not None:
            return contact_phases.expected
        return self.motion.stable_support_rung_id[self.time_steps]

    @property
    def actual_stable_support_rung_id(self):
        ids = super().actual_stable_support_rung_id
        # PhysX GPU does not provide pair-filtered forces for /World/ground.
        # Net ankle force is available; accept it as ground only near z=0
        # and only when no filtered ladder rung carries the foot.
        for limb in (2, 3):
            ground_sensor = self._stable_anchor_ground_contact_sensors[limb]
            rung_sensor = self._stable_anchor_limb_contact_sensors[limb]
            ground_force = torch.linalg.vector_norm(
                ground_sensor.data.net_forces_w, dim=-1
            ).amax(dim=-1)
            rung_force = torch.linalg.vector_norm(
                rung_sensor.data.force_matrix_w, dim=-1
            ).amax(dim=(1, 2))
            height = (self.robot_limb_pos_w[:, limb, 2]
                      - self._env.scene.env_origins[:, 2]).abs()
            on_ground = (
                (ground_force > self.cfg.contact_force_threshold)
                & (rung_force <= self.cfg.contact_force_threshold)
                & (height <= self.cfg.task_ground_height_tolerance_m)
            )
            ids[:, limb] = torch.where(on_ground, torch.zeros_like(ids[:, limb]), ids[:, limb])
        return ids

    @property
    def actual_stable_support_contacts(self):
        return self.actual_stable_support_rung_id >= 0

    @property
    def effective_climb_command(self):
        return self.task_directions[self.task_index, None]

    def _adaptive_sampling(self, env_ids):
        # Sample every nonterminal motion frame, then look up its support segment.
        if self.cfg.fixed_reference_frame is not None:
            frame = self.cfg.fixed_reference_frame
            if not 0 <= frame < self.task_anchors[-1].item():
                raise ValueError("Contact-task fixed_reference_frame must have a next stable anchor")
            self.time_steps[env_ids] = frame
        else:
            self.time_steps[env_ids] = torch.randint(
                self.task_anchors[-1].item(), (len(env_ids),), device=self.device)
        self.task_index[env_ids] = self.motion.stable_reset_segment_ids[self.time_steps[env_ids]]
        self._fixed_start_episode[env_ids] = self.time_steps[env_ids] == 0
        self._secondary_fixed_start_episode[env_ids] = False
        self._uniform_start_episode[env_ids] = True
        self._current_reset_valid_fraction[env_ids] = 1.0

    def _set_task_goals(self, env_ids):
        goals = self._stable_support_goal_positions_for_frames(self.task_anchors[self.task_index + 1])
        self.goal_pos_w[env_ids] = goals[env_ids]
        remaining_frames = self.task_anchors[self.task_index[env_ids] + 1] - self.time_steps[env_ids]
        reference_budget = torch.ceil(
            remaining_frames.float() * self.cfg.task_reference_duration_scale
            + self.cfg.task_deadline_extra_s * self._control_fps
        ).long()
        self.task_deadline_steps[env_ids] = torch.maximum(
            reference_budget,
            torch.full_like(reference_budget, round(self.cfg.stable_anchor_failure_after_s * self._control_fps)),
        )
        self.metrics["task_deadline_s"][env_ids] = self.task_deadline_steps[env_ids].float() / self._control_fps
        self.task_steps[env_ids] = 0
        self.confirmation[env_ids] = 0
        self.missing_steps[env_ids] = 0
        self.unsafe_steps[env_ids] = 0
        self.danger_steps[env_ids] = 0
        self.previous_distance[env_ids] = torch.linalg.vector_norm(
            self.robot_limb_pos_w[env_ids] - self.goal_pos_w[env_ids], dim=-1)
        if self.cfg.contact_task_body_goal_enabled:
            # Same scene-origin translation as reference foot/free-hand goals.
            # This is the configured torso link ("body"), NOT whole-body CoM.
            frames = self.task_anchors[self.task_index[env_ids] + 1]
            self.body_goal_pos_w[env_ids] = (
                self.motion.body_pos_w[frames, self.motion_anchor_body_index]
                + self._env.scene.env_origins[env_ids])
            self.previous_body_distance[env_ids] = torch.linalg.vector_norm(
                self.body_goal_pos_w[env_ids] - self.robot_anchor_pos_w[env_ids], dim=-1)
        contact_phases = getattr(self, "contact_phases", None)
        if contact_phases is not None:
            contact_phases.reset(
                env_ids, self.source_ids[env_ids], self.target_ids[env_ids],
                self.motion.stable_support_rung_id[self.time_steps[env_ids]])
        guidance = getattr(self, "reference_guidance", None)
        if guidance is not None:
            guidance.reset(env_ids, self.time_steps[env_ids])

    def _resample_command(self, env_ids):
        # The base resampler writes robot states and seeds reference buffers.
        # Its only metrics write needs this field during reset.
        self.metrics.setdefault("reset_valid_fraction", torch.zeros(self.num_envs, device=self.device))
        super()._resample_command(env_ids)
        self._reset_steps[env_ids] = 0
        self._set_task_goals(env_ids)
        self.completed[env_ids] = False
        self.failed[env_ids] = False
        self.metrics["reset_frame"][env_ids] = self.time_steps[env_ids].float()
        self.metrics["sequence_start_task_index"][env_ids] = self.task_index[env_ids].float()
        self.metrics["sequence_remaining_tasks"][env_ids] = len(self.task_sources) - self.task_index[env_ids].float()
        for value in self._reward_values.values():
            value[env_ids] = 0
        if self.cfg.contact_task_foot_safety_enabled:
            for value in self.foot_reward_diagnostics.values():
                value[env_ids] = 0
        for index in range(len(self.task_sources)):
            self.metrics[f"task_{index:02d}_attempts"][env_ids] = (self.task_index[env_ids] == index).float()

    def reset(self, env_ids=None):
        # Parent's metrics finalizer assumes tracking metrics which this task
        # deliberately does not define. Use IsaacLab CommandTerm's reset.
        from isaaclab.managers import CommandTerm
        ids = slice(None) if env_ids is None else env_ids
        for name, value in self.last_episode_metrics.items():
            value[ids] = self.metrics[name][ids]
        return CommandTerm.reset(self, env_ids)

    def _refresh_foot_rewards(self, actual, running):
        """Reward rates for feet only; raw support IDs and completion stay intact.

        Filtered forces are summed normal forces for each foot/rung pair, so
        face classification is a directional proxy at edges/multiple contacts.
        Evaluate swing intent BEFORE advancing the phase for the next step.
        """
        surface_forces = {}
        for surface in ("front", "underside"):
            surface_forces[surface] = torch.stack([
                filtered_surface_force_max(
                    sensor.data.force_matrix_w, surface=surface,
                    min_top_normal_fraction=self.cfg.task_foot_min_top_normal_fraction)
                for sensor in self._stable_anchor_limb_contact_sensors[2:4]
            ], dim=-1)
        front = (surface_forces["front"] > self.cfg.task_foot_surface_force_threshold_n) & running[:, None]
        underside = (surface_forces["underside"] > self.cfg.task_foot_surface_force_threshold_n) & running[:, None]
        # No target/source exemption: hitting ANY step's side or bottom is bad.
        self._reward_values["foot_front_contact"] = front.sum(-1).float()
        self._reward_values["foot_underside_contact"] = underside.sum(-1).float()

        phases = self.contact_phases
        swing = (self.active_limbs[:, 2:4] & running[:, None]
                 & ((actual[:, 2:4] < 0)
                    | ((phases.phase[:, 2:4] != phases.PREPARE) & ~phases.acquired[:, 2:4])))
        quats = self.robot.data.body_quat_w[:, self.limb_body_indexes[2:4]]
        up = torch.zeros_like(quats[..., :3])
        up[..., 2] = 1.0
        up_in_foot = quat_apply_inverse(quats.reshape(-1, 4), up.reshape(-1, 3)).reshape(-1, 2, 3)
        # CR1 ankle-link +X points to the toe, +Y laterally and +Z above the sole.
        # Positive toe_up means the toe is higher than the heel, independent of yaw.
        toe_up = torch.rad2deg(torch.asin(up_in_foot[..., 0].clamp(-1, 1)))
        side_tilt = torch.rad2deg(torch.asin(up_in_foot[..., 1].clamp(-1, 1)))
        low, high = self.cfg.task_foot_swing_pitch_range_deg
        scale = self.cfg.task_foot_swing_error_scale_deg
        pose_error = (((toe_up - high).clamp_min(0) / scale).square()
                      + ((low - toe_up).clamp_min(0) / scale).square()
                      + ((side_tilt.abs() - self.cfg.task_foot_swing_side_tilt_limit_deg)
                         .clamp_min(0) / scale).square()
                      + (-up_in_foot[..., 2]).clamp_min(0)).clamp_max(1)
        self._reward_values["foot_swing_pose"] = (pose_error * swing).sum(-1)
        self.foot_reward_diagnostics = {
            "front_force_n": surface_forces["front"],
            "underside_force_n": surface_forces["underside"],
            "toe_up_deg": toe_up, "side_tilt_deg": side_tilt, "swing_mask": swing,
        }
        dt = self._env.step_dt
        for index, foot in enumerate(("left", "right")):
            prefix = f"foot_{foot}_"
            for surface, contact in (("front", front), ("underside", underside)):
                self.metrics[prefix + surface + "_contact_time_s"] += contact[:, index] * dt
                key = prefix + surface + "_force_peak_n"
                self.metrics[key] = torch.maximum(
                    self.metrics[key], surface_forces[surface][:, index] * running)
            mask = swing[:, index]
            self.metrics[prefix + "swing_time_s"] += mask * dt
            self.metrics[prefix + "swing_toe_up_excess_time_s"] += (mask & (toe_up[:, index] > high)) * dt
            self.metrics[prefix + "swing_pose_error_integral_s"] += pose_error[:, index] * mask * dt
            for name, angle in (("toe_up", toe_up.clamp_min(0)), ("side_tilt", side_tilt.abs())):
                key = prefix + f"swing_{name}_max_deg"
                self.metrics[key] = torch.maximum(self.metrics[key], angle[:, index] * mask)

    def refresh(self):
        """Evaluate post-physics state once, before rewards AND terminations.

        IsaacLab advances commands after reward/reset. Caching here avoids a
        one-step delayed success reward or rewarding against the next goal.
        """
        step = self._env.common_step_counter
        if self._last_refresh_step == step:
            return
        self._last_refresh_step = step
        running = self._env.episode_length_buf > 0
        self.task_steps += running.long()
        self._reset_steps += running.long()
        reset_settled = self._reset_steps >= self.cfg.task_settle_steps
        actual = self.actual_stable_support_rung_id
        desired = self.target_ids >= 0
        active = self.active_limbs
        distance = torch.linalg.vector_norm(self.robot_limb_pos_w - self.goal_pos_w, dim=-1)
        velocity = self.robot.data.body_lin_vel_w[:, self.limb_body_indexes]
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        near = distance <= self._limb_target_distance_thresholds
        slow = speed <= self.cfg.task_limb_speed_limit
        gravity = torch.zeros_like(self.robot_anchor_pos_w)
        gravity[:, 2] = -1
        upright = -quat_apply_inverse(self.robot_anchor_quat_w, gravity)[:, 2]
        root_height = self.robot_anchor_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        danger = self._stable_anchor_has_dangerous_contact()
        fall = (upright < self.cfg.task_min_upright) | (root_height < self.cfg.task_min_root_height)
        unsafe = fall | danger
        self.unsafe_steps = torch.where(fall & running, self.unsafe_steps + 1, 0)
        self.danger_steps = torch.where(danger & running, self.danger_steps + 1, 0)
        safe = (~unsafe
                & (self.robot_anchor_lin_vel_w.norm(dim=-1) <= self.cfg.stable_anchor_root_linear_speed_threshold)
                & (self.robot_anchor_ang_vel_w.norm(dim=-1) <= self.cfg.stable_anchor_root_angular_speed_threshold))
        self.missing_steps = torch.where(actual < 0, self.missing_steps + 1, 0)
        eligible, matched, maintained = support_task_completion(
            self.source_ids, self.target_ids, active, actual, near, slow, safe,
            self.task_steps, self.missing_steps,
            relaxed_after_steps=round(self.cfg.stable_anchor_relaxed_after_s * self._control_fps))
        # Only physical resets need settling. A normal goal handoff retains
        # live contacts and still requires consecutive eligible confirmations.
        eligible &= running & reset_settled
        self.confirmation = torch.where(eligible, self.confirmation + 1, 0)
        self.completed = self.confirmation == self.cfg.stable_anchor_confirmation_steps
        timed_out = (self.task_steps >= self.task_deadline_steps) & ~self.completed
        fell = self.unsafe_steps >= self.cfg.task_failure_confirmation_steps
        collided = self.danger_steps >= self.cfg.task_collision_failure_confirmation_steps
        self.failed = timed_out | fell | collided
        correct = (actual == self.target_ids) & desired & near
        support_match = (actual == self.target_ids).float().mean(dim=-1)
        allowed = (actual < 0) | (actual == self.target_ids) | (active & (actual == self.source_ids))
        progress = signed_goal_progress(self.previous_distance, distance, active, desired)
        progress = torch.where(self.task_steps > 1, progress, torch.zeros_like(progress))
        return_progress = signed_goal_progress(self.previous_distance, distance, active, ~desired)
        return_progress = torch.where(self.task_steps > 1, return_progress, 0.0)
        self.previous_distance.copy_(distance)
        active_match = torch.where(desired, correct & slow, actual < 0)
        self._reward_values = {
            "progress": progress / self._env.step_dt,
            "reach": ((torch.exp(-distance.square() / 0.04) * active * desired).sum(-1)
                      / (active & desired).sum(-1).clamp_min(1)),
            "contact": (active_match * active).sum(-1) / active.sum(-1).clamp_min(1),
            # Exact next-stable support relation.  ID -1 is an intentional
            # release target, so an absent contact is rewarded only when the
            # target annotation also says that limb should be released.
            "support_match": support_match,
            "support_loss": ((~matched) & maintained).sum(-1).float(),
            "slip": (speed.square() * maintained * (actual >= 0)).sum(-1),
            "wrong_contact": (~allowed).sum(-1).float(),
            "danger": danger.float(),
            "completion": self.completed.float() / self._env.step_dt,
            "failure": self.failed.float() / self._env.step_dt,
        }
        if self.cfg.contact_task_body_goal_enabled:
            body_distance = torch.linalg.vector_norm(
                self.body_goal_pos_w - self.robot_anchor_pos_w, dim=-1)
            tolerance = self.cfg.task_body_goal_tolerance_m
            # A constant low weight across phases avoids both strong early
            # pulling and an artificial potential jump at RELEASE/ACQUIRE.
            body_progress = ((self.previous_body_distance - tolerance).clamp_min(0)
                             - (body_distance - tolerance).clamp_min(0))
            body_progress = torch.where(running & (self.task_steps > 1), body_progress, 0.0)
            self.previous_body_distance.copy_(body_distance)
            self._reward_values["body_progress"] = body_progress / self._env.step_dt
            self.metrics["body_target_distance_m"] = body_distance
            self.metrics["body_goal_progress_m"] += body_progress
            # Mean error over completed goals, not diluted by non-event steps.
            count = self.metrics["task_completed_count"]
            self.metrics["body_completion_error_m"] = torch.where(
                self.completed,
                (self.metrics["body_completion_error_m"] * count + body_distance) / (count + 1),
                self.metrics["body_completion_error_m"])
        if self.cfg.contact_task_foot_safety_enabled:
            self._refresh_foot_rewards(actual, running)
        guidance = getattr(self, "reference_guidance", None)
        if guidance is not None:
            guidance.evaluate(self, running & ~unsafe & ~self.failed)
        contact_phases = getattr(self, "contact_phases", None)
        if contact_phases is not None:
            phase_running = running & reset_settled & ~self.failed
            phase = contact_phases.step(actual, near, slow, safe, phase_running, self.missing_steps)
            # The old positive dwell rewards do not exist in the phased task.
            self._reward_values.pop("reach")
            self._reward_values.pop("contact")
            self._reward_values.pop("support_match")
            self._reward_values.update({
                "phase_contact_error": phase["phase_contact_error"],
                "support_loss": phase["support_loss"],
                "wrong_contact": phase["wrong_contact"],
                "slip": (speed.square() * phase["protected"] * (actual >= 0)).sum(-1),
                "release_event": phase["release_event"] / self._env.step_dt,
                "contact_event": phase["contact_event"] / self._env.step_dt,
                "return_progress": return_progress / self._env.step_dt,
            })
            self.metrics["phase_contact_error"] = phase["phase_contact_error"]
            self.metrics["phase_new_support_loss"] = phase["new_support_loss"]
            for event in ("release", "contact"):
                self.metrics[f"phase_{event}_events"] += phase[f"{event}_event"]
            self.metrics["phase_changes"] += phase["phase_changes"]
            for index, name in enumerate(SupportContactPhases.NAMES):
                self.metrics[f"phase_{name}"] = (
                    (contact_phases.phase == index) & active).sum(-1) / active.sum(-1).clamp_min(1)
                self.metrics[f"phase_{name}_time_s"] += (
                    self.metrics[f"phase_{name}"] * phase_running * self._env.step_dt)
        self.metrics["task_index"] = self.task_index.float()
        self.metrics["task_success"] = self.completed.float()
        self.metrics["task_timeout"] = timed_out.float()
        self.metrics["task_fall"] = fell.float()
        self.metrics["task_collision"] = collided.float()
        self.metrics["task_completed_count"] += self.completed.float()
        required = len(self.task_sources) - self.metrics["sequence_start_task_index"]
        self.metrics["sequence_remaining_tasks"] = required - self.metrics["task_completed_count"]
        self.metrics["sequence_completion_fraction"] = self.metrics["task_completed_count"] / required
        self.metrics["sequence_success"] = (
            self.completed & ~self.failed & (self.task_index == len(self.task_sources) - 1)
        ).float()
        self.metrics["target_distance_m"] = (distance * active * desired).sum(-1) / (active & desired).sum(-1).clamp_min(1)
        self.metrics["target_contact_fraction"] = correct.sum(-1) / desired.sum(-1).clamp_min(1)
        self.metrics["task_elapsed_s"] = self.task_steps.float() / self._control_fps
        self.metrics["task_deadline_s"] = self.task_deadline_steps.float() / self._control_fps
        self.metrics["task_unsafe"] = unsafe.float()
        for index in range(len(self.task_sources)):
            selected = self.task_index == index
            self.metrics[f"task_{index:02d}_successes"] += (selected & self.completed).float()
            self.metrics[f"task_{index:02d}_failures"] += (selected & self.failed).float()

    def _update_metrics(self):
        self.refresh()

    def _update_command(self):
        self.refresh()
        advance = torch.zeros_like(self.completed)
        if not self.cfg.single_task_episode:
            advance = self.completed & ~self.failed & (self.task_index < len(self.task_sources) - 1)
            ids = advance.nonzero().flatten()
            self.task_index[ids] += 1
            self.time_steps[ids] = self.task_anchors[self.task_index[ids]]
            self._set_task_goals(ids)
            # Consume the event once; reward values still belong to the old
            # goal until the next physics step refreshes them.
            self.completed[ids] = False
            for index in range(len(self.task_sources)):
                self.metrics[f"task_{index:02d}_attempts"] += (advance & (self.task_index == index)).float()
        guidance = getattr(self, "reference_guidance", None)
        if guidance is not None:
            guidance.advance(self, (self._env.episode_length_buf > 0) & ~advance & ~self.failed
                             & (self.unsafe_steps == 0) & (self.danger_steps == 0))


@configclass
class ContactTaskCommandCfg(LadderMotionCommandCfg):
    class_type: type = ContactTaskCommand
    single_task_episode: bool = False
    contact_task_phases_enabled: bool = False
    contact_task_body_goal_enabled: bool = False
    contact_task_foot_safety_enabled: bool = False
    contact_task_reference_guidance_enabled: bool = False
    guidance_window_frames: int = 3
    guidance_lead_frames: int = 5
    guidance_pause_decay_enabled: bool = False
    guidance_pause_grace_s: float = 0.2
    guidance_pause_fade_s: float = 0.5
    guidance_lower_body_names: tuple[str, ...] = ()
    guidance_upper_body_names: tuple[str, ...] = ()
    guidance_lower_joint_pattern: str = ""
    guidance_upper_joint_pattern: str = ""
    task_body_goal_tolerance_m: float = 0.10
    task_foot_surface_force_threshold_n: float = 5.0
    task_foot_min_top_normal_fraction: float = 0.5
    task_foot_swing_pitch_range_deg: tuple[float, float] = (-55.0, 15.0)
    task_foot_swing_side_tilt_limit_deg: float = 20.0
    task_foot_swing_error_scale_deg: float = 30.0
    task_release_grace_steps: int = 5
    task_limb_speed_limit: float = 0.25
    task_min_upright: float = 0.5
    task_min_root_height: float = 0.4
    task_ground_height_tolerance_m: float = 0.12
    task_reference_duration_scale: float = 1.25
    task_deadline_extra_s: float = 0.5
    task_settle_steps: int = 10  # physical reset only, not intermediate goal switches
    task_failure_confirmation_steps: int = 10
    task_collision_failure_confirmation_steps: int = 20


def contact_task_observation(env, command_name="motion"):
    c = env.command_manager.get_term(command_name)
    quat = c.robot_anchor_quat_w[:, None].expand(-1, 4, -1).reshape(-1, 4)
    def local(vector):
        return quat_apply_inverse(quat, vector.reshape(-1, 3)).reshape(env.num_envs, 4, 3)
    goals = local(c.goal_pos_w - c.robot_anchor_pos_w[:, None])
    positions = local(c.robot_limb_pos_w - c.robot_anchor_pos_w[:, None])
    velocities = local(c.robot.data.body_lin_vel_w[:, c.limb_body_indexes]) * 0.2
    # IDs share the motion annotation: -1=no support, 0=ground, 1..N=rungs.
    # Positions are meters; velocity is scaled by 1/5 m/s.
    contact_phases = getattr(c, "contact_phases", None)
    protected = (((~c.active_limbs) & (c.source_ids >= 0)) if contact_phases is None
                 else contact_phases.protected)
    expected_support = getattr(c, "current_expected_support_ids", None)
    if expected_support is None:
        expected_support = c.motion.stable_support_rung_id[c.time_steps]
    observation = torch.cat((goals.flatten(1).clamp(-3, 3), positions.flatten(1).clamp(-3, 3),
                      velocities.flatten(1).clamp(-3, 3),
                      c.actual_stable_support_rung_id.float(),
                      expected_support.float(),
                      c.target_ids.float(), c.active_limbs.float(),
                      protected.float(),
                      c.effective_climb_command), dim=-1)
    if contact_phases is not None:
        observation = torch.cat((observation, contact_phases.phase.float() / 3.0), dim=-1)
    if c.cfg.contact_task_body_goal_enabled:
        body_error_b = quat_apply_inverse(
            c.robot_anchor_quat_w, c.body_goal_pos_w - c.robot_anchor_pos_w)
        observation = torch.cat((observation, body_error_b.clamp(-3, 3)), dim=-1)
    return observation


def contact_task_critic(env, command_name="motion"):
    from .observations import amp_style_observation
    return torch.cat((contact_task_observation(env, command_name), amp_style_observation(env, command_name)), -1)


def contact_task_reward(env, term, command_name="motion"):
    c = env.command_manager.get_term(command_name)
    c.refresh()
    return c._reward_values[term]


def contact_task_done(env, success, command_name="motion"):
    c = env.command_manager.get_term(command_name)
    c.refresh()
    if not success:
        return c.failed
    return c.completed & ~c.failed & (c.cfg.single_task_episode | (c.task_index == len(c.task_sources) - 1))
