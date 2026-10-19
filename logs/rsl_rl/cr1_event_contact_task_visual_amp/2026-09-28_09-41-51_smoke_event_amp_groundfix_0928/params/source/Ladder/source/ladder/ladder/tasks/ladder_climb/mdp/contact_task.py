"""Event-driven limb-placement tasks using reference poses only for reset/AMP."""

from __future__ import annotations

import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from .commands import LadderMotionCommand, LadderMotionCommandCfg
from ladder.utils.contact_task import support_task_plan, support_task_completion, signed_goal_progress


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
        self.confirmation = torch.zeros_like(self.task_index)
        self.missing_steps = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device)
        self.previous_distance = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros_like(self.completed)
        self.unsafe_steps = torch.zeros_like(self.task_index)
        self._last_refresh_step = -1
        self._reward_values = {}
        self.metrics = {name: torch.zeros(self.num_envs, device=self.device) for name in (
            "task_index", "task_success", "task_timeout", "task_completed_count",
            "target_distance_m", "target_contact_fraction", "task_elapsed_s", "task_unsafe",
        )}
        for index in range(len(self.task_sources)):
            for name in ("attempts", "successes", "failures"):
                self.metrics[f"task_{index:02d}_{name}"] = torch.zeros(self.num_envs, device=self.device)

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

    @property
    def joint_vel(self):
        return torch.zeros_like(self.motion.joint_vel[self.time_steps])

    @property
    def body_lin_vel_w(self):
        return torch.zeros_like(self.motion.body_lin_vel_w[self.time_steps])

    @property
    def body_ang_vel_w(self):
        return torch.zeros_like(self.motion.body_ang_vel_w[self.time_steps])

    def _adaptive_sampling(self, env_ids):
        # Uniform stable-start tasks. No random point inside a swing phase.
        if self.cfg.fixed_reference_frame is not None:
            matches = torch.nonzero(self.task_anchors[:-1] == self.cfg.fixed_reference_frame).flatten()
            if matches.numel() != 1:
                raise ValueError("Contact-task fixed_reference_frame must be a nonterminal stable anchor")
            self.task_index[env_ids] = matches[0]
        else:
            self.task_index[env_ids] = torch.randint(len(self.task_sources), (len(env_ids),), device=self.device)
        self.time_steps[env_ids] = self.task_anchors[self.task_index[env_ids]]
        self._fixed_start_episode[env_ids] = self.time_steps[env_ids] == 0
        self._secondary_fixed_start_episode[env_ids] = False
        self._uniform_start_episode[env_ids] = True
        self._current_reset_valid_fraction[env_ids] = 1.0

    def _set_task_goals(self, env_ids):
        goals = self._stable_support_goal_positions_for_frames(self.task_anchors[self.task_index + 1])
        self.goal_pos_w[env_ids] = goals[env_ids]
        # References are used to seed physically meaningful reset poses only.
        self.time_steps[env_ids] = self.task_anchors[self.task_index[env_ids]]
        self.task_steps[env_ids] = 0
        self.confirmation[env_ids] = 0
        self.missing_steps[env_ids] = 0
        self.previous_distance[env_ids] = torch.linalg.vector_norm(
            self.robot_limb_pos_w[env_ids] - self.goal_pos_w[env_ids], dim=-1)

    def _resample_command(self, env_ids):
        # The base resampler writes robot states and seeds reference buffers.
        # Its only metrics write needs this field during reset.
        self.metrics.setdefault("reset_valid_fraction", torch.zeros(self.num_envs, device=self.device))
        super()._resample_command(env_ids)
        self._set_task_goals(env_ids)
        self.completed[env_ids] = False
        self.failed[env_ids] = False
        self.unsafe_steps[env_ids] = 0
        for value in self._reward_values.values():
            value[env_ids] = 0
        for index in range(len(self.task_sources)):
            self.metrics[f"task_{index:02d}_attempts"][env_ids] = (self.task_index[env_ids] == index).float()

    def reset(self, env_ids=None):
        # Parent's metrics finalizer assumes tracking metrics which this task
        # deliberately does not define. Use IsaacLab CommandTerm's reset.
        from isaaclab.managers import CommandTerm
        return CommandTerm.reset(self, env_ids)

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
        unsafe = (upright < self.cfg.task_min_upright) | (root_height < self.cfg.task_min_root_height) | danger
        self.unsafe_steps = torch.where(unsafe & running, self.unsafe_steps + 1, 0)
        safe = (~unsafe
                & (self.robot_anchor_lin_vel_w.norm(dim=-1) <= self.cfg.stable_anchor_root_linear_speed_threshold)
                & (self.robot_anchor_ang_vel_w.norm(dim=-1) <= self.cfg.stable_anchor_root_angular_speed_threshold))
        self.missing_steps = torch.where(actual < 0, self.missing_steps + 1, 0)
        eligible, matched, maintained = support_task_completion(
            self.source_ids, self.target_ids, active, actual, near, slow, safe,
            self.task_steps, self.missing_steps,
            relaxed_after_steps=round(self.cfg.stable_anchor_relaxed_after_s * self._control_fps))
        # Let reset contacts settle before accepting a task as completed.
        eligible &= running & (self.task_steps >= self.cfg.task_settle_steps)
        self.confirmation = torch.where(eligible, self.confirmation + 1, 0)
        self.completed = self.confirmation == self.cfg.stable_anchor_confirmation_steps
        self.failed = ((self.task_steps >= round(self.cfg.stable_anchor_failure_after_s * self._control_fps))
                       & ~self.completed) | (self.unsafe_steps >= self.cfg.task_failure_confirmation_steps)
        correct = (actual == self.target_ids) & desired & near
        allowed = (actual < 0) | (actual == self.target_ids) | (active & (actual == self.source_ids))
        progress = signed_goal_progress(self.previous_distance, distance, active, desired)
        progress = torch.where(self.task_steps > 1, progress, torch.zeros_like(progress))
        self.previous_distance.copy_(distance)
        active_match = torch.where(desired, correct & slow, actual < 0)
        self._reward_values = {
            "progress": progress / self._env.step_dt,
            "reach": ((torch.exp(-distance.square() / 0.04) * active * desired).sum(-1)
                      / (active & desired).sum(-1).clamp_min(1)),
            "contact": (active_match * active).sum(-1) / active.sum(-1).clamp_min(1),
            "support_loss": ((~matched) & maintained).sum(-1).float(),
            "slip": (speed.square() * maintained * (actual >= 0)).sum(-1),
            "wrong_contact": (~allowed).sum(-1).float(),
            "danger": danger.float(),
            "completion": self.completed.float() / self._env.step_dt,
            "failure": self.failed.float() / self._env.step_dt,
        }
        self.metrics["task_index"] = self.task_index.float()
        self.metrics["task_success"] = self.completed.float()
        self.metrics["task_timeout"] = self.failed.float()
        self.metrics["task_completed_count"] += self.completed.float()
        self.metrics["target_distance_m"] = (distance * active * desired).sum(-1) / (active & desired).sum(-1).clamp_min(1)
        self.metrics["target_contact_fraction"] = correct.sum(-1) / desired.sum(-1).clamp_min(1)
        self.metrics["task_elapsed_s"] = self.task_steps.float() / self._control_fps
        self.metrics["task_unsafe"] = unsafe.float()
        for index in range(len(self.task_sources)):
            selected = self.task_index == index
            self.metrics[f"task_{index:02d}_successes"] += (selected & self.completed).float()
            self.metrics[f"task_{index:02d}_failures"] += (selected & self.failed).float()

    def _update_metrics(self):
        self.refresh()

    def _update_command(self):
        self.refresh()
        if not self.cfg.single_task_episode:
            advance = self.completed & ~self.failed & (self.task_index < len(self.task_sources) - 1)
            ids = advance.nonzero().flatten()
            self.task_index[ids] += 1
            self._set_task_goals(ids)
            # Consume the event once; reward values still belong to the old
            # goal until the next physics step refreshes them.
            self.completed[ids] = False
            for index in range(len(self.task_sources)):
                self.metrics[f"task_{index:02d}_attempts"] += (advance & (self.task_index == index)).float()


@configclass
class ContactTaskCommandCfg(LadderMotionCommandCfg):
    class_type: type = ContactTaskCommand
    single_task_episode: bool = True
    task_limb_speed_limit: float = 0.25
    task_min_upright: float = 0.5
    task_min_root_height: float = 0.4
    task_ground_height_tolerance_m: float = 0.12
    task_settle_steps: int = 10
    task_failure_confirmation_steps: int = 5


def contact_task_observation(env, command_name="motion"):
    c = env.command_manager.get_term(command_name)
    quat = c.robot_anchor_quat_w[:, None].expand(-1, 4, -1).reshape(-1, 4)
    desired = c.target_ids >= 0
    def local(vector):
        return quat_apply_inverse(quat, vector.reshape(-1, 3)).reshape(env.num_envs, 4, 3)
    goals = local(c.goal_pos_w - c.robot_anchor_pos_w[:, None]) * desired[..., None]
    positions = local(c.robot_limb_pos_w - c.robot_anchor_pos_w[:, None])
    velocities = local(c.robot.data.body_lin_vel_w[:, c.limb_body_indexes]) * 0.2
    # SI positions / 1m; velocity / 5m/s; masks and direction already bounded.
    return torch.cat((goals.flatten(1).clamp(-3, 3), positions.flatten(1).clamp(-3, 3),
                      velocities.flatten(1).clamp(-3, 3),
                      c.actual_stable_support_contacts.float(), c.active_limbs.float(),
                      desired.float(), ((~c.active_limbs) & (c.source_ids >= 0)).float(),
                      c.effective_climb_command), dim=-1)


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
