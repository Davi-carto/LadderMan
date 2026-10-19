"""Reward-only reference guidance; never advances the physical support task.

Teacher reward kernels follow ladder_climb/mdp/rewards.py. Position alignment
follows LadderMotionCommand: align anchor XY/yaw, retain reference height.
"""

import re

import torch
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat

from .contact_task import SupportContactPhases


REFERENCE_TERMS = (
    "reference_body_lower", "reference_body_upper", "reference_joint_lower",
    "reference_joint_upper", "reference_swing_foot", "reference_anchor_orientation",
)


class PhaseReferenceClock:
    """One whole-body clock bounded by all four limbs' execution phases.

Loops here build a small static table at startup, never loop over environments.
RELEASE includes unloading/preparation before detachment. This permits staged
two-hand releases without inventing a pose assembled from different frames.
"""

    def __init__(self, anchors, support_ids, num_envs, *, window=3, lead=5):
        if window < 0 or lead < 0:
            raise ValueError("Reference window and lead must be nonnegative")
        self.anchors = anchors
        self.final_frame = int(anchors[-1])
        self.window = window
        self.lead = lead
        self.offsets = torch.arange(-window, window + 1, device=anchors.device)
        self.lower = anchors[:-1, None, None].expand(-1, 4, 4).clone()
        self.upper = anchors[1:, None, None].expand(-1, 4, 4).clone()
        self.landing = anchors[1:, None].expand(-1, 4).clone()
        self.frame = torch.zeros(num_envs, device=anchors.device)
        # Inspect CPU copies only once, avoiding repeated CUDA synchronizations.
        ids = support_ids.cpu().tolist()
        for segment, (start, end) in enumerate(zip(anchors.tolist(), anchors[1:].tolist())):
            for limb in range(4):
                source, target = ids[start][limb], ids[end][limb]
                values = [row[limb] for row in ids[start:end + 1]]
                runs = [v for i, v in enumerate(values) if i == 0 or v != values[i - 1]]
                expected = [source] if source == target else list(dict.fromkeys((source, -1, target)))
                if runs != expected:
                    raise ValueError(f"Unsupported reference contact sequence at segment {segment}, limb {limb}: {runs}")
                if source == target:
                    continue
                release = start if source < 0 else start + values.index(-1)
                landing = end if target < 0 else start + values.index(target)
                self.landing[segment, limb] = landing
                self.upper[segment, limb, SupportContactPhases.PREPARE] = max(start, release - 1)
                self.upper[segment, limb, SupportContactPhases.RELEASE] = min(
                    release + lead, landing - 1 if target >= 0 else end)
                self.lower[segment, limb, SupportContactPhases.MOVE] = release
                self.upper[segment, limb, SupportContactPhases.MOVE] = landing
                self.lower[segment, limb, SupportContactPhases.ACQUIRE] = (
                    max(release, landing - lead) if target >= 0 else release)

    def reset(self, env_ids, frames):
        self.frame[env_ids] = frames.float()

    def bounds(self, task_index, phases):
        lower = self.lower[task_index].gather(2, phases.phase.unsqueeze(-1)).squeeze(-1)
        upper = self.upper[task_index].gather(2, phases.phase.unsqueeze(-1)).squeeze(-1)
        # Established support must not receive an earlier flight reference.
        lower = torch.where(phases.acquired & (phases.target >= 0),
                            torch.maximum(lower, self.landing[task_index]), lower)
        return lower.amax(-1), upper.amin(-1)

    def candidates(self, task_index, phases):
        lower, upper = self.bounds(task_index, phases)
        raw = self.frame.floor().long()[:, None] + self.offsets
        valid = (raw >= lower[:, None]) & (raw <= upper[:, None])
        # Clamping is solely for safe gathers; invalid candidates stay masked.
        return raw.clamp(0, self.final_frame), valid

    def advance(self, task_index, phases, running, frame_increment):
        lower, upper = self.bounds(task_index, phases)
        proposed = torch.minimum(torch.maximum(self.frame + frame_increment, lower.float()), upper.float())
        valid = running & (lower <= upper)
        # A faster actual phase can skip obsolete poses, never rewind the clock.
        next_frame = torch.where(valid, torch.maximum(self.frame, proposed), self.frame)
        delta = next_frame - self.frame
        self.frame.copy_(next_frame)
        return delta


def reference_reward_candidates(body_error_sq, joint_error_sq, orientation_error_sq,
                                lower_bodies, upper_bodies, lower_joints, upper_joints,
                                feet, swing):
    """Return six [env, window] rate kernels and their physical error measures."""
    body_lower = body_error_sq[..., lower_bodies].mean(-1)
    body_upper = body_error_sq[..., upper_bodies].mean(-1)
    joint_lower = joint_error_sq[..., lower_joints].sum(-1)
    joint_upper = joint_error_sq[..., upper_joints].sum(-1)
    foot_error = body_error_sq[..., feet]
    count = swing.sum(-1).clamp_min(1)[:, None]
    foot_reward = (torch.exp(-foot_error / .02) * swing[:, None]).sum(-1) / count
    foot_mse = (foot_error * swing[:, None]).sum(-1) / count
    rewards = torch.stack((
        torch.exp(-body_lower / .18**2), torch.exp(-body_upper / .25**2),
        torch.exp(-joint_lower / .8), torch.exp(-joint_upper / 1.0),
        foot_reward, torch.exp(-orientation_error_sq / .35**2),
    ), dim=-1)
    errors = torch.stack((body_lower.sqrt(), body_upper.sqrt(),
                          (joint_lower / len(lower_joints)).sqrt(),
                          (joint_upper / len(upper_joints)).sqrt(),
                          foot_mse.sqrt(), orientation_error_sq.sqrt()), dim=-1)
    return rewards, errors


def select_reference_frame(frames, valid, rewards, errors, weights):
    """Select one shared frame, not six independent easiest reference poses."""
    score = (rewards * weights).sum(-1).masked_fill(~valid, -torch.inf)
    choice = score.argmax(-1)
    rows = torch.arange(frames.shape[0], device=frames.device)
    has_match = valid.any(-1)
    selected = torch.where(has_match, frames[rows, choice], -1)
    return (selected, rewards[rows, choice] * has_match[:, None],
            errors[rows, choice] * has_match[:, None], has_match)


class ContactTaskReferenceGuidance:
    """Adapter over the command's reordered motion data; no actor/AMP changes."""

    def __init__(self, command):
        c = command
        if c.contact_phases is None or c._geometry_randomization_enabled:
            raise ValueError("Reference guidance requires phased tasks and nominal ladder geometry")
        cfg = c.cfg
        self.clock = PhaseReferenceClock(c.task_anchors, c.motion.stable_support_rung_id, c.num_envs,
                                        window=cfg.guidance_window_frames, lead=cfg.guidance_lead_frames)
        self.lower_bodies = [cfg.body_names.index(n) for n in cfg.guidance_lower_body_names]
        self.upper_bodies = [cfg.body_names.index(n) for n in cfg.guidance_upper_body_names]
        self.feet = [cfg.body_names.index(n) for n in ("left_ankle_x_link", "right_ankle_x_link")]
        self.lower_joints = [i for i, n in enumerate(c.robot.joint_names)
                             if re.fullmatch(cfg.guidance_lower_joint_pattern, n)]
        self.upper_joints = [i for i, n in enumerate(c.robot.joint_names)
                             if re.fullmatch(cfg.guidance_upper_joint_pattern, n)]
        if not all((self.lower_bodies, self.upper_bodies, self.lower_joints, self.upper_joints)):
            raise ValueError("Reference reward body/joint groups must be nonempty")
        self.weights = torch.tensor([getattr(c._env.cfg.rewards, n).weight for n in REFERENCE_TERMS],
                                    device=c.device)
        if not torch.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError("Reference selection requires finite nonnegative reward weights")
        self.diagnostics = {}
        self.body_targets_w = torch.zeros(c.num_envs, len(cfg.body_names), 3, device=c.device)
        self._last_advance_step = -1
        for name in ("frame", "matched_frame", "valid", "pause_time_s", "conflict_time_s",
                     "phase_skip_frames", "weighted_reward_integral", "segment_end_wait_time_s"):
            c.metrics[f"guidance_{name}"] = torch.zeros(c.num_envs, device=c.device)
        for name in REFERENCE_TERMS:
            c.metrics[f"guidance_{name}_error"] = torch.zeros(c.num_envs, device=c.device)

    def reset(self, env_ids, frames):
        self.clock.reset(env_ids, frames)
        self.body_targets_w[env_ids] = 0
        for value in self.diagnostics.values():
            value[env_ids] = 0

    @torch.no_grad()
    def evaluate(self, c, running):
        # Called before contact_phases.step(): rewards evaluate the issued intent.
        frames, valid = self.clock.candidates(c.task_index, c.contact_phases)
        valid &= running[:, None]
        reference = c.motion.body_pos_w[frames]
        anchor = reference[:, :, c.motion_anchor_body_index]
        q_ref = c.motion.body_quat_w[frames, c.motion_anchor_body_index]
        q_robot = c.robot_anchor_quat_w[:, None].expand_as(q_ref)
        delta_q = yaw_quat(quat_mul(q_robot, quat_inv(q_ref)))
        delta_p = c.robot_anchor_pos_w[:, None].expand_as(anchor).clone()
        delta_p[..., 2] = anchor[..., 2] + c._env.scene.env_origins[:, None, 2]
        rotation = delta_q[:, :, None].expand(*reference.shape[:-1], 4)
        aligned = delta_p[:, :, None] + quat_apply(rotation, reference - anchor[:, :, None])
        body_error = (aligned - c.robot_body_pos_w[:, None]).square().sum(-1)
        joint_error = (c.motion.joint_pos[frames] - c.robot_joint_pos[:, None]).square()
        angle_error = quat_error_magnitude(q_ref, q_robot).square()
        phases = c.contact_phases
        swing = (phases.active[:, 2:] & (phases.phase[:, 2:] != phases.PREPARE)
                 & ~phases.acquired[:, 2:])
        rewards, errors = reference_reward_candidates(
            body_error, joint_error, angle_error, self.lower_bodies, self.upper_bodies,
            self.lower_joints, self.upper_joints, self.feet, swing)
        selected, values, selected_errors, has_match = select_reference_frame(
            frames, valid, rewards, errors, self.weights)
        for i, name in enumerate(REFERENCE_TERMS):
            c._reward_values[name] = values[:, i]  # rates; RewardManager multiplies dt exactly once
            c.metrics[f"guidance_{name}_error"] = selected_errors[:, i]
        c.metrics["guidance_frame"] = self.clock.frame.clone()
        c.metrics["guidance_matched_frame"] = selected.float()
        c.metrics["guidance_valid"] = has_match.float()
        c.metrics["guidance_conflict_time_s"] += (running & ~has_match) * c._env.step_dt
        c.metrics["guidance_weighted_reward_integral"] += (values * self.weights).sum(-1) * c._env.step_dt
        # Only one selected pose is retained for lightweight playback diagnostics.
        rows = torch.arange(c.num_envs, device=c.device)
        choice = (frames == selected[:, None]).long().argmax(-1)
        self.body_targets_w = aligned[rows, choice] * has_match[:, None, None]
        self.diagnostics = {
            "frame": self.clock.frame.clone(), "matched_frame": selected,
            "valid": has_match, "foot_targets_w": self.body_targets_w[:, self.feet],
            "weighted_rate": (values * self.weights).sum(-1),
        }

    def advance(self, c, running):
        # After phase updates/rewards, prepare next step's reference. Never
        # modify time_steps, task_index, task deadlines, or contact milestones.
        if self._last_advance_step == c._env.common_step_counter:
            return
        self._last_advance_step = c._env.common_step_counter
        delta = self.clock.advance(c.task_index, c.contact_phases, running,
                                   c.motion.fps * c._env.step_dt)
        c.metrics["guidance_pause_time_s"] += (running & (delta == 0)) * c._env.step_dt
        c.metrics["guidance_phase_skip_frames"] += (delta - c.motion.fps * c._env.step_dt).clamp_min(0)
        at_end = self.clock.frame >= c.task_anchors[c.task_index + 1]
        c.metrics["guidance_segment_end_wait_time_s"] += (running & at_end) * c._env.step_dt
        lower, upper = self.clock.bounds(c.task_index, c.contact_phases)
        # 0=advancing, 1=phase boundary, 2=segment end, 3=no shared range,
        # 4=reset/terminal/unsafe. This is next-step timing, not reward selection.
        reason = torch.where(delta > 0, 0, torch.where(at_end, 2, 1))
        reason = torch.where(lower > upper, 3, reason)
        reason = torch.where(running, reason, 4)
        self.diagnostics["next_frame"] = self.clock.frame.clone()
        self.diagnostics["pause_reason"] = reason
