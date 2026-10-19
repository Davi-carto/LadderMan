"""Temporal storage for frozen-policy TransformerVision pretraining."""

from __future__ import annotations

import torch


class TransformerVisionRolloutStorage:
    """Store supervision aligned with the frozen actor's real trajectory."""

    def __init__(self, num_steps: int, num_envs: int, device: str | torch.device) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step = 0
        self.proprio_history = torch.zeros(num_steps, num_envs, 10, 85, device=self.device)
        self.depth_history = torch.zeros(num_steps, num_envs, 2, 60, 80, device=self.device)
        self.velocity_targets = torch.zeros(num_steps, num_envs, 3, device=self.device)
        self.limb_heightmaps = torch.zeros(num_steps, num_envs, 4, 10, 3, device=self.device)
        self.heightmap_valid = torch.zeros(num_steps, num_envs, 4, 10, 3, device=self.device)
        self.heightmap_hit_class = torch.zeros(
            num_steps, num_envs, 4, 10, 3, dtype=torch.long, device=self.device
        )
        self.next_proprio_targets = torch.zeros(num_steps, num_envs, 85, device=self.device)
        self.limb_goal_targets = torch.zeros(num_steps, num_envs, 4, 3, device=self.device)
        self.limb_goal_valid = torch.zeros(num_steps, num_envs, 4, device=self.device)
        self.contact_indicator_targets = torch.zeros(num_steps, num_envs, 4, device=self.device)
        self.motion_joint_pos_targets = torch.zeros(
            num_steps, num_envs, 25, device=self.device
        )
        self.motion_root_orientation_targets = torch.zeros(
            num_steps, num_envs, 6, device=self.device
        )
        self.motion_target_valid = torch.zeros(
            num_steps, num_envs, dtype=torch.bool, device=self.device
        )
        self.task_goal_available = False
        self.dones = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=self.device)
        self.future_valid = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=self.device)
        self.initial_hidden_state = torch.zeros(1, num_envs, 256, device=self.device)

    def add(
        self,
        *,
        proprio_history: torch.Tensor,
        depth_history: torch.Tensor,
        velocity: torch.Tensor,
        limb_heightmaps: torch.Tensor,
        heightmap_valid: torch.Tensor,
        heightmap_hit_class: torch.Tensor,
        next_proprio: torch.Tensor,
        dones: torch.Tensor,
        limb_goal_b: torch.Tensor | None = None,
        limb_goal_valid: torch.Tensor | None = None,
        contact_indicator: torch.Tensor | None = None,
        motion_joint_pos: torch.Tensor | None = None,
        motion_root_orientation_6d: torch.Tensor | None = None,
        motion_valid: torch.Tensor | None = None,
    ) -> None:
        if self.step >= self.num_steps:
            raise OverflowError("TransformerVision rollout storage is full")
        index = self.step
        expected = {
            "proprio_history": (self.num_envs, 10, 85),
            "depth_history": (self.num_envs, 2, 60, 80),
            "velocity": (self.num_envs, 3),
            "limb_heightmaps": (self.num_envs, 4, 10, 3),
            "heightmap_valid": (self.num_envs, 4, 10, 3),
            "heightmap_hit_class": (self.num_envs, 4, 10, 3),
            "next_proprio": (self.num_envs, 85),
            "dones": (self.num_envs,),
        }
        for name, value in (
            ("proprio_history", proprio_history),
            ("depth_history", depth_history),
            ("velocity", velocity),
            ("limb_heightmaps", limb_heightmaps),
            ("heightmap_valid", heightmap_valid),
            ("heightmap_hit_class", heightmap_hit_class),
            ("next_proprio", next_proprio),
            ("dones", dones),
        ):
            if tuple(value.shape) != expected[name]:
                raise ValueError(f"{name} must have shape {expected[name]}, got {tuple(value.shape)}")
        task_values = (
            limb_goal_b,
            limb_goal_valid,
            contact_indicator,
            motion_joint_pos,
            motion_root_orientation_6d,
            motion_valid,
        )
        if any(value is not None for value in task_values) and not all(
            value is not None for value in task_values
        ):
            raise ValueError(
                "limb-goal, contact, and stable-motion targets must be provided together"
            )
        if limb_goal_b is not None:
            task_expected = {
                "limb_goal_b": (self.num_envs, 4, 3),
                "limb_goal_valid": (self.num_envs, 4),
                "contact_indicator": (self.num_envs, 4),
                "motion_joint_pos": (self.num_envs, 25),
                "motion_root_orientation_6d": (self.num_envs, 6),
                "motion_valid": (self.num_envs,),
            }
            for name, value in (
                ("limb_goal_b", limb_goal_b),
                ("limb_goal_valid", limb_goal_valid),
                ("contact_indicator", contact_indicator),
                ("motion_joint_pos", motion_joint_pos),
                ("motion_root_orientation_6d", motion_root_orientation_6d),
                ("motion_valid", motion_valid),
            ):
                if tuple(value.shape) != task_expected[name]:
                    raise ValueError(
                        f"{name} must have shape {task_expected[name]}, got {tuple(value.shape)}"
                    )
        self.proprio_history[index].copy_(proprio_history)
        self.depth_history[index].copy_(depth_history)
        self.velocity_targets[index].copy_(velocity)
        self.limb_heightmaps[index].copy_(limb_heightmaps)
        self.heightmap_valid[index].copy_(heightmap_valid)
        self.heightmap_hit_class[index].copy_(heightmap_hit_class.to(dtype=torch.long))
        self.next_proprio_targets[index].copy_(next_proprio)
        if limb_goal_b is not None:
            self.limb_goal_targets[index].copy_(limb_goal_b)
            self.limb_goal_valid[index].copy_(limb_goal_valid)
            self.contact_indicator_targets[index].copy_(contact_indicator)
            self.motion_joint_pos_targets[index].copy_(motion_joint_pos)
            self.motion_root_orientation_targets[index].copy_(
                motion_root_orientation_6d
            )
            self.motion_target_valid[index].copy_(motion_valid.to(dtype=torch.bool))
            self.task_goal_available = True
        self.dones[index].copy_(dones.to(dtype=torch.bool))
        self.future_valid[index].copy_(~self.dones[index])
        self.step += 1

    def reset_before_mask(self) -> torch.Tensor:
        result = torch.zeros_like(self.dones)
        result[1:] = self.dones[:-1]
        return result

    def mark_previous_transition_done(self) -> None:
        """Insert a sequence boundary before a requested manual reset."""

        if self.step == 0:
            return
        self.dones[self.step - 1].fill_(True)
        self.future_valid[self.step - 1].fill_(False)

    def clear(self) -> None:
        self.step = 0
        self.task_goal_available = False
