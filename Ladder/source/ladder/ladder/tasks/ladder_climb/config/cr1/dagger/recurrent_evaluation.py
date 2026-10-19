"""Tensor-only accounting for each environment's first evaluation episode."""

from collections import Counter

import torch


class FirstEpisodeAccumulator:
    """The collector supplies post-physics state captured before automatic reset."""

    def __init__(self, initial_state, step_dt, downward_start_frame=None):
        self.initial = {key: value.detach().clone() for key, value in initial_state.items()}
        self.last = {key: value.clone() for key, value in self.initial.items()}
        self.step_dt = float(step_dt)
        self.downward_start_frame = downward_start_frame
        frame = initial_state["reference_frame"]
        self.active = torch.ones_like(frame, dtype=torch.bool)
        self.steps = torch.zeros_like(frame, dtype=torch.long)
        self.reward_sum = torch.zeros_like(frame, dtype=torch.float64)
        self.action_mse_sum = torch.zeros_like(self.reward_sum)
        self.root_z_min = initial_state["root_pos"][:, 2].clone()
        self.root_z_max = self.root_z_min.clone()
        self.reached_downward_start = torch.zeros_like(self.active)
        self.terminal_terms = {}

    @torch.no_grad()
    def add(self, state, rewards, dones, termination_flags, action_mse):
        active = self.active.clone()
        self.steps += active.long()
        self.reward_sum += torch.where(active, rewards.reshape_as(active), 0.0)
        self.action_mse_sum += torch.where(active, action_mse.reshape_as(active), 0.0)
        for key, value in state.items():
            self.last[key][active] = value[active]
        height = state["root_pos"][:, 2]
        self.root_z_min[active] = torch.minimum(self.root_z_min[active], height[active])
        self.root_z_max[active] = torch.maximum(self.root_z_max[active], height[active])
        failure = torch.zeros_like(active)
        for name, flag in termination_flags.items():
            if name not in ("motion_complete", "time_out"):
                failure |= flag.bool()
        if self.downward_start_frame is not None:
            self.reached_downward_start |= (
                active & ~failure
                & (self.initial["reference_frame"] < self.downward_start_frame)
                & (state["reference_frame"] >= self.downward_start_frame)
            )
        first_done = active & dones.bool()
        for name, flags in termination_flags.items():
            self.terminal_terms.setdefault(name, torch.zeros_like(active))
            self.terminal_terms[name] |= first_done & flags.bool()
        self.active &= ~dones.bool()

    def report(self):
        episodes = []
        for env_id in range(self.active.numel()):
            terms = sorted(name for name, flags in self.terminal_terms.items() if bool(flags[env_id]))
            censored = bool(self.active[env_id])
            failures = [name for name in terms if name not in ("motion_complete", "time_out")]
            complete = "motion_complete" in terms
            success = not censored and complete and not failures
            steps = int(self.steps[env_id])
            eligible_ascent = (
                self.downward_start_frame is not None
                and int(self.initial["reference_frame"][env_id]) < self.downward_start_frame
            )
            episodes.append({
                "env_id": env_id, "outcome": "censored" if censored else "success" if success else "failure",
                "success": success, "censored": censored, "termination_terms": terms,
                "length_steps": steps, "duration_s": steps * self.step_dt,
                "reward_sum": float(self.reward_sum[env_id]),
                "mean_step_reward": float(self.reward_sum[env_id] / steps) if steps else None,
                "action_mean_rmse_to_teacher": float((self.action_mse_sum[env_id] / steps).sqrt()) if steps else None,
                "initial_state": {key: value[env_id].cpu().tolist() for key, value in self.initial.items()},
                "last_state_before_any_reset": {key: value[env_id].cpu().tolist() for key, value in self.last.items()},
                "last_state_is_terminal": not censored,
                "root_z_min_m": float(self.root_z_min[env_id]),
                "root_z_max_m": float(self.root_z_max[env_id]),
                "root_z_max_increase_m": float(self.root_z_max[env_id] - self.initial["root_pos"][env_id, 2]),
                "root_z_max_decrease_m": float(self.initial["root_pos"][env_id, 2] - self.root_z_min[env_id]),
                "reached_downward_start_without_failure": (
                    bool(self.reached_downward_start[env_id]) if eligible_ascent else None
                ),
            })
        counts = Counter(row["outcome"] for row in episodes)
        total_steps = int(self.steps.sum())
        completed = len(episodes) - counts["censored"]
        return episodes, {
            "num_episodes": len(episodes), "success_count": counts["success"],
            "failure_count": counts["failure"], "censored_count": counts["censored"],
            "success_fraction_all_episodes": counts["success"] / len(episodes),
            "success_fraction_completed_episodes": counts["success"] / completed if completed else None,
            "mean_length_steps": float(self.steps.float().mean()),
            "mean_episode_reward": float(self.reward_sum.mean()),
            "mean_step_reward": float(self.reward_sum.sum() / total_steps) if total_steps else None,
            "action_mean_rmse_to_teacher": float((self.action_mse_sum.sum() / total_steps).sqrt()) if total_steps else None,
            "termination_counts": dict(Counter(name for row in episodes for name in row["termination_terms"])),
            "downward_start_frame": self.downward_start_frame,
            "ascent_eligible_episodes": sum(row["reached_downward_start_without_failure"] is not None for row in episodes),
            "ascent_reached_downward_start_count": sum(row["reached_downward_start_without_failure"] is True for row in episodes),
        }
