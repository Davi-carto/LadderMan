"""CPU-only summaries of first-episode, same-state teacher/student traces.

Threshold crossings are descriptive markers, not failure detectors. In particular,
leaning is normal while climbing; root tilt alone does not establish instability.
The collector must capture post-action state before the environment auto-resets.
"""

import math

import torch


_WINDOWS = (("steps_1_9", 0, 9), ("steps_10_50", 9, 50), ("steps_51_100", 50, 100),
            ("steps_101_200", 100, 200), ("steps_201_plus", 200, None))
_SCALARS = ("anchor_pos_error", "anchor_rot_error", "root_tilt")


def _number(value):
    result = float(value)
    return result if math.isfinite(result) else None


def _metrics(action_error, feature_error):
    count, num_actions = action_error.shape
    if count == 0:
        return {
            "count": 0, "action_mse": None, "action_rmse": None,
            "action_mae": None, "action_max_abs": None, "feature_mse": None,
            "per_joint": [{"joint_index": index, "rmse": None, "mae": None, "max_abs": None}
                          for index in range(num_actions)],
        }
    action_error = action_error.double()
    return {
        "count": count,
        "action_mse": _number(action_error.square().mean()),
        "action_rmse": _number(action_error.square().mean().sqrt()),
        "action_mae": _number(action_error.abs().mean()),
        "action_max_abs": _number(action_error.abs().max()),
        "feature_mse": _number(feature_error.double().square().mean()),
        "per_joint": [
            {"joint_index": index, "rmse": _number(error.square().mean().sqrt()),
             "mae": _number(error.abs().mean()), "max_abs": _number(error.abs().max())}
            for index, error in enumerate(action_error.unbind(dim=-1))
        ],
    }


def _scalar_summary(values):
    if values.numel() == 0:
        return {"count": 0, "mean": None, "max": None}
    return {"count": values.numel(), "mean": _number(values.double().mean()), "max": _number(values.max())}


@torch.no_grad()
def summarize_shadow_trace(trace: dict[str, torch.Tensor], step_dt: float, controller: str) -> dict:
    """Summarize valid first-episode steps without averaging per-episode RMSEs.

    All trace fields have leading dimensions ``[action_step, environment]``.
    ``reference_frame`` describes the pre-action reference, even for post-action
    threshold markers. Windows use one-based human labels, while every stored
    ``action_step`` is zero-based. Post-action times are ``(action_step + 1) * dt``.
    No statistic includes steps following the first valid ``done`` per environment.
    """
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("step_dt must be finite and positive")
    trace = {name: value.detach().cpu() for name, value in trace.items()}
    valid = trace["valid"].bool()
    done = trace["done"].bool() & valid
    # Also exclude accidental post-reset data if a collector leaves valid true.
    valid = valid & ((done.long().cumsum(dim=0) - done.long()) == 0)
    steps, num_envs = valid.shape
    action_error = trace["student_action"] - trace["teacher_action"]
    feature_error = trace["student_feature"] - trace["teacher_feature"]
    action_rmse = action_error.double().square().mean(dim=-1).sqrt()
    terminations = {name.removeprefix("termination/"): value.bool()
                    for name, value in trace.items() if name.startswith("termination/")}
    scalar_series = {f"{timing}_{name}": trace[f"{timing}_{name}"]
                     for timing in ("pre", "post") for name in _SCALARS}
    for timing in ("pre", "post"):
        scalar_series[f"{timing}_root_ang_speed"] = trace[f"{timing}_root_ang_vel"].double().norm(dim=-1)
    required_miss = (trace["post_required_contact"].bool() & ~trace["post_physical_contact"].bool()).any(dim=-1)
    episodes = []

    def first_event(mask, values, env_id, timing):
        selected = (mask[:, env_id] & valid[:, env_id]).nonzero().flatten()
        if selected.numel() == 0:
            return None
        step = int(selected[0])
        return {
            "action_step": step,
            "time_s": (step + int(timing == "post_action")) * step_dt,
            "timing": timing,
            "pre_action_reference_frame": int(trace["reference_frame"][step, env_id]),
            "value": _number(values[step, env_id]),
        }

    event_specs = {}
    for threshold, suffix in ((0.05, "0p05"), (0.10, "0p10")):
        event_specs[f"action_rmse_over_{suffix}"] = (action_rmse > threshold, action_rmse, "pre_action_prediction")
    for quantity, thresholds in (("anchor_pos_error", ((0.05, "0p05"), (0.10, "0p10"), (0.20, "0p20"))),
                                  ("anchor_rot_error", ((0.10, "0p10"), (0.20, "0p20"), (0.40, "0p40"))),
                                  ("root_tilt", ((0.20, "0p20"), (0.40, "0p40")))):
        values = trace[f"post_{quantity}"]
        for threshold, suffix in thresholds:
            event_specs[f"post_{quantity}_over_{suffix}"] = (values > threshold, values, "post_action")
    event_specs["post_required_contact_miss"] = (required_miss, required_miss, "post_action")

    all_step_indices = torch.arange(steps)
    for env_id in range(num_envs):
        selected = valid[:, env_id]
        indices = selected.nonzero().flatten()
        length = int(indices.numel())
        terminal_indices = (done[:, env_id] & selected).nonzero().flatten()
        terminal_step = int(terminal_indices[0]) if terminal_indices.numel() else None
        terminal_terms = sorted(name for name, values in terminations.items()
                                if terminal_step is not None and bool(values[terminal_step, env_id]))
        marker = "motion_complete" in terminal_terms
        failure_causes = [name for name in terminal_terms if name not in ("motion_complete", "time_out")]
        if terminal_step is not None and not marker and "time_out" in terminal_terms:
            failure_causes.append("time_out")
        success = bool(marker and not failure_causes) if terminal_step is not None else None
        if terminal_step is not None and not success and not failure_causes:
            failure_causes.append("unclassified_done")
        outcome = "no_data" if length == 0 else "censored" if terminal_step is None else "success" if success else "failure"
        windows = {}
        for name, start, stop in _WINDOWS:
            window_mask = selected & (all_step_indices >= start)
            if stop is not None:
                window_mask &= all_step_indices < stop
            windows[name] = _metrics(action_error[window_mask, env_id], feature_error[window_mask, env_id])
            windows[name]["action_step_start_inclusive"] = start
            windows[name]["action_step_stop_exclusive"] = stop
        last_indices = indices[-25:]
        windows["last_25_steps"] = _metrics(action_error[last_indices, env_id], feature_error[last_indices, env_id])
        tracking = {name: _scalar_summary(values[selected, env_id]) for name, values in scalar_series.items()}
        for name, values in scalar_series.items():
            tracking[name]["first"] = _number(values[indices[0], env_id]) if length else None
            tracking[name]["last"] = _number(values[indices[-1], env_id]) if length else None
        episodes.append({
            "env_id": env_id, "length_steps": length, "duration_s": length * step_dt,
            "start_reference_frame": int(trace["reference_frame"][indices[0], env_id]) if length else None,
            "end_pre_action_reference_frame": int(trace["reference_frame"][indices[-1], env_id]) if length else None,
            "terminal_action_step": terminal_step,
            "terminal_time_s": (terminal_step + 1) * step_dt if terminal_step is not None else None,
            "terminated": terminal_step is not None, "censored": outcome == "censored", "outcome": outcome,
            "motion_complete_marker": marker, "success": success,
            "terminal_terms": terminal_terms, "failure_causes": failure_causes,
            "action_metrics": _metrics(action_error[selected, env_id], feature_error[selected, env_id]),
            "windows": windows, "tracking": tracking,
            "post_required_contact_miss_steps": int(required_miss[selected, env_id].sum()),
            "delay_steps": sorted(int(value) for value in trace["delay"][selected, env_id].unique()),
            "first_events": {name: first_event(mask, values, env_id, timing)
                             for name, (mask, values, timing) in event_specs.items()},
        })

    observed = [episode for episode in episodes if episode["length_steps"] > 0]
    completed = [episode for episode in observed if episode["terminated"]]
    success_count = sum(episode["success"] is True for episode in completed)
    failure_counts = {}
    for episode in completed:
        for cause in episode["failure_causes"]:
            failure_counts[cause] = failure_counts.get(cause, 0) + 1
    return {
        "schema_version": 1, "controller": controller, "step_dt": step_dt,
        "notes": [
            "Metrics include only each environment's first episode, including its terminating action.",
            "Action errors compare deterministic student and teacher predictions on the same pre-action state.",
            "Action units are unscaled policy units; position is meters and rotation/tilt is radians.",
            "Post-action state must be captured before automatic reset; reference_frame is pre-action.",
            "Thresholds are descriptive, not causal or failure declarations; nominal climbing may involve root tilt.",
            "Censored episodes have no observed termination and are not counted as successes or failures.",
            "Success requires a motion_complete marker without a simultaneous non-timeout failure term.",
            "Aggregate action/feature errors are weighted by sample count, not equally by episode.",
        ],
        "aggregate": {
            "num_envs": num_envs, "observed_episodes": len(observed), "empty_environments": num_envs - len(observed),
            "terminated_episodes": len(completed), "successful_episodes": success_count,
            "failed_episodes": len(completed) - success_count,
            "censored_episodes": sum(episode["censored"] for episode in observed),
            "motion_complete_markers": sum(episode["motion_complete_marker"] for episode in completed),
            "success_fraction_of_terminated": success_count / len(completed) if completed else None,
            "failure_causes": failure_counts,
            "action_metrics": _metrics(action_error[valid], feature_error[valid]),
            "tracking": {name: _scalar_summary(values[valid]) for name, values in scalar_series.items()},
            "post_required_contact_miss_steps": int(required_miss[valid].sum()),
        },
        "episodes": episodes,
    }
