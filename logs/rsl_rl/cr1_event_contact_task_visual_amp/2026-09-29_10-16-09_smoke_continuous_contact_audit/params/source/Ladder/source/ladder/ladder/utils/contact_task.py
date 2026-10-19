"""Clock-free support-task planning and completion, independent of Isaac Sim."""

import torch


def support_task_plan(anchors, support_ids, direction):
    """Keep annotated support transitions, including simultaneous hand changes."""
    if anchors.ndim != 1 or anchors.numel() < 2 or not torch.all(anchors[1:] > anchors[:-1]):
        raise ValueError("Support tasks require ordered start and end anchors")
    source, target = support_ids[anchors[:-1]], support_ids[anchors[1:]]
    active = source != target
    if not active.any(-1).all():
        raise ValueError("A support task must change at least one support identity")
    # Anchor direction can be zero at a standing endpoint. Use the nonzero
    # intent within its segment, not a direction queried from a running clock.
    directions = []
    for start, end in zip(anchors[:-1].tolist(), anchors[1:].tolist()):
        values = direction[start:end + 1]
        values = values[values != 0]
        directions.append(values[0] if values.numel() else direction.new_zeros(()))
    return source, target, active, torch.stack(directions)


def support_task_completion(source, target, active, actual, near, slow, safe, elapsed,
                            missing_steps, *, relaxed_after_steps, flicker_steps=2):
    """New targets are mandatory; only short old-support sensor gaps may relax."""
    desired = target >= 0
    matched = (actual == target) & (~desired | (near & slow))
    maintained = (~active) & (source >= 0)
    strict = matched.all(-1) & safe
    # An old support may momentarily unload, but a wrong-rung contact is never
    # a sensor flicker. New contact and release tasks are never waived.
    flicker = maintained & (actual < 0) & (missing_steps <= flicker_steps)
    relaxed = (matched | flicker).all(-1) & (flicker.sum(-1) <= 1) & safe
    return strict | ((elapsed >= relaxed_after_steps) & relaxed), matched, maintained


def signed_goal_progress(previous_distance, distance, active, desired):
    mask = active & desired
    return (((previous_distance - distance) * mask).sum(-1)
            / mask.sum(-1).clamp_min(1))
