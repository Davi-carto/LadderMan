"""Clock-free support-task planning and completion, independent of Isaac Sim."""

import torch


class SupportContactPhases:
    """Per-limb, event-driven intent; no reference clock or per-env Python loop.

    PREPARE keeps the old support, RELEASE requests detachment, MOVE allows
    flight, and ACQUIRE requests/maintains the new support (or final release).
    Physical target contact can skip RELEASE/MOVE; it must never be punished
    just because the command still requested flight on the preceding step.
    """

    PREPARE, RELEASE, MOVE, ACQUIRE = range(4)
    NAMES = ("prepare", "release", "move", "acquire")

    def __init__(self, num_envs, device, confirmation_steps=4, release_grace_steps=5):
        self.confirmation_steps = confirmation_steps
        self.release_grace_steps = release_grace_steps
        self.source = torch.full((num_envs, 4), -1, dtype=torch.long, device=device)
        self.target = torch.full_like(self.source, -1)
        self.phase = torch.zeros_like(self.source)
        self.phase_steps = torch.zeros_like(self.source)
        self.release_count = torch.zeros_like(self.source)
        self.safe_release_count = torch.zeros_like(self.source)
        self.contact_count = torch.zeros_like(self.source)
        self.prepare_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.released = torch.zeros_like(self.source, dtype=torch.bool)
        self.acquired = torch.zeros_like(self.released)

    @property
    def active(self):
        return self.source != self.target

    @property
    def expected(self):
        return torch.where(
            self.active & (self.phase != self.PREPARE),
            torch.where(self.phase == self.ACQUIRE, self.target, -1), self.source)

    @property
    def protected(self):
        return (((~self.active) | (self.phase == self.PREPARE)) & (self.source >= 0)
                | (self.acquired & (self.target >= 0)))

    def reset(self, env_ids, source, target, initial_support):
        self.source[env_ids], self.target[env_ids] = source, target
        active = source != target
        initial_target = active & (initial_support == target)
        initial_flight = active & ~initial_target & ((initial_support != source) | (source < 0))
        self.phase[env_ids] = torch.where(initial_target, self.ACQUIRE,
                                         torch.where(initial_flight, self.MOVE, self.PREPARE))
        # Reference annotation seeds episode state, not measured sensor state.
        # Milestones already present at a random reset never pay retroactively.
        self.released[env_ids] = active & ((initial_support != source) | (source < 0))
        self.acquired[env_ids] = initial_target & (target >= 0)
        self.phase_steps[env_ids] = 0
        self.release_count[env_ids] = 0
        self.safe_release_count[env_ids] = 0
        self.contact_count[env_ids] = 0
        self.prepare_count[env_ids] = 0

    def step(self, actual, near, slow, safe, running, missing_steps):
        """Return rewards for the issued intent, then advance next-step intent.

        Event magnitudes here are fractions of a per-goal budget, not rates.
        ``running`` excludes reset settling and terminal failures.
        """
        active, desired = self.active, self.target >= 0
        previous_phase = self.phase.clone()
        expected, protected = self.expected, self.protected
        target_touch = active & desired & (actual == self.target)
        flicker = protected & (actual < 0) & (missing_steps <= 2)
        early_target = target_touch & (self.phase != self.ACQUIRE)
        grace = (active & (self.phase == self.RELEASE) & ~self.released
                 & (self.phase_steps < self.release_grace_steps) & (actual == self.source))
        mismatch = (actual != expected) & ~flicker & ~early_target & ~grace
        lost = protected & (actual != expected) & ~flicker & ~early_target
        allowed_surface = ((actual < 0) | (actual == self.target)
                           | (active & ~self.released & (actual == self.source)))
        values = {
            "phase_contact_error": mismatch.float().mean(-1),
            "support_loss": lost.sum(-1).float(),
            "wrong_contact": (~allowed_surface).sum(-1).float(),
            "new_support_loss": (lost & self.acquired).sum(-1).float(),
            "protected": protected,
        }
        valid = running & safe
        # Old maintained and already established new contacts are mandatory
        # for paying milestones; preparing limbs may safely skip to the target.
        necessary = ((~active) & (self.source >= 0)) | self.acquired
        support_ok = (~necessary | (actual == torch.where(self.acquired, self.target, self.source))
                      | (necessary & (actual < 0) & (missing_steps <= 2))).all(-1)
        # A transient missing reading may relax an already-established support,
        # but it must not be interpreted as the active limb's intentional
        # release while still in PREPARE.  Otherwise one sensor gap can start
        # the release counter and pay a false release event.
        # Another active limb may already be flying/landing after a random
        # reset. Do not require its commanded contact before allowing the
        # preparing limb to start; established supports remain mandatory.
        preparing = active & (previous_phase == self.PREPARE)
        prepare_ready = ~preparing | (actual == self.source) | early_target
        ready = valid & support_ok & prepare_ready.all(-1)
        self.prepare_count = torch.where(ready, self.prepare_count + 1, 0)
        can_start = (self.prepare_count >= self.confirmation_steps)[:, None]
        start = active & (previous_phase == self.PREPARE) & can_start
        self.phase = torch.where(start, self.RELEASE, self.phase)

        can_pay = (valid & support_ok)[:, None]
        # Recognize physical detachment even when velocity/safety prevents a
        # bonus. Otherwise a moving limb can remain in RELEASE indefinitely.
        # In PREPARE, first reject short sensor gaps, then acknowledge sustained
        # early detachment rather than demand a return to the obsolete source.
        detaching = (active & ((previous_phase == self.RELEASE)
                              | (preparing & (missing_steps > 2)))
                     & (actual < 0) & running[:, None])
        self.release_count = torch.where(detaching, self.release_count + 1, 0)
        rewarded_detaching = detaching & (previous_phase == self.RELEASE) & can_pay
        self.safe_release_count = torch.where(rewarded_detaching, self.safe_release_count + 1, 0)
        detached = (self.release_count >= self.confirmation_steps) & ~self.released
        release_event = detached & (self.safe_release_count >= self.confirmation_steps)
        # Early/unsafe detachment consumes the milestone without a delayed
        # payout; the existing support-loss/safety penalties are not waived.
        self.released |= detached
        self.phase = torch.where(detached, torch.where(desired, self.MOVE, self.ACQUIRE), self.phase)
        # Ask for landing before contact, once a flying limb enters its target
        # region. Never regress and ask it to detach again after a sensor gap.
        landing = active & desired & (((previous_phase == self.MOVE) & near) | target_touch)
        self.phase = torch.where(landing & running[:, None], self.ACQUIRE, self.phase)
        confirmed = target_touch & near & slow & can_pay
        self.contact_count = torch.where(confirmed, self.contact_count + 1, 0)
        contact_event = (self.contact_count >= self.confirmation_steps) & ~self.acquired
        self.acquired |= contact_event
        self.released |= contact_event  # Direct safe landing does not invent a release bonus.
        denominator = active.sum(-1).clamp_min(1)
        values["release_event"] = release_event.sum(-1).float() / denominator
        values["contact_event"] = contact_event.sum(-1).float() / denominator
        values["phase_changes"] = (self.phase != previous_phase).sum(-1).float()
        self.phase_steps = torch.where(self.phase != previous_phase, 0,
                                       self.phase_steps + running[:, None].long())
        for name in ("phase_contact_error", "support_loss", "wrong_contact", "new_support_loss"):
            values[name] = torch.where(running, values[name], 0.0)
        return values


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
