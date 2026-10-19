"""Clock-free support-phase state-machine regression tests."""

import torch

from ladder.utils.contact_task import SupportContactPhases


def _step(machine, actual, *, missing=None, near=True, slow=True, safe=True,
          running=True):
    actual = torch.as_tensor(actual, dtype=torch.long).reshape(1, 4)
    if missing is None:
        missing = torch.zeros((1, 4), dtype=torch.long)
    else:
        missing = torch.as_tensor(missing, dtype=torch.long).reshape(1, 4)
    near = (
        torch.full((1, 4), near, dtype=torch.bool)
        if isinstance(near, bool)
        else torch.as_tensor(near, dtype=torch.bool).reshape(1, 4)
    )
    slow = (
        torch.full((1, 4), slow, dtype=torch.bool)
        if isinstance(slow, bool)
        else torch.as_tensor(slow, dtype=torch.bool).reshape(1, 4)
    )
    return machine.step(
        actual, near, slow, torch.tensor([safe]), torch.tensor([running]), missing
    )


def _machine(source=(5, 6, 1, 1), target=(5, 6, 2, 1), initial=None):
    machine = SupportContactPhases(1, "cpu", confirmation_steps=2, release_grace_steps=1)
    source = torch.tensor([source], dtype=torch.long)
    target = torch.tensor([target], dtype=torch.long)
    initial = source if initial is None else torch.tensor([initial], dtype=torch.long)
    machine.reset(torch.tensor([0]), source, target, initial)
    return machine


def test_release_and_acquire_are_single_events_with_dynamic_expected_ids():
    machine = _machine()
    assert machine.expected.tolist() == [[5, 6, 1, 1]]
    _step(machine, [5, 6, 1, 1])
    _step(machine, [5, 6, 1, 1])
    assert machine.expected.tolist() == [[5, 6, -1, 1]]

    first = _step(machine, [5, 6, -1, 1])
    second = _step(machine, [5, 6, -1, 1])
    assert first["release_event"].item() == 0.0
    assert second["release_event"].item() == 1.0
    assert machine.expected.tolist() == [[5, 6, -1, 1]]

    first = _step(machine, [5, 6, 2, 1])
    second = _step(machine, [5, 6, 2, 1])
    assert first["contact_event"].item() == 0.0
    assert second["contact_event"].item() == 1.0
    assert machine.expected.tolist() == [[5, 6, 2, 1]]
    assert _step(machine, [5, 6, 2, 1])["contact_event"].item() == 0.0


def test_acquired_support_loss_is_not_waived_after_sensor_grace():
    machine = _machine()
    _step(machine, [5, 6, 1, 1])
    _step(machine, [5, 6, 1, 1])
    _step(machine, [5, 6, -1, 1])
    _step(machine, [5, 6, -1, 1])
    _step(machine, [5, 6, 2, 1])
    _step(machine, [5, 6, 2, 1])
    assert machine.acquired[0, 2]

    assert _step(machine, [5, 6, -1, 1], missing=[0, 0, 1, 0])["support_loss"].item() == 0.0
    assert _step(machine, [5, 6, -1, 1], missing=[0, 0, 3, 0])["support_loss"].item() == 1.0
    for _ in range(3):
        assert _step(machine, [5, 6, 2, 1])["contact_event"].item() == 0.0


def test_reset_in_flight_does_not_pay_past_release_or_contact_events():
    machine = _machine(initial=[5, 6, -1, 1])
    assert machine.phase[0, 2] == machine.MOVE
    assert machine.released[0, 2]
    assert _step(machine, [5, 6, -1, 1])["release_event"].item() == 0.0

    machine = _machine(initial=[5, 6, 2, 1])
    assert machine.phase[0, 2] == machine.ACQUIRE
    assert machine.acquired[0, 2]
    assert _step(machine, [5, 6, 2, 1])["contact_event"].item() == 0.0


def test_batched_env_reset_and_state_updates_are_independent():
    machine = SupportContactPhases(2, "cpu", confirmation_steps=2, release_grace_steps=1)
    source = torch.tensor([[5, 6, 1, 1], [5, 6, 1, 1]])
    target = torch.tensor([[5, 6, 2, 1], [5, 6, 2, 1]])
    machine.reset(torch.tensor([0, 1]), source, target, source)
    # Exercise the actual batched path directly; _step is intentionally single-env.
    actual = torch.tensor([[5, 6, 1, 1], [5, 6, 1, 1]])
    flags = torch.ones((2, 4), dtype=torch.bool)
    machine.step(actual, flags, flags, torch.ones(2, dtype=torch.bool),
                 torch.ones(2, dtype=torch.bool), torch.zeros((2, 4), dtype=torch.long))
    actual[1, 2] = -1
    machine.step(actual, flags, flags, torch.ones(2, dtype=torch.bool),
                 torch.ones(2, dtype=torch.bool), torch.zeros((2, 4), dtype=torch.long))
    assert machine.phase[0, 2] == machine.RELEASE
    assert machine.phase[1, 2] == machine.PREPARE
    assert machine.prepare_count.tolist() == [2, 0]
    machine.safe_release_count[:] = 1
    machine.reset(torch.tensor([1]), source[1:], target[1:], source[1:])
    assert machine.phase[0, 2] == machine.RELEASE
    assert machine.phase[1, 2] == machine.PREPARE
    assert machine.prepare_count.tolist() == [2, 0]
    assert machine.safe_release_count.tolist() == [[1, 1, 1, 1], [0, 0, 0, 0]]


def test_prepare_sensor_flicker_does_not_advance_but_sustained_early_release_does():
    machine = _machine()
    for missing in (1, 2):
        result = _step(machine, [5, 6, -1, 1], missing=[0, 0, missing, 0], near=False)
        assert machine.phase[0, 2] == machine.PREPARE
        assert result["release_event"].item() == 0.0
        assert result["support_loss"].item() == 0.0
    for missing in (3, 4):
        result = _step(machine, [5, 6, -1, 1], missing=[0, 0, missing, 0], near=False)
        assert result["release_event"].item() == 0.0
        assert result["support_loss"].item() == 1.0
    assert machine.phase[0, 2] == machine.MOVE
    assert machine.released[0, 2]
    result = _step(machine, [5, 6, -1, 1], missing=[0, 0, 5, 0], near=False)
    assert result["support_loss"].item() == 0.0
    assert result["release_event"].item() == 0.0


def test_unsafe_physical_release_advances_without_reward_or_late_payout():
    machine = _machine()
    _step(machine, [5, 6, 1, 1])
    _step(machine, [5, 6, 1, 1])
    for missing in (1, 2):
        result = _step(machine, [5, 6, -1, 1], missing=[0, 0, missing, 0],
                       near=False, safe=False)
        assert result["release_event"].item() == 0.0
    assert machine.phase[0, 2] == machine.MOVE
    for missing in (3, 4, 5):
        result = _step(machine, [5, 6, -1, 1], missing=[0, 0, missing, 0], near=False)
        assert result["release_event"].item() == 0.0


def test_release_bonus_requires_consecutive_safe_steps():
    machine = _machine()
    _step(machine, [5, 6, 1, 1])
    _step(machine, [5, 6, 1, 1])
    _step(machine, [5, 6, -1, 1], near=False, safe=False)
    result = _step(machine, [5, 6, -1, 1], near=False, safe=True)
    assert machine.phase[0, 2] == machine.MOVE
    assert result["release_event"].item() == 0.0


def test_other_limb_in_flight_or_unconfirmed_acquire_does_not_block_prepare():
    for flight_near in (False, True):
        machine = _machine(source=(5, 6, 1, 1), target=(6, 7, 1, 1),
                           initial=(-1, 6, 1, 1))
        near = [flight_near, False, True, True]
        _step(machine, [-1, 6, 1, 1], near=near)
        _step(machine, [-1, 6, 1, 1], near=near)
        assert machine.phase[0, 0] == (machine.ACQUIRE if flight_near else machine.MOVE)
        assert machine.phase[0, 1] == machine.RELEASE
        assert machine.expected.tolist() == [[6 if flight_near else -1, -1, 1, 1]]


def test_established_support_loss_still_blocks_prepare_and_release_bonus():
    machine = _machine(source=(5, 6, 1, 1), target=(6, 7, 1, 1),
                       initial=(6, 6, 1, 1))
    for _ in range(3):
        result = _step(machine, [-1, 6, 1, 1], missing=[3, 0, 0, 0])
        assert result["support_loss"].item() == 1.0
    assert machine.phase[0, 1] == machine.PREPARE
    machine = _machine()
    _step(machine, [5, 6, 1, 1])
    _step(machine, [5, 6, 1, 1])
    for _ in range(2):
        result = _step(machine, [5, -1, -1, 1], missing=[0, 3, 3, 0], near=False)
        assert result["release_event"].item() == 0.0
        assert result["support_loss"].item() == 1.0
    assert machine.phase[0, 2] == machine.MOVE


def test_multi_limb_milestone_budget_and_running_mask():
    machine = _machine(source=(5, 6, 1, 1), target=(6, 7, 1, 1))
    _step(machine, [5, 6, 1, 1])
    _step(machine, [5, 6, 1, 1])
    for _ in range(3):
        result = _step(machine, [-1, -1, 1, 1], near=False, running=False)
        assert result["release_event"].item() == 0.0
        assert machine.phase[0, :2].tolist() == [machine.RELEASE, machine.RELEASE]
    _step(machine, [-1, -1, 1, 1], near=False)
    result = _step(machine, [-1, -1, 1, 1], near=False)
    assert result["release_event"].item() == 1.0
    _step(machine, [6, 7, 1, 1])
    result = _step(machine, [6, 7, 1, 1])
    assert result["contact_event"].item() == 1.0
    for _ in range(3):
        result = _step(machine, [6, 7, 1, 1])
        assert result["release_event"].item() == 0.0
        assert result["contact_event"].item() == 0.0
