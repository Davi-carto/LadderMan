import pytest


torch = pytest.importorskip("torch")

from ladder.utils.contact_metrics import (
    classify_limb_contacts,
    encode_support_relation,
    stable_anchor_contact_completion,
    stable_anchor_gate_transition,
)


def test_support_relation_keeps_masks_and_relative_rung_order_unambiguous():
    next_ids = torch.tensor([[4, 5, 1, -1], [-1, -1, 0, 0]])
    actual_ids = torch.tensor([[3, -1, 1, 0], [-1, 2, 0, -1]])

    encoded = encode_support_relation(actual_ids, next_ids, num_rungs=7).reshape(2, 4, 3)

    torch.testing.assert_close(encoded[0, :, 0], torch.tensor([1.0, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(encoded[0, :, 1], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(
        encoded[0, :, 2], torch.tensor([2.0 / 7.0, 0.0, 0.0, 0.0])
    )
    # No positive next rung means ground is the shared origin; rung 2 remains normalized.
    assert encoded[1, 1, 2].item() == pytest.approx(2.0 / 7.0)


def test_contact_metrics_separate_reference_required_from_physical_and_target_contact():
    indicator = torch.tensor([[1.0, 0.0, 1.0]])
    positions = torch.tensor([[[0.00, 0.0, 0.0], [0.0, 0.0, 0.0], [0.10, 0.0, 0.0]]])
    targets = torch.zeros_like(positions)
    forces = torch.tensor([[[6.0, 0.0, 0.0], [8.0, 0.0, 0.0], [6.0, 0.0, 0.0]]])

    required, physical, target = classify_limb_contacts(
        indicator, positions, targets, forces, torch.tensor([0.05, 0.05, 0.05]), 5.0
    )

    assert required.tolist() == [[True, False, True]]
    assert physical.tolist() == [[True, True, True]]
    assert target.tolist() == [[True, False, False]]


def test_contact_metrics_reject_invalid_thresholds():
    values = torch.zeros((1, 1, 3))
    with pytest.raises(ValueError, match="force_threshold"):
        classify_limb_contacts(torch.ones((1, 1)), values, values, values, 0.1, -1.0)
    with pytest.raises(ValueError, match="positive"):
        classify_limb_contacts(torch.ones((1, 1)), values, values, values, 0.0, 5.0)


def test_stable_anchor_relaxation_only_ignores_one_old_support_flicker():
    desired = torch.ones((3, 4), dtype=torch.bool)
    actual = desired.clone()
    target = desired.clone()
    newly_required = torch.tensor(
        [[False, True, False, False]] * 3, dtype=torch.bool
    )
    # Row 0 is exact. Row 1 loses an old support and may pass only relaxed.
    # Row 2 loses the newly placed support and must not pass either condition.
    actual[1, 2] = target[1, 2] = False
    actual[2, 1] = target[2, 1] = False
    strict, relaxed, match, target_fraction = stable_anchor_contact_completion(
        desired, actual, target, newly_required
    )
    assert strict.tolist() == [True, False, False]
    assert relaxed.tolist() == [True, True, False]
    torch.testing.assert_close(match, torch.tensor([1.0, 0.75, 0.75]))
    torch.testing.assert_close(target_fraction, torch.tensor([1.0, 0.75, 0.75]))


def test_stable_anchor_relaxation_forbids_unexpected_rung_contact():
    desired = torch.tensor([[False, False, True, True]])
    actual = torch.tensor([[True, False, True, True]])
    target = torch.tensor([[False, False, True, True]])
    newly_required = torch.zeros_like(desired)
    strict, relaxed, _, _ = stable_anchor_contact_completion(
        desired, actual, target, newly_required
    )
    assert not strict.item()
    assert not relaxed.item()


def test_stable_anchor_gate_uses_strict_then_relaxed_and_times_out():
    holding = torch.ones(3, dtype=torch.bool)
    hold_steps = torch.tensor([0, 99, 274])
    confirmations = torch.tensor([3, 3, 0])
    strict = torch.tensor([True, False, False])
    relaxed = torch.tensor([False, True, False])
    next_hold, next_confirmations, release, timeout = stable_anchor_gate_transition(
        holding,
        hold_steps,
        confirmations,
        strict,
        relaxed,
        relaxed_after_steps=100,
        failure_after_steps=275,
        confirmation_steps=4,
    )
    assert next_hold.tolist() == [1, 100, 275]
    assert next_confirmations.tolist() == [4, 4, 0]
    assert release.tolist() == [True, True, False]
    assert timeout.tolist() == [False, False, True]
