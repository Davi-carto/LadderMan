import torch

from ladder.utils.stop_command import next_valid_frame_per_env, next_zero_command_frame


def test_next_zero_command_frame_selects_current_or_next_valid_frame():
    valid = torch.zeros(12, dtype=torch.bool)
    valid[torch.tensor([2, 5, 9])] = True

    result = next_zero_command_frame(valid, torch.tensor([0, 2, 3, 8, 9, 10]))

    assert result.tolist() == [2, 2, 5, 9, 9, -1]


def test_next_zero_command_frame_handles_empty_mask():
    result = next_zero_command_frame(
        torch.zeros(5, dtype=torch.bool), torch.tensor([0, 4])
    )

    assert result.tolist() == [-1, -1]


def test_next_valid_frame_per_env_uses_each_environment_mask():
    valid = torch.zeros(3, 8, dtype=torch.bool)
    valid[0, torch.tensor([1, 5])] = True
    valid[1, torch.tensor([3, 7])] = True
    valid[2, 0] = True

    result = next_valid_frame_per_env(valid, torch.tensor([2, 3, 1]))

    assert result.tolist() == [5, 3, -1]


def test_next_valid_frame_per_env_validates_shapes():
    try:
        next_valid_frame_per_env(torch.zeros(5, dtype=torch.bool), torch.zeros(1, dtype=torch.long))
    except ValueError as error:
        assert "[N, T]" in str(error)
    else:
        raise AssertionError("expected invalid stop-mask shapes to fail")
