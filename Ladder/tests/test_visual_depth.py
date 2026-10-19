import torch

from ladder.utils.visual_depth import (
    DepthAugmentationCfg,
    add_depth_noise_and_dropout,
    apply_rung_focused_masking,
    prepare_metric_depth,
    process_visual_depth,
    process_warp_visual_depth,
    rfm_keep_mask_from_semantic,
    rfm_semantic_ids_from_info,
)


def test_prepare_metric_depth_returns_finite_nchw_tensor():
    depth = torch.tensor([[[[float("nan")], [float("inf")]], [[0.05], [3.0]]]])

    prepared = prepare_metric_depth(depth)

    assert prepared.shape == (1, 1, 2, 2)
    torch.testing.assert_close(prepared, torch.tensor([[[[2.0, 2.0], [0.1, 2.0]]]]))


def test_rfm_mask_uses_runtime_rung_and_robot_labels_only():
    semantic = torch.tensor([[[[0], [1], [2]], [[3], [4], [2]]]], dtype=torch.int32)
    semantic_info = {
        "idToLabels": {
            "0": {"class": "BACKGROUND"},
            "1": {"class": "UNLABELLED"},
            "2": {"class": "rung"},
            "3": {"class": "ladder"},
            "4": {"class": "robot"},
        }
    }

    mask = rfm_keep_mask_from_semantic(semantic, semantic_info)

    assert rfm_semantic_ids_from_info(semantic_info) == (2, 4)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[[[False, False, True], [False, True, True]]]]


def test_rfm_mask_is_empty_until_semantic_metadata_is_available():
    semantic = torch.tensor([[[[2], [4]]]], dtype=torch.int32)

    assert not rfm_keep_mask_from_semantic(semantic, None).any()


def test_depth_dropout_uses_invalid_zero_value():
    cfg = DepthAugmentationCfg(gaussian_std=0.0, pixel_dropout_probability=1.0)

    augmented = add_depth_noise_and_dropout(torch.ones(2, 1, 3, 4), cfg)

    assert torch.count_nonzero(augmented) == 0


def test_rung_focused_masking_only_removes_non_rung_pixels():
    depth = torch.ones(2, 1, 2, 2)
    rung_mask = torch.tensor(
        [[[[True, False], [False, True]]], [[[False, True], [True, False]]]]
    )

    masked, applied = apply_rung_focused_masking(depth, rung_mask, probability=1.0)

    assert applied.tolist() == [True, True]
    torch.testing.assert_close(masked, rung_mask.float())


def test_process_visual_depth_forced_rfm_excludes_noise_branch():
    depth = torch.full((1, 2, 2, 1), 0.7)
    semantic = torch.tensor([[[[0], [2]], [[3], [4]]]], dtype=torch.int32)
    semantic_info = {
        "idToLabels": {
            "0": {"class": "BACKGROUND"},
            "2": {"class": "rung"},
            "3": {"class": "ladder"},
            "4": {"class": "robot"},
        }
    }
    cfg = DepthAugmentationCfg(gaussian_std=0.0, pixel_dropout_probability=1.0)

    processed, applied = process_visual_depth(
        depth, semantic, semantic_info, cfg, rfm_probability=1.0
    )

    assert applied.tolist() == [True]
    torch.testing.assert_close(processed, torch.tensor([[[[0.0, 0.7], [0.0, 0.7]]]]))


def test_process_visual_depth_without_rfm_uses_noise_branch():
    depth = torch.full((1, 2, 2, 1), 0.7)
    semantic = torch.ones((1, 2, 2, 1), dtype=torch.int32)
    semantic_info = {"idToLabels": {"1": {"class": "UNLABELLED"}}}
    cfg = DepthAugmentationCfg(gaussian_std=0.0, pixel_dropout_probability=1.0)

    processed, applied = process_visual_depth(
        depth, semantic, semantic_info, cfg, rfm_probability=0.0
    )

    assert applied.tolist() == [False]
    assert torch.count_nonzero(processed) == 0


def test_process_warp_visual_depth_uses_exact_rung_and_robot_classes():
    depth = torch.full((1, 2, 3), 0.7)
    classes = torch.tensor([[[0, 1, 2], [3, 2, 0]]], dtype=torch.int32)
    cfg = DepthAugmentationCfg(gaussian_std=0.0, pixel_dropout_probability=1.0)

    processed, applied = process_warp_visual_depth(
        depth,
        classes,
        rung_class=1,
        robot_class=2,
        cfg=cfg,
        rfm_probability=1.0,
    )

    assert applied.tolist() == [True]
    torch.testing.assert_close(
        processed,
        torch.tensor([[[[0.0, 0.7, 0.7], [0.0, 0.7, 0.0]]]]),
    )
