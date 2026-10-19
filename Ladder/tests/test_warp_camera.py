from dataclasses import dataclass
import math
from pathlib import Path

import pytest
import torch
import warp as wp

from ladder.utils.warp_camera import (
    ROBOT,
    RUNG,
    CR1WarpBatchCamera,
    CR1WarpCamera,
    WarpCameraCfg,
    WarpCameraOutput,
    WARP_CENTER_RAY_VALUE_NAMES,
    WARP_DIAGNOSTIC_COUNT_NAMES,
    aperture_from_fov,
    load_robot_collision_geometries,
    pinhole_fov_remap_grid,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CR1_URDF = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)


@dataclass(frozen=True)
class _BoxSpec:
    name: str
    size: tuple[float, float, float]


def test_default_projection_and_zero_crop_keep_80_by_60():
    cfg = WarpCameraCfg()
    image = torch.arange(60 * 80).reshape(60, 80)

    assert cfg.output_width == 80
    assert cfg.output_height == 60
    torch.testing.assert_close(cfg.crop(image), image)
    assert cfg.horizontal_tangent == pytest.approx(math.tan(math.radians(87.0) / 2.0))
    assert cfg.vertical_tangent == pytest.approx(math.tan(math.radians(58.0) / 2.0))


def test_crop_changes_output_shape_without_changing_projection():
    full = WarpCameraCfg()
    cropped = WarpCameraCfg(crop_top=2, crop_bottom=3, crop_left=4, crop_right=5)

    assert (cropped.output_height, cropped.output_width) == (55, 71)
    assert cropped.crop(torch.zeros(60, 80)).shape == (55, 71)
    assert cropped.horizontal_tangent == full.horizontal_tangent
    assert cropped.vertical_tangent == full.vertical_tangent


def test_80_by_60_usd_camera_derives_vertical_fov_from_square_pixels():
    focal_length = 24.0
    horizontal_aperture = aperture_from_fov(87.0, focal_length)
    focal_pixels = 80.0 * focal_length / horizontal_aperture
    vertical = math.degrees(2.0 * math.atan(60.0 / (2.0 * focal_pixels)))

    assert vertical == pytest.approx(70.8807, abs=1.0e-4)


def test_pinhole_remap_grid_is_identity_for_matching_intrinsics():
    width, height = 80, 60
    horizontal_fov = 87.0
    vertical_fov = 58.0
    intrinsics = torch.tensor(
        [
            [width / (2.0 * math.tan(math.radians(horizontal_fov) / 2.0)), 0.0, width / 2.0],
            [0.0, height / (2.0 * math.tan(math.radians(vertical_fov) / 2.0)), height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    ).numpy()

    map_x, map_y = pinhole_fov_remap_grid(
        intrinsics,
        output_width=width,
        output_height=height,
        horizontal_fov_deg=horizontal_fov,
        vertical_fov_deg=vertical_fov,
    )

    torch.testing.assert_close(torch.from_numpy(map_x[0]), torch.arange(width).float())
    torch.testing.assert_close(torch.from_numpy(map_y[:, 0]), torch.arange(height).float())


def test_pinhole_remap_grid_crops_square_pixel_rtx_vertical_fov():
    width, height = 80, 60
    focal = width / (2.0 * math.tan(math.radians(87.0) / 2.0))
    intrinsics = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    ).numpy()

    map_x, map_y = pinhole_fov_remap_grid(
        intrinsics,
        output_width=width,
        output_height=height,
        horizontal_fov_deg=87.0,
        vertical_fov_deg=58.0,
    )

    torch.testing.assert_close(torch.from_numpy(map_x[0]), torch.arange(width).float())
    assert map_y[0, 0] == pytest.approx(6.52462, abs=1.0e-5)
    assert map_y[-1, 0] == pytest.approx(52.47538, abs=1.0e-5)


def test_rfm_output_is_exact_rung_and_robot_union():
    output = WarpCameraOutput(
        depth=torch.tensor([[0.5, 0.6, 0.7, 0.8]]),
        hit_class=torch.tensor([[0, RUNG, ROBOT, 3]], dtype=torch.int32),
    )

    assert output.rung_mask.tolist() == [[False, True, False, False]]
    assert output.robot_mask.tolist() == [[False, False, True, False]]
    assert output.rfm_keep_mask.tolist() == [[False, True, True, False]]
    torch.testing.assert_close(output.rfm_depth, torch.tensor([[0.0, 0.6, 0.7, 0.0]]))


def test_cr1_urdf_parser_loads_30_meshes_and_two_hand_spheres():
    geometries = load_robot_collision_geometries(CR1_URDF)
    meshes = [geometry for geometry in geometries if geometry.kind == "mesh"]
    spheres = [geometry for geometry in geometries if geometry.kind == "sphere"]

    assert len(geometries) == 32
    assert len(meshes) == 30
    assert len(spheres) == 2
    assert {geometry.link_name for geometry in spheres} == {
        "left_sphere_hand_link",
        "right_sphere_hand_link",
    }
    assert all(geometry.mesh_path.is_file() for geometry in meshes)
    assert all(geometry.sphere_radius == pytest.approx(0.045) for geometry in spheres)


def test_warp_kernel_selects_nearest_ladder_and_robot_hits(tmp_path, monkeypatch):
    urdf = tmp_path / "sphere_robot.urdf"
    urdf.write_text(
        """<robot name="sphere_robot">
  <link name="sphere_link">
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.1"/></geometry>
    </collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    cfg = WarpCameraCfg(width=1, height=1, horizontal_fov_deg=87.0, vertical_fov_deg=58.0)
    renderer = CR1WarpCamera(
        cfg,
        robot_urdf=urdf,
        robot_body_names=["sphere_link"],
        ladder_box_specs=(_BoxSpec("step_01", (0.5, 0.5, 0.5)),),
        ladder_object_names=["step_01"],
        rung_box_names=("step_01",),
        device="cpu",
    )
    camera_position = torch.tensor([0.0, 0.0, 1.0])
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    synchronize_calls = []
    synchronize_device = wp.synchronize_device

    def record_synchronize(device):
        synchronize_calls.append(str(device))
        synchronize_device(device)

    monkeypatch.setattr(wp, "synchronize_device", record_synchronize)

    ladder_hit = renderer.render(
        camera_position_w=camera_position,
        camera_quaternion_wxyz_opengl=identity,
        robot_body_positions_w=torch.tensor([[10.0, 0.0, 0.5]]),
        robot_body_quaternions_wxyz=identity.reshape(1, 4),
        ladder_positions_w=torch.tensor([[0.0, 0.0, 0.5]]),
        ladder_quaternions_wxyz=identity.reshape(1, 4),
        ground_height=0.0,
        collect_diagnostics=True,
    )
    assert ladder_hit.hit_class.item() == RUNG
    assert ladder_hit.depth.item() == pytest.approx(0.25, abs=1.0e-5)
    diagnostic_counts = dict(
        zip(WARP_DIAGNOSTIC_COUNT_NAMES, ladder_hit.diagnostic_counts.tolist())
    )
    center_ray = dict(zip(WARP_CENTER_RAY_VALUE_NAMES, ladder_hit.center_ray_values.tolist()))
    assert diagnostic_counts["rays_total"] == 1
    assert diagnostic_counts["rays_hitting_rung_box"] == 1
    assert diagnostic_counts["final_rung_pixels"] == 1
    assert center_ray["world_ray_z"] == pytest.approx(-1.0)
    assert center_ray["near_ray_distance"] == pytest.approx(0.1)
    assert center_ray["nearest_ray_distance"] == pytest.approx(0.25)

    near_clipped_robot = renderer.render(
        camera_position_w=camera_position,
        camera_quaternion_wxyz_opengl=identity,
        robot_body_positions_w=torch.tensor([[0.0, 0.0, 1.05]]),
        robot_body_quaternions_wxyz=identity.reshape(1, 4),
        ladder_positions_w=torch.tensor([[0.0, 0.0, 0.5]]),
        ladder_quaternions_wxyz=identity.reshape(1, 4),
        ground_height=0.0,
        include_ground=False,
    )
    assert near_clipped_robot.hit_class.item() == RUNG
    assert near_clipped_robot.depth.item() == pytest.approx(0.25, abs=1.0e-5)

    robot_hit = renderer.render(
        camera_position_w=camera_position,
        camera_quaternion_wxyz_opengl=identity,
        robot_body_positions_w=torch.tensor([[0.0, 0.0, 0.5]]),
        robot_body_quaternions_wxyz=identity.reshape(1, 4),
        ladder_positions_w=torch.tensor([[10.0, 0.0, 0.5]]),
        ladder_quaternions_wxyz=identity.reshape(1, 4),
        ground_height=0.0,
        include_ladder=False,
        include_ground=False,
        collect_diagnostics=True,
    )
    assert robot_hit.hit_class.item() == ROBOT
    assert robot_hit.depth.item() == pytest.approx(0.4, abs=1.0e-5)
    robot_counts = dict(zip(WARP_DIAGNOSTIC_COUNT_NAMES, robot_hit.diagnostic_counts.tolist()))
    assert robot_counts["rays_hitting_robot_sphere"] == 1
    assert robot_counts["rays_hitting_ground_plane"] == 0
    assert robot_counts["final_robot_pixels"] == 1
    assert synchronize_calls == ["cpu", "cpu", "cpu"]


def test_batch_warp_camera_renders_multiple_environment_classes_without_frame_sync(tmp_path, monkeypatch):
    urdf = tmp_path / "sphere_robot.urdf"
    urdf.write_text(
        """<robot name="sphere_robot">
  <link name="sphere_link">
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.1"/></geometry>
    </collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    renderer = CR1WarpBatchCamera(
        WarpCameraCfg(width=1, height=1),
        num_envs=2,
        robot_urdf=urdf,
        robot_body_names=["sphere_link"],
        ladder_box_specs=(_BoxSpec("step_01", (0.5, 0.5, 0.5)),),
        ladder_object_names=["step_01"],
        rung_box_names=("step_01",),
        device="cpu",
    )
    synchronize_calls = []
    synchronize_device = wp.synchronize_device

    def record_synchronize(device):
        synchronize_calls.append(str(device))
        synchronize_device(device)

    monkeypatch.setattr(wp, "synchronize_device", record_synchronize)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    output = renderer.render(
        camera_positions_w=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        camera_quaternions_wxyz_opengl=identity,
        robot_body_positions_w=torch.tensor([[[0.0, 0.0, 0.5]], [[10.0, 0.0, 0.5]]]),
        robot_body_quaternions_wxyz=identity.unsqueeze(1),
        ladder_positions_w=torch.tensor([[[10.0, 0.0, 0.5]], [[0.0, 0.0, 0.5]]]),
        ladder_quaternions_wxyz=identity.unsqueeze(1),
        ground_height=0.0,
        include_ground=False,
    )

    assert tuple(output.depth.shape) == (2, 1, 1)
    assert output.hit_class[:, 0, 0].tolist() == [ROBOT, RUNG]
    torch.testing.assert_close(output.depth[:, 0, 0], torch.tensor([0.4, 0.25]))
    assert synchronize_calls == []


def test_batch_warp_camera_writes_partial_batch_to_requested_environment_rows(tmp_path):
    urdf = tmp_path / "sphere_robot.urdf"
    urdf.write_text(
        """<robot name="sphere_robot">
  <link name="sphere_link">
    <collision><geometry><sphere radius="0.1"/></geometry></collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    renderer = CR1WarpBatchCamera(
        WarpCameraCfg(width=1, height=1),
        num_envs=3,
        robot_urdf=urdf,
        robot_body_names=["sphere_link"],
        ladder_box_specs=(_BoxSpec("step_01", (0.5, 0.5, 0.5)),),
        ladder_object_names=["step_01"],
        rung_box_names=("step_01",),
        device="cpu",
    )
    output = renderer.render(
        camera_positions_w=torch.tensor([[0.0, 0.0, 1.0]]),
        camera_quaternions_wxyz_opengl=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        robot_body_positions_w=torch.tensor([[[10.0, 0.0, 0.5]]]),
        robot_body_quaternions_wxyz=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        ladder_positions_w=torch.tensor([[[0.0, 0.0, 0.5]]]),
        ladder_quaternions_wxyz=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        ground_height=0.0,
        env_ids=torch.tensor([2]),
        include_ground=False,
    )

    assert tuple(output.depth.shape) == (1, 1, 1)
    assert output.hit_class.item() == RUNG
    assert renderer._hit_class[2, 0, 0].item() == RUNG
