from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from ladder.tasks.ladder_climb.ladder_geometry import (
    LADDER_A,
    LadderGeometry,
    limb_lateral_offsets_tensor,
)
from ladder.utils.contact_force import (
    filtered_surface_force_max,
    non_target_rung_forces,
    target_rung_forces,
)
from ladder.utils.contact_metrics import (
    classify_limb_contacts,
    stable_anchor_contact_completion,
    stable_anchor_gate_transition,
)
from ladder.utils.contact_schema import DEFAULT_LIMB_BODY_NAMES, LIMB_NAMES
from ladder.utils.motion_reset import (
    load_reset_valid_mask,
    mix_nominal_ladder_geometry_samples,
    mix_reset_frame_samples,
    mix_two_anchor_reset_frame_samples,
    pose_range_bounds,
    sample_reset_frames_by_segments,
    select_geometry_cell_reset_masks,
)
from ladder.utils.motion_semantics import (
    direction_bounded_limb_goals,
    stable_motion_target_frames,
)
from ladder.utils.stop_command import (
    STOP_STATE_HOLD,
    STOP_STATE_RUN,
    STOP_STATE_SEEK,
    next_valid_frame_per_env,
    next_zero_command_frame,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_REFERENCE_PROGRESS_DISTANCE_THRESHOLD = 0.05
_REFERENCE_MOVING_SPEED_THRESHOLD = 0.05
_ROBOT_STATIONARY_SPEED_THRESHOLD = 0.03


class LadderMotionLoader:
    """Load LadderMan motion-tracking references with contact annotations."""

    def __init__(
        self,
        motion_file: str,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        device: str = "cpu",
    ):
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(
                f"Invalid motion file path: {motion_file}. "
                "Provide a LadderMan climbing motion npz before creating the environment."
            )
        data = np.load(motion_file)
        required_keys = (
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "contact_indicator",
            "contact_rung_id",
            "joint_names",
            "body_names",
        )
        missing_keys = [key for key in required_keys if key not in data.files]
        if missing_keys:
            raise KeyError(
                f"Ladder climbing motion file '{motion_file}' is missing {missing_keys}. "
                "Expected the whole-body tracking motion keys plus contact_indicator/contact_rung_id."
            )

        motion_joint_names = [str(name) for name in data["joint_names"].tolist()]
        motion_body_names = [str(name) for name in data["body_names"].tolist()]
        missing_joints = [name for name in joint_names if name not in motion_joint_names]
        missing_bodies = [name for name in body_names if name not in motion_body_names]
        if missing_joints or missing_bodies:
            raise ValueError(
                f"motion is missing robot joints {missing_joints} or tracking bodies {missing_bodies}"
            )
        joint_indexes = [motion_joint_names.index(name) for name in joint_names]
        body_indexes = [motion_body_names.index(name) for name in body_names]

        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        self.joint_pos = torch.tensor(data["joint_pos"][:, joint_indexes], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"][:, joint_indexes], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.contact_indicator = torch.tensor(data["contact_indicator"], dtype=torch.float32, device=device)
        self.contact_rung_id = torch.tensor(data["contact_rung_id"], dtype=torch.long, device=device)
        has_stable_support_contact = "stable_support_contact_indicator" in data.files
        has_stable_support_rung = "stable_support_rung_id" in data.files
        if has_stable_support_contact != has_stable_support_rung:
            raise KeyError(
                "stable_support_contact_indicator and stable_support_rung_id must be provided together"
            )
        if has_stable_support_contact:
            self.stable_support_contact_indicator = torch.tensor(
                data["stable_support_contact_indicator"],
                dtype=torch.float32,
                device=device,
            )
            self.stable_support_rung_id = torch.tensor(
                data["stable_support_rung_id"], dtype=torch.long, device=device
            )
            if "limb_body_names" not in data.files:
                raise KeyError(
                    "ground-aware stable support requires limb_body_names in the motion"
                )
            support_limb_body_names = [
                str(name) for name in data["limb_body_names"].tolist()
            ]
            missing_support_bodies = [
                name for name in support_limb_body_names if name not in motion_body_names
            ]
            if missing_support_bodies:
                raise ValueError(
                    "motion is missing stable-support limb bodies "
                    f"{missing_support_bodies}"
                )
            self._stable_support_limb_body_indexes = torch.tensor(
                [motion_body_names.index(name) for name in support_limb_body_names],
                dtype=torch.long,
                device=device,
            )
        else:
            legacy_contact = self.contact_indicator > 0.5
            self.stable_support_contact_indicator = self.contact_indicator.clone()
            self.stable_support_rung_id = torch.where(
                legacy_contact,
                self.contact_rung_id + 1,
                torch.full_like(self.contact_rung_id, -1),
            )
            self._stable_support_limb_body_indexes = None
        expected_contact_shape = (self.joint_pos.shape[0], len(LIMB_NAMES))
        if self.stable_support_contact_indicator.shape != expected_contact_shape:
            raise ValueError(
                "stable-support contact annotations must have shape "
                f"{expected_contact_shape}, got "
                f"{tuple(self.stable_support_contact_indicator.shape)}"
            )
        if self.stable_support_rung_id.shape != expected_contact_shape:
            raise ValueError(
                "stable-support rung IDs must match the stable-support contacts"
            )
        stable_support_active = self.stable_support_contact_indicator > 0.5
        if torch.any(self.stable_support_rung_id[stable_support_active] < 0):
            raise ValueError("active stable supports must have nonnegative support IDs")
        if torch.any(self.stable_support_rung_id[~stable_support_active] != -1):
            raise ValueError("inactive stable supports must use support ID -1")
        climb_direction = (
            np.asarray(data["climb_direction"], dtype=np.float32).reshape(-1)
            if "climb_direction" in data.files
            else np.ones(self.joint_pos.shape[0], dtype=np.float32)
        )
        if climb_direction.shape != (self.joint_pos.shape[0],):
            raise ValueError("climb_direction must contain exactly one value per motion frame")
        if not np.all(np.isin(climb_direction, (-1.0, 1.0))):
            raise ValueError("motion climb_direction values must be -1 or +1")
        self.climb_direction = torch.tensor(
            climb_direction, dtype=torch.float32, device=device
        )
        task_goal_rung_id, task_goal_valid = direction_bounded_limb_goals(
            data["contact_indicator"], data["contact_rung_id"], climb_direction
        )
        self.task_goal_rung_id = torch.tensor(
            task_goal_rung_id, dtype=torch.long, device=device
        )
        self.task_goal_valid = torch.tensor(
            task_goal_valid, dtype=torch.bool, device=device
        )
        zero_command_valid_mask = (
            np.asarray(data["zero_command_valid_mask"], dtype=np.bool_).reshape(-1)
            if "zero_command_valid_mask" in data.files
            else np.zeros(self.joint_pos.shape[0], dtype=np.bool_)
        )
        if zero_command_valid_mask.shape != (self.joint_pos.shape[0],):
            raise ValueError("zero_command_valid_mask must contain exactly one value per motion frame")
        self.zero_command_valid_mask = torch.tensor(
            zero_command_valid_mask, dtype=torch.bool, device=device
        )
        natural_stop_frames = (
            np.asarray(
                data["zero_command_all_limb_contact_segment_selected_frames"],
                dtype=np.int64,
            ).reshape(-1)
            if "zero_command_all_limb_contact_segment_selected_frames" in data.files
            else np.empty(0, dtype=np.int64)
        )
        if np.any(natural_stop_frames < 0) or np.any(
            natural_stop_frames >= self.joint_pos.shape[0]
        ):
            raise ValueError("natural zero-command stop frames lie outside the motion")
        if natural_stop_frames.size and not np.all(
            zero_command_valid_mask[natural_stop_frames]
        ):
            raise ValueError("every natural zero-command stop frame must be marked valid")
        self.zero_command_natural_stop_frames = torch.tensor(
            natural_stop_frames, dtype=torch.long, device=device
        )
        direct_hold_frames = (
            np.asarray(data["zero_command_direct_hold_frames"], dtype=np.int64).reshape(-1)
            if "zero_command_direct_hold_frames" in data.files
            else natural_stop_frames
        )
        if np.any(direct_hold_frames < 0) or np.any(
            direct_hold_frames >= self.joint_pos.shape[0]
        ):
            raise ValueError("zero-command direct-hold frames lie outside the motion")
        if direct_hold_frames.size and not np.all(
            zero_command_valid_mask[direct_hold_frames]
        ):
            raise ValueError("every zero-command direct-hold frame must be marked valid")
        self.zero_command_direct_hold_frames = torch.tensor(
            direct_hold_frames, dtype=torch.long, device=device
        )
        stable_support_anchor_frames = (
            np.asarray(data["stable_support_anchor_frames"], dtype=np.int64).reshape(-1)
            if "stable_support_anchor_frames" in data.files
            else direct_hold_frames
        )
        if np.any(stable_support_anchor_frames < 0) or np.any(
            stable_support_anchor_frames >= self.joint_pos.shape[0]
        ):
            raise ValueError("stable-support anchor frame lies outside the motion")
        if stable_support_anchor_frames.size and not np.all(
            np.diff(stable_support_anchor_frames) > 0
        ):
            raise ValueError("stable-support anchor frames must be strictly increasing")
        self.stable_motion_anchor_frames = torch.tensor(
            stable_support_anchor_frames, dtype=torch.long, device=device
        )
        if "stable_support_anchor_kind" in data.files:
            anchor_kind = np.asarray(
                data["stable_support_anchor_kind"], dtype=np.int64
            ).reshape(-1)
        else:
            # Legacy command artifacts used direct-hold frames as anchors.  Do
            # not claim that a non-contact standing frame is four-limb
            # support; mark it as a special stable frame for compatibility.
            legacy_support_contact = (
                np.asarray(data["stable_support_contact_indicator"])
                if has_stable_support_contact
                else np.asarray(data["contact_indicator"])
            )
            legacy_all_contact = np.all(legacy_support_contact > 0.5, axis=1)
            anchor_kind = np.where(
                legacy_all_contact[stable_support_anchor_frames], 1, 2
            ).astype(np.int64)
        if anchor_kind.shape != stable_support_anchor_frames.shape:
            raise ValueError(
                "stable_support_anchor_kind must contain one value per anchor"
            )
        if np.any(~np.isin(anchor_kind, (0, 1, 2, 3))):
            raise ValueError("stable-support anchor kinds must be 0, 1, 2, or 3")
        anchor_is_full_support = (
            np.asarray(
                data["stable_support_anchor_is_full_support"], dtype=np.bool_
            ).reshape(-1)
            if "stable_support_anchor_is_full_support" in data.files
            else anchor_kind == 1
        )
        if anchor_is_full_support.shape != stable_support_anchor_frames.shape:
            raise ValueError(
                "stable_support_anchor_is_full_support must contain one value per anchor"
            )
        if not np.array_equal(anchor_is_full_support, anchor_kind == 1):
            raise ValueError(
                "stable-support anchor full-support flags disagree with anchor kinds"
            )
        self.stable_motion_anchor_kind = torch.tensor(
            anchor_kind, dtype=torch.long, device=device
        )
        self.stable_motion_anchor_full_support = torch.tensor(
            anchor_is_full_support, dtype=torch.bool, device=device
        )
        if self.stable_motion_anchor_frames.numel() > 0 and not torch.all(
            self.stable_support_contact_indicator[
                self.stable_motion_anchor_frames[self.stable_motion_anchor_full_support]
            ]
            > 0.5
        ):
            raise ValueError(
                "every full-support stable anchor must have all configured limbs in contact"
            )

        # Keep reset partitioning in the motion artifact so reset, playback,
        # and reference gating share the exact same anchor boundaries.  Older
        # artifacts fall back to the equivalent partition derived from anchors.
        reset_segment_fields = (
            "stable_support_reset_segment_starts",
            "stable_support_reset_segment_ends",
            "stable_support_reset_segment_anchor_frames",
            "stable_support_reset_segment_ids",
        )
        saved_reset_segment_fields = [
            name in data.files for name in reset_segment_fields
        ]
        if any(saved_reset_segment_fields) and not all(saved_reset_segment_fields):
            raise KeyError(
                "stable-support reset segment fields must be provided together"
            )
        has_saved_reset_segments = all(saved_reset_segment_fields)
        default_boundaries = np.unique(
            np.concatenate(
                (
                    np.asarray([0], dtype=np.int64),
                    stable_support_anchor_frames,
                    np.asarray([self.joint_pos.shape[0] - 1], dtype=np.int64),
                )
            )
        )
        if default_boundaries.size >= 2:
            default_segment_starts = default_boundaries[:-1].copy()
            default_segment_ends = default_boundaries[1:] - 1
            default_segment_ends[-1] = self.joint_pos.shape[0] - 1
        else:
            default_segment_starts = np.asarray([0], dtype=np.int64)
            default_segment_ends = np.asarray(
                [self.joint_pos.shape[0] - 1], dtype=np.int64
            )
        segment_starts = (
            np.asarray(
                data["stable_support_reset_segment_starts"], dtype=np.int64
            ).reshape(-1)
            if "stable_support_reset_segment_starts" in data.files
            else default_segment_starts
        )
        segment_ends = (
            np.asarray(
                data["stable_support_reset_segment_ends"], dtype=np.int64
            ).reshape(-1)
            if "stable_support_reset_segment_ends" in data.files
            else default_segment_ends
        )
        segment_anchor_frames = (
            np.asarray(
                data["stable_support_reset_segment_anchor_frames"], dtype=np.int64
            ).reshape(-1)
            if "stable_support_reset_segment_anchor_frames" in data.files
            else default_segment_starts.copy()
        )
        segment_ids = (
            np.asarray(data["stable_support_reset_segment_ids"], dtype=np.int64).reshape(-1)
            if "stable_support_reset_segment_ids" in data.files
            else np.searchsorted(
                segment_starts,
                np.arange(self.joint_pos.shape[0], dtype=np.int64),
                side="right",
            )
            - 1
        )
        if not (
            segment_starts.shape
            == segment_ends.shape
            == segment_anchor_frames.shape
        ):
            raise ValueError("stable-support reset segment arrays must have matching shapes")
        if segment_starts.size == 0:
            raise ValueError("stable-support reset segments must not be empty")
        if has_saved_reset_segments:
            if (
                stable_support_anchor_frames.size < 2
                or stable_support_anchor_frames[0] != 0
                or stable_support_anchor_frames[-1] != self.joint_pos.shape[0] - 1
            ):
                raise ValueError(
                    "saved reset segments require stable anchors at the first and "
                    "terminal motion frames"
                )
            expected_segment_starts = stable_support_anchor_frames[:-1]
            expected_segment_ends = stable_support_anchor_frames[1:] - 1
            expected_segment_ends[-1] = self.joint_pos.shape[0] - 1
            if not (
                np.array_equal(segment_starts, expected_segment_starts)
                and np.array_equal(segment_ends, expected_segment_ends)
            ):
                raise ValueError(
                    "saved reset segment bounds must be derived from the shared "
                    "stable-support anchors"
                )
        if has_saved_reset_segments and (
            segment_starts[0] != 0
            or segment_ends[-1] != self.joint_pos.shape[0] - 1
            or np.any(segment_starts != segment_anchor_frames)
            or np.any(segment_starts > segment_ends)
            or np.any(segment_ends[:-1] + 1 != segment_starts[1:])
        ):
            raise ValueError(
                "stable-support reset segments must partition the complete motion "
                "from frame 0 through the terminal frame"
            )
        if has_saved_reset_segments and (
            segment_ids.shape != (self.joint_pos.shape[0],)
            or np.any(
            segment_ids != np.searchsorted(segment_starts, np.arange(self.joint_pos.shape[0]), side="right") - 1
            )
        ):
            raise ValueError("stable-support reset segment IDs do not match segment bounds")
        self.stable_reset_segment_starts = torch.tensor(
            segment_starts, dtype=torch.long, device=device
        )
        self.stable_reset_segment_ends = torch.tensor(
            segment_ends, dtype=torch.long, device=device
        )
        self.stable_reset_segment_anchor_frames = torch.tensor(
            segment_anchor_frames, dtype=torch.long, device=device
        )
        self.stable_reset_segment_ids = torch.tensor(
            segment_ids, dtype=torch.long, device=device
        )
        self.stable_reset_segment_bounds = tuple(
            zip(segment_starts.tolist(), segment_ends.tolist(), strict=True)
        )
        if stable_support_anchor_frames.size:
            motion_target_frame, motion_target_successor = stable_motion_target_frames(
                climb_direction, stable_support_anchor_frames
            )
        else:
            motion_target_frame = np.full(climb_direction.shape, -1, dtype=np.int64)
            motion_target_successor = np.arange(climb_direction.size, dtype=np.int64)
        self.stable_motion_anchor_mask = torch.zeros(
            self.joint_pos.shape[0], dtype=torch.bool, device=device
        )
        self.stable_motion_anchor_predecessor = torch.arange(
            self.joint_pos.shape[0], dtype=torch.long, device=device
        )
        if self.stable_motion_anchor_frames.numel() > 0:
            self.stable_motion_anchor_mask[self.stable_motion_anchor_frames] = True
            if self.stable_motion_anchor_frames.numel() > 1:
                self.stable_motion_anchor_predecessor[
                    self.stable_motion_anchor_frames[1:]
                ] = self.stable_motion_anchor_frames[:-1]
        self.initial_stable_motion_target_frame = torch.tensor(
            motion_target_frame, dtype=torch.long, device=device
        )
        self.stable_motion_target_successor = torch.tensor(
            motion_target_successor, dtype=torch.long, device=device
        )
        self.ladder_params = (
            np.asarray(data["contact_annotation_ladder_params"], dtype=np.float32)
            if "contact_annotation_ladder_params" in data.files
            else None
        )
        self.ladder_origin = (
            np.asarray(data["contact_annotation_ladder_origin"], dtype=np.float32)
            if "contact_annotation_ladder_origin" in data.files
            else None
        )
        self.ladder_num_rungs = (
            int(np.asarray(data["contact_annotation_ladder_num_rungs"]).reshape(-1)[0])
            if "contact_annotation_ladder_num_rungs" in data.files
            else None
        )
        self.geometry_grid_inclination_deltas = (
            np.asarray(data["reset_ladder_geometry_grid_inclination_deltas"], dtype=np.float32)
            if "reset_ladder_geometry_grid_inclination_deltas" in data.files
            else None
        )
        self.geometry_grid_spacing_deltas = (
            np.asarray(data["reset_ladder_geometry_grid_spacing_deltas"], dtype=np.float32)
            if "reset_ladder_geometry_grid_spacing_deltas" in data.files
            else None
        )
        self.geometry_cell_reset_valid_masks = (
            np.asarray(data["reset_ladder_geometry_cell_valid_mask"], dtype=np.bool_)
            if "reset_ladder_geometry_cell_valid_mask" in data.files
            else None
        )
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

        reset_valid_mask, self.has_reset_valid_mask = load_reset_valid_mask(
            data, self.time_step_total
        )
        self.reset_valid_mask = torch.tensor(reset_valid_mask, dtype=torch.bool, device=device)
        self.reset_valid_frame_indexes = torch.nonzero(
            self.reset_valid_mask, as_tuple=False
        ).squeeze(-1)
        self.using_geometry_conditioned_reset_masks = False

        if self.contact_indicator.shape != self.contact_rung_id.shape:
            raise ValueError("contact_indicator and contact_rung_id must have identical shapes.")
        if self.contact_indicator.shape[0] != self.time_step_total:
            raise ValueError("contact annotations must have one row per motion frame.")
        if self.contact_indicator.shape[1] != len(LIMB_NAMES):
            raise ValueError(f"expected {len(LIMB_NAMES)} contact limbs, got {self.contact_indicator.shape[1]}")

    def use_all_frames_for_reset(self) -> None:
        """Allow every reference frame to be sampled as a reset frame."""

        self.reset_valid_mask = torch.ones_like(self.reset_valid_mask)
        self.reset_valid_frame_indexes = torch.arange(
            self.time_step_total, dtype=torch.long, device=self.joint_pos.device
        )
        self.using_geometry_conditioned_reset_masks = False

    def use_geometry_conditioned_reset_masks(
        self,
        inclination_delta_range: tuple[float, float],
        spacing_delta_range: tuple[float, float],
    ) -> None:
        """Load per-cell reset masks covering the requested ladder randomization envelope."""

        if (
            self.geometry_grid_inclination_deltas is None
            or self.geometry_grid_spacing_deltas is None
            or self.geometry_cell_reset_valid_masks is None
        ):
            raise ValueError(
                "ladder geometry randomization requires geometry-grid axes and "
                "reset_ladder_geometry_cell_valid_mask in the motion file"
            )
        inclination_grid = self.geometry_grid_inclination_deltas.reshape(-1)
        spacing_grid = self.geometry_grid_spacing_deltas.reshape(-1)
        if inclination_grid.size < 2 or spacing_grid.size < 2:
            raise ValueError("ladder geometry reset grids must each contain at least two values")
        if not np.all(np.diff(inclination_grid) > 0.0) or not np.all(np.diff(spacing_grid) > 0.0):
            raise ValueError("ladder geometry reset grid axes must be strictly increasing")
        expected_shape = (inclination_grid.size - 1, spacing_grid.size - 1, self.time_step_total)
        if self.geometry_cell_reset_valid_masks.shape != expected_shape:
            raise ValueError(
                f"reset_ladder_geometry_cell_valid_mask must have shape {expected_shape}, "
                f"got {self.geometry_cell_reset_valid_masks.shape}"
            )
        requested_inclination = np.asarray(inclination_delta_range, dtype=np.float32)
        requested_spacing = np.asarray(spacing_delta_range, dtype=np.float32)
        tolerance = 1.0e-6
        if (
            requested_inclination[0] < inclination_grid[0] - tolerance
            or requested_inclination[1] > inclination_grid[-1] + tolerance
            or requested_spacing[0] < spacing_grid[0] - tolerance
            or requested_spacing[1] > spacing_grid[-1] + tolerance
        ):
            raise ValueError(
                "motion reset geometry grid does not cover requested ladder randomization: "
                f"requested inclination={requested_inclination.tolist()}, spacing={requested_spacing.tolist()}, "
                f"available inclination={[float(inclination_grid[0]), float(inclination_grid[-1])]}, "
                f"spacing={[float(spacing_grid[0]), float(spacing_grid[-1])]}"
            )
        cell_masks = torch.tensor(
            self.geometry_cell_reset_valid_masks, dtype=torch.bool, device=self.joint_pos.device
        )
        if torch.any(~torch.any(cell_masks, dim=-1)):
            raise ValueError("at least one ladder geometry grid cell excludes every reference frame")
        self._geometry_grid_inclination_deltas = torch.tensor(
            inclination_grid, dtype=torch.float32, device=self.joint_pos.device
        )
        self._geometry_grid_spacing_deltas = torch.tensor(
            spacing_grid, dtype=torch.float32, device=self.joint_pos.device
        )
        self._geometry_cell_reset_valid_masks = cell_masks
        union_mask = torch.any(cell_masks, dim=(0, 1))
        self.reset_valid_mask = union_mask
        self.reset_valid_frame_indexes = torch.nonzero(union_mask, as_tuple=False).squeeze(-1)
        self.using_geometry_conditioned_reset_masks = True

    def geometry_reset_valid_masks(
        self, inclination_deltas: torch.Tensor, spacing_deltas: torch.Tensor
    ) -> torch.Tensor:
        if not self.using_geometry_conditioned_reset_masks:
            return self.reset_valid_mask.unsqueeze(0).expand(len(inclination_deltas), -1)
        return select_geometry_cell_reset_masks(
            self._geometry_cell_reset_valid_masks,
            self._geometry_grid_inclination_deltas,
            self._geometry_grid_spacing_deltas,
            inclination_deltas,
            spacing_deltas,
        )

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class LadderMotionCommand(CommandTerm):
    cfg: LadderMotionCommandCfg

    def __init__(self, cfg: LadderMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.motion = LadderMotionLoader(
            self.cfg.motion_file, self.robot.joint_names, self.cfg.body_names, device=self.device
        )
        control_fps = 1.0 / (env.cfg.decimation * env.cfg.sim.dt)
        self._control_fps = control_fps
        if not math.isclose(self.motion.fps, control_fps, rel_tol=0.0, abs_tol=1.0e-4):
            raise ValueError(
                f"motion fps ({self.motion.fps:g}) must match environment control fps ({control_fps:g}); "
                "resample the reference motion before training"
            )
        self.ladder_geometry = cfg.ladder_geometry
        self._validate_ladder_metadata()
        inclination_range = tuple(self.cfg.ladder_inclination_delta_range)
        spacing_range = tuple(self.cfg.ladder_spacing_delta_range)
        if inclination_range[0] > inclination_range[1]:
            raise ValueError(f"invalid ladder_inclination_delta_range: {inclination_range}")
        if spacing_range[0] > spacing_range[1]:
            raise ValueError(f"invalid ladder_spacing_delta_range: {spacing_range}")
        if not 0.0 <= self.cfg.nominal_ladder_geometry_probability <= 1.0:
            raise ValueError(
                "nominal_ladder_geometry_probability must be in [0, 1], got "
                f"{self.cfg.nominal_ladder_geometry_probability}"
            )
        if self.ladder_geometry.rung_spacing + spacing_range[0] <= 0.0:
            raise ValueError("ladder spacing randomization must remain positive")
        if not (
            0.0 < self.ladder_geometry.phi + inclination_range[0]
            and self.ladder_geometry.phi + inclination_range[1] < math.pi / 2.0
        ):
            raise ValueError("ladder inclination randomization must remain between 0 and 90 degrees")
        self._geometry_randomization_enabled = (
            inclination_range != (0.0, 0.0) or spacing_range != (0.0, 0.0)
        )
        if not self.cfg.use_reset_valid_mask:
            self.motion.use_all_frames_for_reset()
        elif (
            self._geometry_randomization_enabled
            and self.cfg.use_geometry_conditioned_reset_masks
        ):
            self.motion.use_geometry_conditioned_reset_masks(inclination_range, spacing_range)
        if self.cfg.fixed_reference_frame is not None and not (
            0 <= self.cfg.fixed_reference_frame < self.motion.time_step_total
        ):
            raise ValueError(
                f"fixed_reference_frame must be in [0, {self.motion.time_step_total}), "
                f"got {self.cfg.fixed_reference_frame}"
            )
        if not 0.0 <= self.cfg.fixed_start_probability <= 1.0:
            raise ValueError(
                f"fixed_start_probability must be in [0, 1], got {self.cfg.fixed_start_probability}"
            )
        if not 0.0 <= self.cfg.uniform_start_probability <= 1.0:
            raise ValueError(
                f"uniform_start_probability must be in [0, 1], got {self.cfg.uniform_start_probability}"
            )
        if not 0.0 <= self.cfg.secondary_fixed_start_probability <= 1.0:
            raise ValueError(
                "secondary_fixed_start_probability must be in [0, 1], got "
                f"{self.cfg.secondary_fixed_start_probability}"
            )
        reset_mixture_probability = (
            self.cfg.fixed_start_probability
            + self.cfg.secondary_fixed_start_probability
            + self.cfg.uniform_start_probability
        )
        if reset_mixture_probability > 1.0:
            raise ValueError(
                "fixed_start_probability + secondary_fixed_start_probability + "
                f"uniform_start_probability must be at most 1, got {reset_mixture_probability}"
            )
        if not 0 <= self.cfg.fixed_start_frame < self.motion.time_step_total:
            raise ValueError(
                f"fixed_start_frame must be in [0, {self.motion.time_step_total}), "
                f"got {self.cfg.fixed_start_frame}"
            )
        if self.cfg.fixed_start_probability > 0.0 and not self.motion.reset_valid_mask[
            self.cfg.fixed_start_frame
        ].item():
            raise ValueError(
                f"fixed_start_frame {self.cfg.fixed_start_frame} is excluded by reset_valid_mask"
            )
        secondary_frame = self.cfg.secondary_fixed_start_frame
        if self.cfg.secondary_fixed_start_probability > 0.0:
            if secondary_frame is None or not 0 <= secondary_frame < self.motion.time_step_total:
                raise ValueError(
                    "secondary_fixed_start_frame must be in "
                    f"[0, {self.motion.time_step_total}) when enabled, got {secondary_frame}"
                )
            if not self.motion.reset_valid_mask[secondary_frame].item():
                raise ValueError(
                    f"secondary_fixed_start_frame {secondary_frame} is excluded by reset_valid_mask"
                )
            if (
                self.cfg.secondary_fixed_start_requires_nominal_ladder
                and self._geometry_randomization_enabled
                and self.cfg.secondary_fixed_start_probability
                > self.cfg.nominal_ladder_geometry_probability
            ):
                raise ValueError(
                    "secondary_fixed_start_probability cannot exceed "
                    "nominal_ladder_geometry_probability when the secondary start "
                    "requires nominal ladder geometry"
                )
        if (
            self.cfg.fixed_start_probability > 0.0
            and self.motion.using_geometry_conditioned_reset_masks
            and not torch.all(
                self.motion._geometry_cell_reset_valid_masks[..., self.cfg.fixed_start_frame]
            )
        ):
            raise ValueError(
                f"fixed_start_frame {self.cfg.fixed_start_frame} is not valid in every ladder geometry cell"
            )
        nonfixed_mask = self.motion.reset_valid_frame_indexes != self.cfg.fixed_start_frame
        if secondary_frame is not None:
            nonfixed_mask &= self.motion.reset_valid_frame_indexes != secondary_frame
        self.nonfixed_reset_valid_frame_indexes = self.motion.reset_valid_frame_indexes[nonfixed_mask]
        nonfixed_probability = (
            1.0
            - self.cfg.fixed_start_probability
            - self.cfg.secondary_fixed_start_probability
        )
        if nonfixed_probability > 0.0 and len(self.nonfixed_reset_valid_frame_indexes) == 0:
            raise ValueError(
                "fixed/uniform/adaptive reset mixing requires at least one non-fixed valid reset frame"
            )
        self._validate_stop_command_cfg()
        self._validate_motion_semantic_cfg()
        self._validate_stable_anchor_gating_cfg()

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_start_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._fixed_start_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._secondary_fixed_start_episode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._fixed_start_pose_randomized_episode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._uniform_start_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stop_state = torch.full(
            (self.num_envs,), STOP_STATE_RUN, dtype=torch.long, device=self.device
        )
        self._stop_target_frame = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._stop_trigger_frame = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._stop_hold_steps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._stop_training_direct_hold_episode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stop_training_seek_episode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._motion_target_frame = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._motion_progress_direction = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._motion_progress_confirmation_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._stable_anchor_holding = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stable_anchor_hold_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._stable_anchor_confirmation_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._stable_anchor_timeout_failed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stable_anchor_completion_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stable_anchor_completion_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._current_reset_valid_fraction = torch.ones(self.num_envs, device=self.device)
        self._tracking_path_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tracking_start_reference_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._tracking_start_robot_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._tracking_previous_reference_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._tracking_previous_robot_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._tracking_episode_steps = torch.zeros(self.num_envs, device=self.device)
        self._tracking_reference_motion_steps = torch.zeros(self.num_envs, device=self.device)
        self._tracking_stationary_motion_steps = torch.zeros(self.num_envs, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor([self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device)
        self.kernel = self.kernel / self.kernel.sum()

        if not self.cfg.use_reset_valid_mask:
            reset_mask_source = "all frames (reset mask disabled)"
        elif self.motion.using_geometry_conditioned_reset_masks:
            reset_mask_source = "embedded geometry-conditioned reset masks"
        else:
            reset_mask_source = "embedded reset_valid_mask" if self.motion.has_reset_valid_mask else "all frames"
        print(
            f"[INFO] Ladder reset frames: {len(self.motion.reset_valid_frame_indexes)}/"
            f"{self.motion.time_step_total} valid ({reset_mask_source})."
        )
        if (
            self._geometry_randomization_enabled
            and self.cfg.use_reset_valid_mask
            and not self.cfg.use_geometry_conditioned_reset_masks
        ):
            print(
                "[WARNING] Randomized ladder geometry is using the nominal-ladder reset mask; "
                "some reset states may initially penetrate the sampled ladder."
            )

        self.limb_names = tuple(cfg.limb_names)
        self.limb_body_names = [cfg.limb_body_names[name] for name in self.limb_names]
        self.limb_body_indexes = torch.tensor(
            self.robot.find_bodies(self.limb_body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        if self.cfg.contact_sensor_name not in env.scene.sensors:
            raise ValueError(
                f"contact sensor {self.cfg.contact_sensor_name!r} is not present in the scene; "
                "required for physical contact metrics"
            )
        self._contact_sensor = env.scene.sensors[self.cfg.contact_sensor_name]
        contact_body_indexes, contact_body_names = self._contact_sensor.find_bodies(
            self.limb_body_names, preserve_order=True
        )
        if contact_body_names != self.limb_body_names:
            raise ValueError(
                "contact sensor must expose one body per configured limb in order; "
                f"requested={self.limb_body_names}, resolved={contact_body_names}"
            )
        self._contact_limb_body_indexes = torch.tensor(
            contact_body_indexes, dtype=torch.long, device=self.device
        )
        self._stable_anchor_limb_contact_sensors = []
        self._stable_anchor_ground_contact_sensors = []
        self._stable_anchor_rail_contact_sensors = []
        if self.cfg.stable_anchor_gating_enabled:
            sensor_names = tuple(self.cfg.stable_anchor_limb_contact_sensor_names)
            if len(sensor_names) != len(self.limb_names):
                raise ValueError(
                    "stable_anchor_limb_contact_sensor_names must contain one sensor "
                    "per configured limb"
                )
            missing_sensors = [name for name in sensor_names if name not in env.scene.sensors]
            if missing_sensors:
                raise ValueError(
                    f"stable-anchor gating is missing limb contact sensors: {missing_sensors}"
                )
            self._stable_anchor_limb_contact_sensors = [
                env.scene.sensors[name] for name in sensor_names
            ]
            ground_sensor_names = tuple(
                self.cfg.stable_anchor_ground_contact_sensor_names
            )
            if ground_sensor_names and len(ground_sensor_names) != len(
                self.limb_names
            ):
                raise ValueError(
                    "stable_anchor_ground_contact_sensor_names must be empty or "
                    "contain one sensor per configured limb"
                )
            missing_ground_sensors = [
                name for name in ground_sensor_names if name not in env.scene.sensors
            ]
            if missing_ground_sensors:
                raise ValueError(
                    "stable-anchor gating is missing ground contact sensors: "
                    f"{missing_ground_sensors}"
                )
            self._stable_anchor_ground_contact_sensors = [
                env.scene.sensors[name] for name in ground_sensor_names
            ]
            rail_sensor_names = tuple(self.cfg.stable_anchor_rail_contact_sensor_names)
            missing_rail_sensors = [
                name for name in rail_sensor_names if name not in env.scene.sensors
            ]
            if missing_rail_sensors:
                raise ValueError(
                    f"stable-anchor gating is missing rail contact sensors: {missing_rail_sensors}"
                )
            self._stable_anchor_rail_contact_sensors = [
                env.scene.sensors[name] for name in rail_sensor_names
            ]
        allowed_contact_body_names = set(self.limb_body_names)
        for sensor in self._stable_anchor_limb_contact_sensors:
            allowed_contact_body_names.update(sensor.body_names)
        allowed_contact_indexes, _ = self._contact_sensor.find_bodies(
            sorted(allowed_contact_body_names), preserve_order=False
        )
        limb_contact_indexes = set(allowed_contact_indexes)
        self._stable_anchor_danger_body_indexes = torch.tensor(
            [
                index
                for index in range(len(self._contact_sensor.body_names))
                if index not in limb_contact_indexes
            ],
            dtype=torch.long,
            device=self.device,
        )
        if self.cfg.limb_target_distance_thresholds is None:
            target_distance_thresholds = [0.10] * len(self.limb_names)
        else:
            missing_thresholds = [
                name for name in self.limb_names if name not in self.cfg.limb_target_distance_thresholds
            ]
            if missing_thresholds:
                raise ValueError(
                    f"limb_target_distance_thresholds is missing limbs: {missing_thresholds}"
                )
            target_distance_thresholds = [
                self.cfg.limb_target_distance_thresholds[name] for name in self.limb_names
            ]
        self._limb_target_distance_thresholds = torch.tensor(
            target_distance_thresholds, dtype=torch.float32, device=self.device
        )
        if torch.any(self._limb_target_distance_thresholds <= 0.0):
            raise ValueError("limb target distance thresholds must be positive")
        if self.cfg.contact_force_threshold < 0.0:
            raise ValueError(
                f"contact_force_threshold must be nonnegative, got {self.cfg.contact_force_threshold}"
            )
        if cfg.limb_target_offsets is None:
            self._limb_offsets = limb_lateral_offsets_tensor(
                self.ladder_geometry, self.limb_names, device=self.device
            )
        else:
            missing_offsets = [name for name in self.limb_names if name not in cfg.limb_target_offsets]
            if missing_offsets:
                raise ValueError(f"limb_target_offsets is missing limbs: {missing_offsets}")
            self._limb_offsets = torch.tensor(
                [cfg.limb_target_offsets[name] for name in self.limb_names],
                dtype=torch.float32,
                device=self.device,
            )

        self.ladder_phi = torch.full(
            (self.num_envs,), self.ladder_geometry.phi, dtype=torch.float32, device=self.device
        )
        self.ladder_spacing = torch.full(
            (self.num_envs,), self.ladder_geometry.rung_spacing, dtype=torch.float32, device=self.device
        )
        self._nominal_ladder_geometry_episode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._ladder_origin_e = torch.tensor(
            self.ladder_geometry.origin, dtype=torch.float32, device=self.device
        )
        self._ladder_delta_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._ladder_delta_quat[:, 0] = 1.0
        self._ladder_asset: RigidObjectCollection | None = None
        if self.cfg.ladder_asset_name is not None:
            ladder_asset = self._env.scene[self.cfg.ladder_asset_name]
            if not isinstance(ladder_asset, RigidObjectCollection):
                raise TypeError(
                    f"scene asset {self.cfg.ladder_asset_name!r} must be a RigidObjectCollection"
                )
            self._ladder_asset = ladder_asset
            object_names = tuple(ladder_asset.object_names)
            configured_names = set(self.cfg.ladder_fixed_box_names) | set(
                self.cfg.ladder_rung_box_names
            ) | set(
                self.cfg.ladder_inclined_box_names
            )
            missing_names = sorted(configured_names - set(object_names))
            if missing_names:
                raise ValueError(f"ladder collection is missing configured boxes: {missing_names}")
            default_state = ladder_asset.data.default_object_state[0]
            self._ladder_box_positions_e = default_state[:, :3].clone()
            self._ladder_box_quats = default_state[:, 3:7].clone()
            self._ladder_fixed_box_mask = torch.tensor(
                [name in self.cfg.ladder_fixed_box_names for name in object_names],
                dtype=torch.bool,
                device=self.device,
            )
            self._ladder_inclined_box_mask = torch.tensor(
                [name in self.cfg.ladder_inclined_box_names for name in object_names],
                dtype=torch.bool,
                device=self.device,
            )
            object_name_to_index = {name: index for index, name in enumerate(object_names)}
            self._ladder_rung_box_indexes = torch.tensor(
                [object_name_to_index[name] for name in self.cfg.ladder_rung_box_names],
                dtype=torch.long,
                device=self.device,
            )
            rung_indexes = torch.arange(
                len(self.cfg.ladder_rung_box_names), dtype=torch.float32, device=self.device
            ).unsqueeze(-1)
            source_direction = torch.tensor(
                self.ladder_geometry.climb_direction, dtype=torch.float32, device=self.device
            )
            nominal_rung_tops = (
                self._ladder_origin_e.unsqueeze(0)
                + rung_indexes * self.ladder_geometry.rung_spacing * source_direction.unsqueeze(0)
            )
            self._ladder_rung_center_offsets_e = (
                self._ladder_box_positions_e[self._ladder_rung_box_indexes] - nominal_rung_tops
            )
            relative = self._ladder_box_positions_e - self._ladder_origin_e
            source_cos = math.cos(self.ladder_geometry.phi)
            source_sin = math.sin(self.ladder_geometry.phi)
            self._ladder_box_axial = relative[:, 0] * source_cos + relative[:, 2] * source_sin
            self._ladder_box_normal = -relative[:, 0] * source_sin + relative[:, 2] * source_cos
        elif self._geometry_randomization_enabled:
            raise ValueError("ladder geometry randomization requires ladder_asset_name")

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_anchor_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_anchor_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_joint_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_joint_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_anchor_path_length"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_anchor_path_length"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_anchor_displacement"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_anchor_displacement"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["anchor_progress_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["progress_evaluable"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_completion_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_moving_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stationary_during_reference_motion_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["adaptive_sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["adaptive_sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ladder_inclination_deg"] = torch.full(
            (self.num_envs,), math.degrees(self.ladder_geometry.phi), device=self.device
        )
        self.metrics["ladder_rung_spacing"] = torch.full(
            (self.num_envs,), self.ladder_geometry.rung_spacing, device=self.device
        )
        self.metrics["nominal_ladder_geometry_episode"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["reset_valid_fraction"] = torch.full(
            (self.num_envs,),
            len(self.motion.reset_valid_frame_indexes) / self.motion.time_step_total,
            device=self.device,
        )
        self.metrics["mean_required_contact_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["required_contact_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["required_physical_contact_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["required_target_contact_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["swing_physical_contact_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["motion_semantic_target_frame_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["motion_semantic_joint_rms_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["motion_semantic_orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["motion_semantic_target_contact_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["motion_semantic_completed"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["next_stable_target_mean_distance"] = torch.zeros(
            self.num_envs, device=self.device
        )
        for name in (
            "stable_anchor_holding",
            "stable_anchor_hold_time_s",
            "stable_anchor_strict_complete",
            "stable_anchor_relaxed_complete",
            "stable_anchor_safe",
            "stable_anchor_contact_match_fraction",
            "stable_anchor_target_contact_fraction",
            "stable_anchor_completion_event",
            "stable_anchor_completion_count",
            "stable_anchor_timeout_failed",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        for limb_name in self.limb_names:
            self.metrics[f"{limb_name}_physical_contact"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics[f"{limb_name}_required_physical_contact"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics[f"{limb_name}_required_target_contact"] = torch.zeros(
                self.num_envs, device=self.device
            )
        for hand_name in ("left_hand", "right_hand"):
            self.metrics[f"{hand_name}_required_contact_error"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics[f"{hand_name}_contact_required"] = torch.zeros(
                self.num_envs, device=self.device
            )
        self.metrics["fixed_start_episode"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["secondary_fixed_start_episode"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["fixed_start_pose_randomized_episode"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["fixed_start_pose_nominal_episode"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["uniform_start_episode"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["adaptive_start_episode"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stop_command_requested"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stop_command_seeking"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stop_command_holding"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stop_target_frame_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stop_time_to_target_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stop_training_direct_hold_episode"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["stop_training_seek_episode"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self._contact_force_metric_specs: dict[str, tuple[object, str]] = {}
        for side in ("left", "right"):
            step_sensor_name = f"{side}_hand_step_contacts"
            if step_sensor_name in env.scene.sensors:
                step_sensor = env.scene.sensors[step_sensor_name]
                for surface in ("top", "front", "underside"):
                    metric_name = f"{side}_hand_{surface}_force_max"
                    self._contact_force_metric_specs[metric_name] = (step_sensor, surface)
                    self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
                self.metrics[f"{side}_hand_wrong_rung_force_max"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                target_metric_name = f"{side}_hand_target_rung_upward_force_max"
                self.metrics[target_metric_name] = torch.zeros(self.num_envs, device=self.device)
            for surface_name, sensor_name in (
                ("side_board", f"{side}_hand_side_board_contacts"),
                ("ground", f"{side}_hand_ground_contacts"),
            ):
                if sensor_name in env.scene.sensors:
                    metric_name = f"{side}_hand_{surface_name}_force_max"
                    self._contact_force_metric_specs[metric_name] = (
                        env.scene.sensors[sensor_name],
                        "all",
                    )
                    self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
            foot_sensor_name = f"{side}_foot_step_contacts"
            if foot_sensor_name in env.scene.sensors:
                foot_sensor = env.scene.sensors[foot_sensor_name]
                metric_name = f"{side}_foot_step_force_max"
                self._contact_force_metric_specs[metric_name] = (foot_sensor, "all")
                self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
                target_metric_name = f"{side}_foot_target_rung_force_max"
                self.metrics[target_metric_name] = torch.zeros(self.num_envs, device=self.device)
        self._contact_force_soft_limits = dict(self.cfg.contact_force_soft_limits)
        self._contact_force_hard_limits = dict(self.cfg.contact_force_hard_limits)
        if self._contact_force_soft_limits.keys() != self._contact_force_hard_limits.keys():
            raise ValueError("contact force soft/hard limit keys must match")
        for name, force_soft in self._contact_force_soft_limits.items():
            force_hard = self._contact_force_hard_limits[name]
            if force_soft < 0.0 or force_hard <= force_soft:
                raise ValueError(
                    f"contact force limit {name!r} must satisfy 0 <= soft < hard, "
                    f"got soft={force_soft}, hard={force_hard}"
                )
        self._force_limit_sample_count: dict[str, torch.Tensor] = {}
        self._force_limit_over_soft_count: dict[str, torch.Tensor] = {}
        self._force_limit_previous_hard: dict[str, torch.Tensor] = {}
        self._force_histogram_bin_count = 64
        self._force_histogram_max = {
            "left_hand": 2000.0,
            "right_hand": 2000.0,
            "left_foot": 5000.0,
            "right_foot": 5000.0,
        }
        self._force_histograms: dict[str, torch.Tensor] = {}
        for limb_name in ("left_hand", "right_hand", "left_foot", "right_foot"):
            self._force_limit_sample_count[limb_name] = torch.zeros(
                self.num_envs, device=self.device
            )
            self._force_limit_over_soft_count[limb_name] = torch.zeros(
                self.num_envs, device=self.device
            )
            self._force_limit_previous_hard[limb_name] = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.metrics[f"{limb_name}_force_over_soft_fraction"] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics[f"{limb_name}_hard_collision_count"] = torch.zeros(
                self.num_envs, device=self.device
            )
            if self._contact_force_soft_limits:
                self._force_histograms[limb_name] = torch.zeros(
                    self.num_envs,
                    self._force_histogram_bin_count,
                    device=self.device,
                )
                self.metrics[f"{limb_name}_force_p95"] = torch.zeros(
                    self.num_envs, device=self.device
                )

    def _validate_stop_command_cfg(self) -> None:
        probabilities = (
            self.cfg.stop_hold_reset_probability,
            self.cfg.stop_seek_episode_probability,
        )
        if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
            raise ValueError("stop-command training probabilities must lie in [0, 1]")
        if sum(probabilities) > 1.0:
            raise ValueError(
                "stop_hold_reset_probability + stop_seek_episode_probability must be at most 1"
            )
        for name, interval in (
            ("stop_hold_time_range_s", self.cfg.stop_hold_time_range_s),
        ):
            if interval[0] < 0.0 or interval[0] > interval[1]:
                raise ValueError(f"invalid {name}: {interval}")
        if self.cfg.stop_command_training_enabled:
            if self.motion.zero_command_direct_hold_frames.numel() == 0:
                raise ValueError(
                    "stop-command training requires "
                    "zero_command_direct_hold_frames or "
                    "zero_command_all_limb_contact_segment_selected_frames in the motion"
                )
            if not torch.any(self.motion.zero_command_valid_mask):
                raise ValueError(
                    "stop-command training requires a non-empty zero_command_valid_mask"
                )
        elif any(probability > 0.0 for probability in probabilities):
            raise ValueError(
                "stop-command training probabilities require stop_command_training_enabled=True"
            )

    def _validate_motion_semantic_cfg(self) -> None:
        if not self.cfg.motion_semantic_progress_enabled:
            return
        if self.motion.stable_motion_anchor_frames.numel() == 0:
            raise ValueError(
                "motion-semantic supervision requires stable_support_anchor_frames "
                "or legacy zero-command stable frames"
            )
        positive_values = {
            "motion_progress_joint_rms_threshold": self.cfg.motion_progress_joint_rms_threshold,
            "motion_progress_orientation_threshold": self.cfg.motion_progress_orientation_threshold,
            "motion_progress_anchor_linear_speed_threshold": (
                self.cfg.motion_progress_anchor_linear_speed_threshold
            ),
            "motion_progress_anchor_angular_speed_threshold": (
                self.cfg.motion_progress_anchor_angular_speed_threshold
            ),
            "motion_progress_joint_speed_rms_threshold": (
                self.cfg.motion_progress_joint_speed_rms_threshold
            ),
        }
        invalid = {name: value for name, value in positive_values.items() if value <= 0.0}
        if invalid:
            raise ValueError(f"motion-semantic progress thresholds must be positive: {invalid}")
        if self.cfg.motion_progress_confirmation_steps <= 0:
            raise ValueError("motion_progress_confirmation_steps must be positive")

    def _validate_stable_anchor_gating_cfg(self) -> None:
        if not self.cfg.stable_anchor_gating_enabled:
            return
        if self.motion.stable_motion_anchor_frames.numel() == 0:
            raise ValueError("stable-anchor gating requires stable motion anchors")
        anchor_support_ids = self.motion.stable_support_rung_id[
            self.motion.stable_motion_anchor_frames
        ]
        if torch.any(anchor_support_ids > self.ladder_geometry.num_rungs):
            raise ValueError(
                "stable-support IDs exceed the configured ladder rung count"
            )
        if torch.any(anchor_support_ids == 0) and len(
            self.cfg.stable_anchor_ground_contact_sensor_names
        ) != len(self.cfg.limb_names):
            raise ValueError(
                "ground-aware stable anchors require one ground contact sensor per limb"
            )
        if self.cfg.stop_command_training_enabled:
            raise ValueError(
                "stable-anchor gating and stop-command training cannot be enabled together"
            )
        if not 3 <= self.cfg.stable_anchor_confirmation_steps <= 5:
            raise ValueError(
                "stable_anchor_confirmation_steps must be between 3 and 5"
            )
        positive_values = {
            "stable_anchor_relaxed_after_s": self.cfg.stable_anchor_relaxed_after_s,
            "stable_anchor_failure_after_s": self.cfg.stable_anchor_failure_after_s,
            "stable_anchor_root_height_error_threshold": (
                self.cfg.stable_anchor_root_height_error_threshold
            ),
            "stable_anchor_root_orientation_threshold": (
                self.cfg.stable_anchor_root_orientation_threshold
            ),
            "stable_anchor_root_linear_speed_threshold": (
                self.cfg.stable_anchor_root_linear_speed_threshold
            ),
            "stable_anchor_root_angular_speed_threshold": (
                self.cfg.stable_anchor_root_angular_speed_threshold
            ),
            "stable_anchor_danger_contact_force_threshold": (
                self.cfg.stable_anchor_danger_contact_force_threshold
            ),
        }
        invalid = {name: value for name, value in positive_values.items() if value <= 0.0}
        if invalid:
            raise ValueError(f"stable-anchor gating values must be positive: {invalid}")
        if self.cfg.stable_anchor_failure_after_s <= self.cfg.stable_anchor_relaxed_after_s:
            raise ValueError(
                "stable_anchor_failure_after_s must exceed stable_anchor_relaxed_after_s"
            )

    def _validate_ladder_metadata(self) -> None:
        if self.motion.ladder_params is not None:
            expected = np.asarray(
                [
                    self.ladder_geometry.phi,
                    self.ladder_geometry.rung_spacing,
                    self.ladder_geometry.rung_length,
                    self.ladder_geometry.rung_width,
                ],
                dtype=np.float32,
            )
            if not np.allclose(self.motion.ladder_params, expected, atol=1.0e-5):
                raise ValueError(
                    f"motion ladder parameters {self.motion.ladder_params.tolist()} do not match "
                    f"environment ladder parameters {expected.tolist()}"
                )
        if self.motion.ladder_origin is not None:
            expected_origin = np.asarray(self.ladder_geometry.origin, dtype=np.float32)
            if not np.allclose(self.motion.ladder_origin, expected_origin, atol=1.0e-5):
                raise ValueError(
                    f"motion ladder origin {self.motion.ladder_origin.tolist()} does not match "
                    f"environment ladder origin {expected_origin.tolist()}"
                )
        if (
            self.motion.ladder_num_rungs is not None
            and self.motion.ladder_num_rungs != self.ladder_geometry.num_rungs
        ):
            raise ValueError(
                f"motion has {self.motion.ladder_num_rungs} rungs but environment has "
                f"{self.ladder_geometry.num_rungs}"
            )

    def _sample_ladder_geometry(self, env_ids: torch.Tensor) -> None:
        inclination_range = self.cfg.ladder_inclination_delta_range
        spacing_range = self.cfg.ladder_spacing_delta_range
        inclination_deltas = sample_uniform(
            inclination_range[0], inclination_range[1], (len(env_ids),), self.device
        )
        spacing_deltas = sample_uniform(
            spacing_range[0], spacing_range[1], (len(env_ids),), self.device
        )
        inclination_deltas, spacing_deltas, nominal_mask = (
            mix_nominal_ladder_geometry_samples(
                inclination_deltas,
                spacing_deltas,
                self.cfg.nominal_ladder_geometry_probability,
            )
        )
        self.ladder_phi[env_ids] = self.ladder_geometry.phi + inclination_deltas
        self.ladder_spacing[env_ids] = self.ladder_geometry.rung_spacing + spacing_deltas
        self._nominal_ladder_geometry_episode[env_ids] = nominal_mask
        zeros = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
        self._ladder_delta_quat[env_ids] = quat_from_euler_xyz(
            zeros,
            self.ladder_geometry.phi - self.ladder_phi[env_ids],
            zeros,
        )
        if self._ladder_asset is not None:
            self._write_ladder_box_poses(env_ids)

    def _write_ladder_box_poses(self, env_ids: torch.Tensor) -> None:
        target_cos = torch.cos(self.ladder_phi[env_ids]).unsqueeze(-1)
        target_sin = torch.sin(self.ladder_phi[env_ids]).unsqueeze(-1)
        axial = self._ladder_box_axial.unsqueeze(0) * (
            self.ladder_spacing[env_ids].unsqueeze(-1) / self.ladder_geometry.rung_spacing
        )
        normal = self._ladder_box_normal.unsqueeze(0)
        positions_e = self._ladder_box_positions_e.unsqueeze(0).repeat(len(env_ids), 1, 1)
        positions_e[..., 0] = self._ladder_origin_e[0] + axial * target_cos - normal * target_sin
        positions_e[..., 2] = self._ladder_origin_e[2] + axial * target_sin + normal * target_cos
        positions_e[:, self._ladder_fixed_box_mask] = self._ladder_box_positions_e[
            self._ladder_fixed_box_mask
        ]
        rung_indexes = torch.arange(
            len(self.cfg.ladder_rung_box_names), dtype=torch.float32, device=self.device
        ).view(1, -1, 1)
        directions = torch.stack(
            (
                torch.cos(self.ladder_phi[env_ids]),
                torch.zeros_like(self.ladder_phi[env_ids]),
                torch.sin(self.ladder_phi[env_ids]),
            ),
            dim=-1,
        ).unsqueeze(1)
        rung_tops = (
            self._ladder_origin_e.view(1, 1, 3)
            + rung_indexes * self.ladder_spacing[env_ids].view(-1, 1, 1) * directions
        )
        positions_e[:, self._ladder_rung_box_indexes] = (
            rung_tops + self._ladder_rung_center_offsets_e.unsqueeze(0)
        )

        quaternions = self._ladder_box_quats.unsqueeze(0).repeat(len(env_ids), 1, 1)
        if torch.any(self._ladder_inclined_box_mask):
            delta_quat = self._ladder_delta_quat[env_ids].unsqueeze(1).expand(-1, quaternions.shape[1], -1)
            rotated_quaternions = quat_mul(delta_quat, quaternions)
            quaternions[:, self._ladder_inclined_box_mask] = rotated_quaternions[
                :, self._ladder_inclined_box_mask
            ]
        positions_w = positions_e + self._env.scene.env_origins[env_ids].unsqueeze(1)
        self._ladder_asset.write_object_pose_to_sim(
            torch.cat((positions_w, quaternions), dim=-1), env_ids=env_ids
        )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def stop_holding(self) -> torch.Tensor:
        return self._stop_state == STOP_STATE_HOLD

    @property
    def effective_climb_command(self) -> torch.Tensor:
        direction = self.motion.climb_direction[self.time_steps]
        return torch.where(
            self.stop_holding, torch.zeros_like(direction), direction
        ).unsqueeze(-1)

    @property
    def stop_requested(self) -> torch.Tensor:
        return self._stop_state != STOP_STATE_RUN

    def _zero_velocity_while_holding(self, value: torch.Tensor) -> torch.Tensor:
        stable_holding = getattr(
            self,
            "_stable_anchor_holding",
            torch.zeros_like(self.stop_holding),
        )
        reference_holding = self.stop_holding | stable_holding
        mask_shape = (self.num_envs,) + (1,) * (value.ndim - 1)
        return torch.where(
            reference_holding.reshape(mask_shape), torch.zeros_like(value), value
        )

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._zero_velocity_while_holding(self.motion.joint_vel[self.time_steps])

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._zero_velocity_while_holding(self.motion.body_lin_vel_w[self.time_steps])

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._zero_velocity_while_holding(self.motion.body_ang_vel_w[self.time_steps])

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._zero_velocity_while_holding(
            self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]
        )

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._zero_velocity_while_holding(
            self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]
        )

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def contact_indicator(self) -> torch.Tensor:
        return self.motion.contact_indicator[self.time_steps]

    @property
    def contact_rung_id(self) -> torch.Tensor:
        return self.motion.contact_rung_id[self.time_steps]

    @property
    def task_goal_rung_id(self) -> torch.Tensor:
        """Current support rung, or the next scheduled rung while swinging."""

        return self.motion.task_goal_rung_id[self.time_steps]

    @property
    def task_goal_valid(self) -> torch.Tensor:
        """Whether each limb still has a current or future ladder target."""

        return self.motion.task_goal_valid[self.time_steps]

    @property
    def motion_semantic_target_valid(self) -> torch.Tensor:
        """Whether a state-locked stable whole-body target is available."""

        return self._motion_target_frame >= 0

    @property
    def motion_semantic_target_joint_pos(self) -> torch.Tensor:
        """Joint pose at the active transition's terminal stable anchor."""

        return self.motion.joint_pos[self._motion_target_frame.clamp_min(0)]

    @property
    def motion_semantic_target_root_quat_w(self) -> torch.Tensor:
        """Root orientation at the active transition's terminal stable anchor."""

        return self.motion.body_quat_w[
            self._motion_target_frame.clamp_min(0), self.motion_anchor_body_index
        ]

    @property
    def motion_semantic_target_contact_indicator(self) -> torch.Tensor:
        return self.motion.stable_support_contact_indicator[
            self._motion_target_frame.clamp_min(0)
        ]

    @property
    def motion_semantic_target_contact_rung_id(self) -> torch.Tensor:
        return self.motion.stable_support_rung_id[
            self._motion_target_frame.clamp_min(0)
        ]

    @property
    def ladder_origins_w(self) -> torch.Tensor:
        return self._ladder_origin_e.unsqueeze(0) + self._env.scene.env_origins

    @property
    def rung_positions_w(self) -> torch.Tensor:
        rung_indexes = torch.arange(
            self.ladder_geometry.num_rungs, dtype=torch.float32, device=self.device
        ).view(1, -1, 1)
        directions = torch.stack(
            (
                torch.cos(self.ladder_phi),
                torch.zeros_like(self.ladder_phi),
                torch.sin(self.ladder_phi),
            ),
            dim=-1,
        ).unsqueeze(1)
        origins = self.ladder_origins_w.unsqueeze(1)
        return origins + rung_indexes * self.ladder_spacing.view(-1, 1, 1) * directions

    @property
    def target_contact_pos_w(self) -> torch.Tensor:
        return self._limb_goal_positions_for_rung_ids(self.contact_rung_id)

    @property
    def stable_support_contact_indicator(self) -> torch.Tensor:
        """Ground-aware support labels at the active reference frame."""

        return self.motion.stable_support_contact_indicator[self.time_steps]

    @property
    def stable_support_rung_id(self) -> torch.Tensor:
        """Ground/rung support IDs at the active reference frame."""

        return self.motion.stable_support_rung_id[self.time_steps]

    @property
    def stable_support_target_pos_w(self) -> torch.Tensor:
        """World-space targets resolved from the active stable-support IDs."""

        return self._stable_support_goal_positions_for_frames(self.time_steps)

    @property
    def task_limb_goal_pos_w(self) -> torch.Tensor:
        """Task goals for support limbs and their next landing goals for swing limbs."""

        return self._limb_goal_positions_for_rung_ids(self.task_goal_rung_id)

    @property
    def motion_semantic_target_contact_pos_w(self) -> torch.Tensor:
        return self._stable_support_goal_positions_for_frames(
            self._motion_target_frame.clamp_min(0)
        )

    @property
    def next_stable_limb_target_pos_w(self) -> torch.Tensor:
        """Ladder target points at the next stable support frame."""

        target_frame = self.next_stable_target_frame.clamp_min(0)
        return self._stable_support_goal_positions_for_frames(target_frame)

    @property
    def next_stable_target_frame(self) -> torch.Tensor:
        """Pending stable-support anchor for each environment."""

        return self.motion.initial_stable_motion_target_frame[self.time_steps]

    @property
    def next_stable_limb_target_valid(self) -> torch.Tensor:
        """Per-limb validity of the next stable support target."""

        target_frame = self.next_stable_target_frame
        return (
            (target_frame >= 0)[:, None]
            & (
                self.motion.stable_support_contact_indicator[
                    target_frame.clamp_min(0)
                ]
                > 0.5
            )
        )

    @property
    def next_stable_contact_indicator(self) -> torch.Tensor:
        """Desired four-limb contacts at the next stable support frame."""

        target_frame = self.next_stable_target_frame
        indicator = self.motion.stable_support_contact_indicator[
            target_frame.clamp_min(0)
        ]
        return torch.where(
            (target_frame >= 0)[:, None], indicator, torch.zeros_like(indicator)
        )

    def _stable_support_goal_positions_for_frames(
        self, target_frames: torch.Tensor
    ) -> torch.Tensor:
        """Resolve ground/rung support IDs to world-space limb targets.

        Ground targets use the reference limb positions at the shared anchor.
        Ladder support ID ``k`` maps to the existing geometry rung ``k - 1``.
        """

        support_rung_ids = self.motion.stable_support_rung_id[target_frames]
        ladder_targets = self._limb_goal_positions_for_rung_ids(
            (support_rung_ids - 1).clamp_min(0)
        )
        if self.motion._stable_support_limb_body_indexes is None:
            return ladder_targets
        reference_targets = self.motion._body_pos_w[
            target_frames[:, None],
            self.motion._stable_support_limb_body_indexes[None, :],
        ] + self._env.scene.env_origins[:, None, :]
        ground_targets = reference_targets
        return torch.where(
            (support_rung_ids < 0).unsqueeze(-1),
            reference_targets,
            torch.where(
                (support_rung_ids == 0).unsqueeze(-1),
                ground_targets,
                ladder_targets,
            ),
        )

    def _limb_goal_positions_for_rung_ids(self, rung_ids: torch.Tensor) -> torch.Tensor:
        rung_ids = torch.clamp(rung_ids, 0, self.ladder_geometry.num_rungs - 1)
        targets = torch.gather(
            self.rung_positions_w,
            1,
            rung_ids.unsqueeze(-1).expand(-1, -1, 3),
        )
        return targets + self._limb_offsets.unsqueeze(0)

    @property
    def robot_limb_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.limb_body_indexes]

    @property
    def ladder_params(self) -> torch.Tensor:
        return torch.stack(
            (
                self.ladder_phi,
                self.ladder_spacing,
                torch.full_like(self.ladder_phi, self.ladder_geometry.rung_length),
                torch.full_like(self.ladder_phi, self.ladder_geometry.rung_width),
            ),
            dim=-1,
        )

    def _update_metrics(self):
        reference_anchor_pos_w = self.anchor_pos_w
        robot_anchor_pos_w = self.robot_anchor_pos_w
        initialized = self._tracking_path_initialized.clone()
        if torch.any(~initialized):
            self._tracking_start_reference_anchor_pos_w[~initialized] = reference_anchor_pos_w[~initialized]
            self._tracking_start_robot_anchor_pos_w[~initialized] = robot_anchor_pos_w[~initialized]
            self._tracking_previous_reference_anchor_pos_w[~initialized] = reference_anchor_pos_w[~initialized]
            self._tracking_previous_robot_anchor_pos_w[~initialized] = robot_anchor_pos_w[~initialized]

        reference_step = torch.linalg.norm(
            reference_anchor_pos_w - self._tracking_previous_reference_anchor_pos_w, dim=-1
        )
        robot_step = torch.linalg.norm(robot_anchor_pos_w - self._tracking_previous_robot_anchor_pos_w, dim=-1)
        reference_step = torch.where(initialized, reference_step, torch.zeros_like(reference_step))
        robot_step = torch.where(initialized, robot_step, torch.zeros_like(robot_step))
        self.metrics["reference_anchor_path_length"] += reference_step
        self.metrics["robot_anchor_path_length"] += robot_step
        self._tracking_previous_reference_anchor_pos_w.copy_(reference_anchor_pos_w)
        self._tracking_previous_robot_anchor_pos_w.copy_(robot_anchor_pos_w)
        self._tracking_path_initialized[:] = True

        reference_displacement_vector = reference_anchor_pos_w - self._tracking_start_reference_anchor_pos_w
        robot_displacement_vector = robot_anchor_pos_w - self._tracking_start_robot_anchor_pos_w
        reference_displacement = torch.linalg.norm(reference_displacement_vector, dim=-1)
        robot_displacement = torch.linalg.norm(robot_displacement_vector, dim=-1)
        progress_evaluable = reference_displacement > _REFERENCE_PROGRESS_DISTANCE_THRESHOLD
        progress_ratio = torch.sum(robot_displacement_vector * reference_displacement_vector, dim=-1) / (
            torch.square(reference_displacement).clamp_min(1.0e-6)
        )
        self.metrics["reference_anchor_displacement"] = reference_displacement
        self.metrics["robot_anchor_displacement"] = robot_displacement
        self.metrics["anchor_progress_ratio"] = torch.where(
            progress_evaluable, progress_ratio, torch.zeros_like(progress_ratio)
        )
        self.metrics["progress_evaluable"] = progress_evaluable.float()

        reference_speed = torch.linalg.norm(self.anchor_lin_vel_w, dim=-1)
        robot_speed = torch.linalg.norm(self.robot_anchor_lin_vel_w, dim=-1)
        reference_moving = reference_speed > _REFERENCE_MOVING_SPEED_THRESHOLD
        robot_stationary = robot_speed < _ROBOT_STATIONARY_SPEED_THRESHOLD
        self._tracking_episode_steps += initialized.float()
        self._tracking_reference_motion_steps += reference_moving.float()
        self._tracking_stationary_motion_steps += (reference_moving & robot_stationary).float()
        self.metrics["reference_anchor_speed"] = reference_speed
        self.metrics["robot_anchor_speed"] = robot_speed
        self.metrics["reference_joint_speed"] = torch.linalg.norm(self.joint_vel, dim=-1)
        self.metrics["robot_joint_speed"] = torch.linalg.norm(self.robot_joint_vel, dim=-1)
        self.metrics["reference_moving_fraction"] = self._tracking_reference_motion_steps / self._tracking_episode_steps.clamp_min(1.0)
        self.metrics["stationary_during_reference_motion_fraction"] = (
            self._tracking_stationary_motion_steps / self._tracking_reference_motion_steps.clamp_min(1.0)
        )
        remaining_frames = (self.motion.time_step_total - 1 - self._episode_start_time_steps).clamp_min(1)
        self.metrics["reference_completion_fraction"] = (
            (self.time_steps - self._episode_start_time_steps).clamp_min(0).float() / remaining_frames.float()
        ).clamp(max=1.0)
        stop_target_valid = self._stop_target_frame >= 0
        stop_frames_remaining = torch.where(
            stop_target_valid,
            (self._stop_target_frame - self.time_steps).clamp_min(0),
            torch.zeros_like(self.time_steps),
        )
        self.metrics["stop_command_requested"] = self.stop_requested.float()
        self.metrics["stop_command_seeking"] = (
            self._stop_state == STOP_STATE_SEEK
        ).float()
        self.metrics["stop_command_holding"] = self.stop_holding.float()
        self.metrics["stop_target_frame_fraction"] = torch.where(
            stop_target_valid,
            self._stop_target_frame.float() / max(self.motion.time_step_total - 1, 1),
            torch.zeros_like(self._stop_target_frame, dtype=torch.float32),
        )
        self.metrics["stop_time_to_target_s"] = (
            stop_frames_remaining.float() / self._control_fps
        )
        self.metrics["stop_training_direct_hold_episode"] = (
            self._stop_training_direct_hold_episode.float()
        )
        self.metrics["stop_training_seek_episode"] = (
            self._stop_training_seek_episode.float()
        )

        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)
        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        next_stable_valid = self.next_stable_limb_target_valid
        next_stable_distance = torch.linalg.vector_norm(
            self.robot_limb_pos_w - self.next_stable_limb_target_pos_w, dim=-1
        )
        next_stable_valid_count = next_stable_valid.sum(dim=-1)
        self.metrics["next_stable_target_mean_distance"] = torch.where(
            next_stable_valid_count > 0,
            (next_stable_distance * next_stable_valid).sum(dim=-1)
            / next_stable_valid_count.clamp_min(1).to(next_stable_distance.dtype),
            torch.zeros_like(next_stable_distance[:, 0]),
        )

        errors = torch.norm(self.robot_limb_pos_w - self.target_contact_pos_w, dim=-1)
        contact_forces_w = self._contact_sensor.data.net_forces_w[:, self._contact_limb_body_indexes, :]
        required, physical_contact, target_contact = classify_limb_contacts(
            self.contact_indicator,
            self.robot_limb_pos_w,
            self.target_contact_pos_w,
            contact_forces_w,
            self._limb_target_distance_thresholds,
            self.cfg.contact_force_threshold,
        )
        required_count = required.sum(dim=-1).clamp(min=1)
        self.metrics["mean_required_contact_error"] = (errors * required).sum(dim=-1) / required_count
        self.metrics["required_contact_fraction"] = required.float().mean(dim=-1)
        required_physical = required & physical_contact
        self.metrics["required_physical_contact_fraction"] = (
            required_physical.sum(dim=-1).float() / required_count
        )
        self.metrics["required_target_contact_fraction"] = (
            target_contact.sum(dim=-1).float() / required_count
        )
        self.metrics["swing_physical_contact_fraction"] = (
            ((~required) & physical_contact).float().mean(dim=-1)
        )
        for limb_index, limb_name in enumerate(self.limb_names):
            self.metrics[f"{limb_name}_physical_contact"] = physical_contact[:, limb_index].float()
            self.metrics[f"{limb_name}_required_physical_contact"] = (
                required_physical[:, limb_index].float()
            )
            self.metrics[f"{limb_name}_required_target_contact"] = target_contact[:, limb_index].float()
        for hand_name in ("left_hand", "right_hand"):
            hand_index = self.limb_names.index(hand_name)
            hand_required = required[:, hand_index]
            self.metrics[f"{hand_name}_required_contact_error"] = (
                errors[:, hand_index] * hand_required
            )
            self.metrics[f"{hand_name}_contact_required"] = hand_required.float()
        self._update_contact_force_metrics()
        self.metrics["fixed_start_episode"] = self._fixed_start_episode.float()
        self.metrics["secondary_fixed_start_episode"] = (
            self._secondary_fixed_start_episode.float()
        )
        self.metrics["fixed_start_pose_randomized_episode"] = (
            self._fixed_start_pose_randomized_episode.float()
        )
        self.metrics["fixed_start_pose_nominal_episode"] = (
            self._fixed_start_episode & ~self._fixed_start_pose_randomized_episode
        ).float()
        self.metrics["uniform_start_episode"] = self._uniform_start_episode.float()
        self.metrics["adaptive_start_episode"] = (
            ~self._fixed_start_episode
            & ~self._secondary_fixed_start_episode
            & ~self._uniform_start_episode
        ).float()
        self.metrics["ladder_inclination_deg"] = torch.rad2deg(self.ladder_phi)
        self.metrics["ladder_rung_spacing"] = self.ladder_spacing
        self.metrics["nominal_ladder_geometry_episode"] = (
            self._nominal_ladder_geometry_episode.float()
        )

    def _finalize_force_histograms(self, env_ids: Sequence[int] | slice | None) -> None:
        if not self._force_histograms:
            return
        resolved_env_ids = slice(None) if env_ids is None else env_ids
        for limb_name, histogram in self._force_histograms.items():
            selected = histogram[resolved_env_ids]
            total = selected.sum(dim=-1)
            target = 0.95 * total
            cumulative = selected.cumsum(dim=-1)
            reached = cumulative >= target.unsqueeze(-1)
            bin_index = reached.float().argmax(dim=-1)
            p95 = (bin_index.float() + 0.5) * (
                self._force_histogram_max[limb_name] / self._force_histogram_bin_count
            )
            p95 = torch.where(total > 0.0, p95, torch.zeros_like(p95))
            self.metrics[f"{limb_name}_force_p95"][resolved_env_ids] = p95

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        self._finalize_force_histograms(env_ids)
        return super().reset(env_ids)

    def _update_contact_force_metrics(self) -> None:
        current_forces: dict[str, torch.Tensor] = {}
        for metric_name, (sensor, surface) in self._contact_force_metric_specs.items():
            force_matrix_w = sensor.data.force_matrix_w
            if force_matrix_w is None:
                continue
            current_max = filtered_surface_force_max(
                force_matrix_w,
                surface=surface,
                min_top_normal_fraction=(
                    self.cfg.contact_front_normal_fraction
                    if surface == "front"
                    else 0.5
                ),
            )
            self.metrics[metric_name] = torch.maximum(
                self.metrics[metric_name], current_max
            )
            current_forces[metric_name.removesuffix("_force_max")] = current_max

        for side, limb_name in (("left", "left_hand"), ("right", "right_hand")):
            sensor_name = f"{side}_hand_step_contacts"
            if sensor_name not in self._env.scene.sensors:
                continue
            force_matrix_w = self._env.scene.sensors[sensor_name].data.force_matrix_w
            if force_matrix_w is None:
                continue
            limb_index = self.limb_names.index(limb_name)
            target_forces_w = target_rung_forces(
                force_matrix_w,
                self.contact_rung_id[:, limb_index : limb_index + 1],
            )
            target_upward = torch.clamp_min(target_forces_w[:, 0, 2], 0.0)
            metric_name = f"{side}_hand_target_rung_upward_force_max"
            self.metrics[metric_name] = torch.maximum(
                self.metrics[metric_name], target_upward
            )
            current_forces[f"{side}_hand_target_rung_upward"] = target_upward
            non_target_forces_w = non_target_rung_forces(
                force_matrix_w,
                self.contact_rung_id[:, limb_index : limb_index + 1],
            )
            wrong_rung_top = filtered_surface_force_max(
                non_target_forces_w,
                surface="top",
                min_top_normal_fraction=0.5,
            )
            wrong_metric_name = f"{side}_hand_wrong_rung_force_max"
            self.metrics[wrong_metric_name] = torch.maximum(
                self.metrics[wrong_metric_name], wrong_rung_top
            )
            current_forces[f"{side}_hand_wrong_rung"] = wrong_rung_top

        for side, limb_name in (("left", "left_foot"), ("right", "right_foot")):
            sensor_name = f"{side}_foot_step_contacts"
            if sensor_name not in self._env.scene.sensors:
                continue
            force_matrix_w = self._env.scene.sensors[sensor_name].data.force_matrix_w
            if force_matrix_w is None:
                continue
            limb_index = self.limb_names.index(limb_name)
            target_forces_w = target_rung_forces(
                force_matrix_w,
                self.contact_rung_id[:, limb_index : limb_index + 1],
            )
            target_force = torch.linalg.vector_norm(target_forces_w[:, 0], dim=-1)
            metric_name = f"{side}_foot_target_rung_force_max"
            self.metrics[metric_name] = torch.maximum(
                self.metrics[metric_name], target_force
            )
            current_forces[f"{side}_foot_target_rung"] = target_force

        if self._force_histograms:
            histogram_values = {
                "left_hand": [
                    value
                    for name, value in current_forces.items()
                    if name.startswith("left_hand_") and not name.endswith("target_rung_upward")
                ],
                "right_hand": [
                    value
                    for name, value in current_forces.items()
                    if name.startswith("right_hand_") and not name.endswith("target_rung_upward")
                ],
                "left_foot": [
                    value
                    for name, value in current_forces.items()
                    if name == "left_foot_step"
                ],
                "right_foot": [
                    value
                    for name, value in current_forces.items()
                    if name == "right_foot_step"
                ],
            }
            for limb_name, values in histogram_values.items():
                if not values:
                    continue
                current_force = torch.stack(values, dim=-1).amax(dim=-1)
                bin_index = torch.clamp(
                    (
                        current_force
                        / self._force_histogram_max[limb_name]
                        * self._force_histogram_bin_count
                    ).long(),
                    min=0,
                    max=self._force_histogram_bin_count - 1,
                )
                self._force_histograms[limb_name].scatter_add_(
                    1,
                    bin_index.unsqueeze(-1),
                    torch.ones(
                        self.num_envs, 1, device=self.device, dtype=self._force_histograms[limb_name].dtype
                    ),
                )

        if self._contact_force_soft_limits:
            limit_values_by_limb = {
                "left_hand": {
                    name: value
                    for name, value in current_forces.items()
                    if name.startswith("left_hand_")
                },
                "right_hand": {
                    name: value
                    for name, value in current_forces.items()
                    if name.startswith("right_hand_")
                },
                "left_foot": {
                    name: value
                    for name, value in current_forces.items()
                    if name.startswith("left_foot_")
                },
                "right_foot": {
                    name: value
                    for name, value in current_forces.items()
                    if name.startswith("right_foot_")
                },
            }
            for limb_name, values in limit_values_by_limb.items():
                comparable = []
                for metric_name, force_value in values.items():
                    limit_key = metric_name.split("_", 1)[1]
                    if limit_key in self._contact_force_soft_limits:
                        comparable.append(
                            (
                                force_value,
                                self._contact_force_soft_limits[limit_key],
                                self._contact_force_hard_limits[limit_key],
                            )
                        )
                if not comparable:
                    continue
                soft_exceeded = torch.stack(
                    [force_value > force_soft for force_value, force_soft, _ in comparable],
                    dim=-1,
                ).any(dim=-1)
                hard_exceeded = torch.stack(
                    [force_value > force_hard for force_value, _, force_hard in comparable],
                    dim=-1,
                ).any(dim=-1)
                self._force_limit_sample_count[limb_name] += 1.0
                self._force_limit_over_soft_count[limb_name] += soft_exceeded.float()
                hard_event = hard_exceeded & ~self._force_limit_previous_hard[limb_name]
                self.metrics[f"{limb_name}_force_over_soft_fraction"] = (
                    self._force_limit_over_soft_count[limb_name]
                    / self._force_limit_sample_count[limb_name].clamp_min(1.0)
                )
                self.metrics[f"{limb_name}_hard_collision_count"] += hard_event.float()
                self._force_limit_previous_hard[limb_name] = hard_exceeded

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        reset_valid_masks = self.motion.geometry_reset_valid_masks(
            self.ladder_phi[env_ids] - self.ladder_geometry.phi,
            self.ladder_spacing[env_ids] - self.ladder_geometry.rung_spacing,
        )
        self._current_reset_valid_fraction[env_ids] = reset_valid_masks.float().mean(dim=-1)
        if self.cfg.fixed_reference_frame is not None:
            if torch.any(~reset_valid_masks[:, self.cfg.fixed_reference_frame]):
                raise ValueError(
                    f"fixed_reference_frame {self.cfg.fixed_reference_frame} is invalid for at least "
                    "one sampled ladder geometry; disable geometry randomization for this diagnostic"
                )
            if (
                self.cfg.fixed_reference_frame
                == self.cfg.secondary_fixed_start_frame
                and self.cfg.secondary_fixed_start_requires_nominal_ladder
                and torch.any(~self._nominal_ladder_geometry_episode[env_ids])
            ):
                raise ValueError(
                    "the secondary fixed start requires nominal ladder geometry; "
                    "disable ladder randomization for this fixed-frame diagnostic"
                )
            self.time_steps[env_ids] = self.cfg.fixed_reference_frame
            # A fixed frame-0 replay is still a fixed-start episode.  Preserve
            # the dedicated frame-0 reset pose randomization in this mode.
            self._fixed_start_episode[env_ids] = (
                self.cfg.fixed_reference_frame == self.cfg.fixed_start_frame
            )
            self._secondary_fixed_start_episode[env_ids] = (
                self.cfg.fixed_reference_frame
                == self.cfg.secondary_fixed_start_frame
            )
            self._uniform_start_episode[env_ids] = False
            return
        if self.cfg.fixed_start_probability == 1.0:
            self.time_steps[env_ids] = self.cfg.fixed_start_frame
            self._fixed_start_episode[env_ids] = True
            self._secondary_fixed_start_episode[env_ids] = False
            self._uniform_start_episode[env_ids] = False
            fixed_bin = min(
                self.cfg.fixed_start_frame * self.bin_count // self.motion.time_step_total,
                self.bin_count - 1,
            )
            self.metrics["sampling_entropy"][env_ids] = 0.0
            self.metrics["sampling_top1_prob"][env_ids] = 1.0
            self.metrics["sampling_top1_bin"][env_ids] = fixed_bin / self.bin_count
            self.metrics["adaptive_sampling_top1_prob"][env_ids] = 0.0
            self.metrics["adaptive_sampling_top1_bin"][env_ids] = 0.0
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        nonfixed_valid_masks = reset_valid_masks.clone()
        if self.cfg.use_reset_valid_mask:
            nonfixed_valid_masks[:, self.cfg.fixed_start_frame] = False
            if self.cfg.secondary_fixed_start_frame is not None:
                nonfixed_valid_masks[:, self.cfg.secondary_fixed_start_frame] = False
        if torch.any(~torch.any(nonfixed_valid_masks, dim=-1)):
            raise ValueError("sampled ladder geometry has no non-fixed valid reset frame")
        all_frame_indexes = torch.arange(
            self.motion.time_step_total, dtype=torch.long, device=self.device
        )
        frame_bins = torch.div(
            all_frame_indexes * self.bin_count,
            self.motion.time_step_total,
            rounding_mode="floor",
        ).clamp(max=self.bin_count - 1)
        valid_count_per_bin = torch.zeros(
            len(env_ids), self.bin_count, dtype=torch.float32, device=self.device
        )
        valid_count_per_bin.scatter_add_(
            1,
            frame_bins.unsqueeze(0).expand(len(env_ids), -1),
            nonfixed_valid_masks.float(),
        )
        per_frame_valid_count = torch.gather(
            valid_count_per_bin, 1, frame_bins.unsqueeze(0).expand(len(env_ids), -1)
        )
        adaptive_frame_probabilities = torch.where(
            nonfixed_valid_masks,
            sampling_probabilities[frame_bins].unsqueeze(0)
            / per_frame_valid_count.clamp_min(1.0),
            0.0,
        )
        adaptive_frame_probabilities /= adaptive_frame_probabilities.sum(dim=-1, keepdim=True)
        uniform_frame_probabilities = nonfixed_valid_masks.float()
        uniform_frame_probabilities /= uniform_frame_probabilities.sum(dim=-1, keepdim=True)
        if self.cfg.reset_by_stable_support_segments:
            adaptive_frames, adaptive_frame_probabilities = (
                sample_reset_frames_by_segments(
                    nonfixed_valid_masks,
                    adaptive_frame_probabilities,
                    self.motion.stable_reset_segment_ids,
                    self.motion.stable_reset_segment_bounds,
                    equal_segments=False,
                )
            )
            uniform_frames, uniform_frame_probabilities = (
                sample_reset_frames_by_segments(
                    nonfixed_valid_masks,
                    uniform_frame_probabilities,
                    self.motion.stable_reset_segment_ids,
                    self.motion.stable_reset_segment_bounds,
                    equal_segments=True,
                )
            )
        else:
            adaptive_frames = torch.multinomial(
                adaptive_frame_probabilities, 1, replacement=True
            ).squeeze(-1)
            uniform_frames = torch.multinomial(
                uniform_frame_probabilities, 1, replacement=True
            ).squeeze(-1)
        if self.cfg.secondary_fixed_start_probability > 0.0:
            if (
                self.cfg.secondary_fixed_start_requires_nominal_ladder
                and self._geometry_randomization_enabled
            ):
                secondary_eligible_mask = self._nominal_ladder_geometry_episode[env_ids]
                secondary_eligible_probability = (
                    self.cfg.nominal_ladder_geometry_probability
                )
            else:
                secondary_eligible_mask = torch.ones(
                    len(env_ids), dtype=torch.bool, device=self.device
                )
                secondary_eligible_probability = 1.0
            (
                sampled_frames,
                fixed_start_mask,
                secondary_fixed_start_mask,
                uniform_start_mask,
            ) = mix_two_anchor_reset_frame_samples(
                adaptive_frames,
                uniform_frames,
                self.cfg.fixed_start_frame,
                self.cfg.fixed_start_probability,
                self.cfg.secondary_fixed_start_frame,
                self.cfg.secondary_fixed_start_probability,
                self.cfg.uniform_start_probability,
                secondary_eligible_mask,
                secondary_eligible_probability,
            )
        else:
            sampled_frames, fixed_start_mask, uniform_start_mask = mix_reset_frame_samples(
                adaptive_frames,
                uniform_frames,
                self.cfg.fixed_start_frame,
                self.cfg.fixed_start_probability,
                self.cfg.uniform_start_probability,
            )
            secondary_fixed_start_mask = torch.zeros_like(
                fixed_start_mask, dtype=torch.bool
            )
        self.time_steps[env_ids] = sampled_frames
        self._fixed_start_episode[env_ids] = fixed_start_mask
        self._secondary_fixed_start_episode[env_ids] = secondary_fixed_start_mask
        self._uniform_start_episode[env_ids] = uniform_start_mask

        adaptive_bin_probabilities = torch.zeros_like(sampling_probabilities)
        adaptive_bin_probabilities.scatter_add_(
            0, frame_bins, adaptive_frame_probabilities.mean(dim=0)
        )
        uniform_bin_probabilities = torch.zeros_like(sampling_probabilities)
        uniform_bin_probabilities.scatter_add_(
            0,
            frame_bins,
            uniform_frame_probabilities.mean(dim=0),
        )
        adaptive_probability = (
            1.0
            - self.cfg.fixed_start_probability
            - self.cfg.secondary_fixed_start_probability
            - self.cfg.uniform_start_probability
        )
        effective_probabilities = (
            adaptive_probability * adaptive_bin_probabilities
            + self.cfg.uniform_start_probability * uniform_bin_probabilities
        )
        fixed_bin = min(
            self.cfg.fixed_start_frame * self.bin_count // self.motion.time_step_total,
            self.bin_count - 1,
        )
        effective_probabilities[fixed_bin] += self.cfg.fixed_start_probability
        if self.cfg.secondary_fixed_start_frame is not None:
            secondary_fixed_bin = min(
                self.cfg.secondary_fixed_start_frame
                * self.bin_count
                // self.motion.time_step_total,
                self.bin_count - 1,
            )
            effective_probabilities[secondary_fixed_bin] += (
                self.cfg.secondary_fixed_start_probability
            )
        entropy = -(effective_probabilities * (effective_probabilities + 1e-12).log()).sum()
        entropy_norm = entropy / math.log(self.bin_count)
        pmax, imax = effective_probabilities.max(dim=0)
        adaptive_pmax, adaptive_imax = adaptive_bin_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][env_ids] = entropy_norm
        self.metrics["sampling_top1_prob"][env_ids] = pmax
        self.metrics["sampling_top1_bin"][env_ids] = imax.float() / self.bin_count
        self.metrics["adaptive_sampling_top1_prob"][env_ids] = adaptive_pmax
        self.metrics["adaptive_sampling_top1_bin"][env_ids] = adaptive_imax.float() / self.bin_count

    def _sample_duration_steps(
        self, interval_s: tuple[float, float], count: int
    ) -> torch.Tensor:
        minimum = int(round(interval_s[0] * self._control_fps))
        maximum = int(round(interval_s[1] * self._control_fps))
        if minimum == maximum:
            return torch.full((count,), minimum, dtype=torch.long, device=self.device)
        return torch.randint(
            minimum, maximum + 1, (count,), dtype=torch.long, device=self.device
        )

    def _sample_direct_hold_frames(self, env_ids: torch.Tensor) -> torch.Tensor:
        candidates = self.motion.zero_command_direct_hold_frames
        reset_masks = self.motion.geometry_reset_valid_masks(
            self.ladder_phi[env_ids] - self.ladder_geometry.phi,
            self.ladder_spacing[env_ids] - self.ladder_geometry.rung_spacing,
        )
        candidate_probabilities = reset_masks[:, candidates].float()
        if torch.any(candidate_probabilities.sum(dim=-1) == 0.0):
            raise ValueError(
                "at least one sampled ladder geometry excludes every natural stop frame"
            )
        sampled_candidate_indexes = torch.multinomial(
            candidate_probabilities, 1, replacement=True
        ).squeeze(-1)
        return candidates[sampled_candidate_indexes]

    def _sample_scheduled_stop_frames(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a future request time without replacing the existing reset frame."""

        reset_masks = self.motion.geometry_reset_valid_masks(
            self.ladder_phi[env_ids] - self.ladder_geometry.phi,
            self.ladder_spacing[env_ids] - self.ladder_geometry.rung_spacing,
        )
        stop_masks = reset_masks & self.motion.zero_command_valid_mask.unsqueeze(0)
        frame_indexes = torch.arange(
            self.motion.time_step_total, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        future_stop_masks = stop_masks & (
            frame_indexes >= self.time_steps[env_ids].unsqueeze(-1)
        )
        has_future = torch.any(future_stop_masks, dim=-1)
        if torch.any(~has_future):
            failed_frames = self.time_steps[env_ids][~has_future].tolist()
            raise ValueError(
                "scheduled zero command has no geometry-valid future stop after "
                f"reference frames {failed_frames}"
            )
        _, reverse_indexes = torch.max(
            torch.flip(future_stop_masks, dims=(-1,)), dim=-1
        )
        last_stop_frames = self.motion.time_step_total - 1 - reverse_indexes
        remaining_frames = last_stop_frames - self.time_steps[env_ids] + 1
        trigger_offsets = torch.floor(
            torch.rand(len(env_ids), device=self.device) * remaining_frames.float()
        ).long()
        trigger_frames = self.time_steps[env_ids] + trigger_offsets
        target_frames = next_valid_frame_per_env(stop_masks, trigger_frames)
        if torch.any(target_frames < 0):
            raise RuntimeError("failed to resolve a scheduled zero-command target")
        return trigger_frames, target_frames

    def _sample_stop_command_training(self, env_ids: torch.Tensor) -> None:
        self._stop_state[env_ids] = STOP_STATE_RUN
        self._stop_target_frame[env_ids] = -1
        self._stop_trigger_frame[env_ids] = -1
        self._stop_hold_steps_remaining[env_ids] = 0
        self._stop_training_direct_hold_episode[env_ids] = False
        self._stop_training_seek_episode[env_ids] = False
        if not self.cfg.stop_command_training_enabled:
            return

        draw = torch.rand(len(env_ids), device=self.device)
        direct_mask = draw < self.cfg.stop_hold_reset_probability
        seek_mask = (draw >= self.cfg.stop_hold_reset_probability) & (
            draw
            < self.cfg.stop_hold_reset_probability
            + self.cfg.stop_seek_episode_probability
        )

        direct_env_ids = env_ids[direct_mask]
        if direct_env_ids.numel() > 0:
            targets = self._sample_direct_hold_frames(direct_env_ids)
            self.time_steps[direct_env_ids] = targets
            self._stop_state[direct_env_ids] = STOP_STATE_HOLD
            self._stop_target_frame[direct_env_ids] = targets
            self._stop_trigger_frame[direct_env_ids] = targets
            self._stop_hold_steps_remaining[direct_env_ids] = self._sample_duration_steps(
                self.cfg.stop_hold_time_range_s, len(direct_env_ids)
            )
            self._stop_training_direct_hold_episode[direct_env_ids] = True

        seek_env_ids = env_ids[seek_mask]
        if seek_env_ids.numel() > 0:
            triggers, targets = self._sample_scheduled_stop_frames(seek_env_ids)
            self._stop_target_frame[seek_env_ids] = targets
            self._stop_trigger_frame[seek_env_ids] = triggers
            self._stop_hold_steps_remaining[seek_env_ids] = self._sample_duration_steps(
                self.cfg.stop_hold_time_range_s, len(seek_env_ids)
            )
            self._stop_training_seek_episode[seek_env_ids] = True

        self._fixed_start_episode[direct_env_ids] = False
        self._secondary_fixed_start_episode[direct_env_ids] = False
        self._uniform_start_episode[direct_env_ids] = False

    def request_zero_command(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Plan a safe stop at the next legal reference frame and hold indefinitely."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        targets = next_zero_command_frame(
            self.motion.zero_command_valid_mask, self.time_steps[env_ids]
        )
        if torch.any(targets < 0):
            failed_frames = self.time_steps[env_ids][targets < 0].tolist()
            raise RuntimeError(
                f"no future zero-command frame exists after reference frames {failed_frames}"
            )
        self._stop_target_frame[env_ids] = targets
        self._stop_trigger_frame[env_ids] = self.time_steps[env_ids]
        already_at_target = targets == self.time_steps[env_ids]
        self._stop_state[env_ids] = torch.where(
            already_at_target,
            torch.full_like(targets, STOP_STATE_HOLD),
            torch.full_like(targets, STOP_STATE_SEEK),
        )
        self._stop_hold_steps_remaining[env_ids] = -1

    def resume_climb_command(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Release a zero-command hold and continue the recorded motion direction."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._stop_state[env_ids] = STOP_STATE_RUN
        self._stop_target_frame[env_ids] = -1
        self._stop_trigger_frame[env_ids] = -1
        self._stop_hold_steps_remaining[env_ids] = 0

    def _reset_motion_progress(self, env_ids: torch.Tensor) -> None:
        self._motion_progress_confirmation_count[env_ids] = 0
        self._motion_progress_direction[env_ids] = self.motion.climb_direction[
            self.time_steps[env_ids]
        ]
        if self.cfg.motion_semantic_progress_enabled:
            self._motion_target_frame[env_ids] = (
                self.motion.initial_stable_motion_target_frame[self.time_steps[env_ids]]
            )
        else:
            self._motion_target_frame[env_ids] = -1

    def _reset_stable_anchor_gate(self, env_ids: torch.Tensor) -> None:
        self._stable_anchor_hold_steps[env_ids] = 0
        self._stable_anchor_confirmation_count[env_ids] = 0
        self._stable_anchor_timeout_failed[env_ids] = False
        self._stable_anchor_completion_event[env_ids] = False
        self._stable_anchor_completion_count[env_ids] = 0
        if self.cfg.stable_anchor_gating_enabled:
            self._stable_anchor_holding[env_ids] = (
                self.motion.stable_motion_anchor_mask[self.time_steps[env_ids]]
            )
        else:
            self._stable_anchor_holding[env_ids] = False

    @property
    def stable_anchor_timeout_failed(self) -> torch.Tensor:
        """Whether an environment exhausted its stable-anchor completion window."""

        return self._stable_anchor_timeout_failed

    @property
    def stable_anchor_completion_event(self) -> torch.Tensor:
        """One-tick event emitted when the unchanged gate releases an anchor."""

        return self._stable_anchor_completion_event

    def _stable_anchor_sensor_contacts(self, sensors: Sequence) -> torch.Tensor:
        contacts = []
        threshold = self.cfg.contact_force_threshold
        for sensor in sensors:
            force_matrix_w = sensor.data.force_matrix_w
            if force_matrix_w is None:
                raise RuntimeError(
                    "stable-anchor contact sensors must define filter_prim_paths_expr"
                )
            force = torch.linalg.vector_norm(force_matrix_w, dim=-1)
            contacts.append(force.reshape(self.num_envs, -1).amax(dim=-1) > threshold)
        return torch.stack(contacts, dim=-1)

    def _stable_anchor_actual_contacts(
        self, desired_support_rung_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_contact = self._stable_anchor_sensor_contacts(
            self._stable_anchor_limb_contact_sensors
        )
        if self._stable_anchor_ground_contact_sensors:
            ground_contact = self._stable_anchor_sensor_contacts(
                self._stable_anchor_ground_contact_sensors
            )
        else:
            ground_contact = torch.zeros_like(step_contact)
        wants_ground = desired_support_rung_id == 0
        wants_step = desired_support_rung_id > 0
        actual = torch.where(
            wants_ground,
            ground_contact,
            torch.where(wants_step, step_contact, torch.zeros_like(step_contact)),
        )
        wrong_surface = torch.any(
            (wants_ground & step_contact)
            | (wants_step & ground_contact)
            | ((~wants_ground & ~wants_step) & (step_contact | ground_contact)),
            dim=-1,
        )
        return actual, wrong_surface

    @property
    def actual_stable_support_contacts(self) -> torch.Tensor:
        """Physical contact with either a ladder rung or the ground surface."""

        step_contact = self._stable_anchor_sensor_contacts(
            self._stable_anchor_limb_contact_sensors
        )
        if not self._stable_anchor_ground_contact_sensors:
            return step_contact
        ground_contact = self._stable_anchor_sensor_contacts(
            self._stable_anchor_ground_contact_sensors
        )
        return step_contact | ground_contact

    def _stable_anchor_has_dangerous_contact(self) -> torch.Tensor:
        threshold = self.cfg.stable_anchor_danger_contact_force_threshold
        dangerous = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._stable_anchor_danger_body_indexes.numel() > 0:
            forces_w = self._contact_sensor.data.net_forces_w[
                :, self._stable_anchor_danger_body_indexes, :
            ]
            dangerous |= torch.linalg.vector_norm(forces_w, dim=-1).amax(dim=-1) > threshold
        for sensor in self._stable_anchor_rail_contact_sensors:
            force_matrix_w = sensor.data.force_matrix_w
            if force_matrix_w is None:
                raise RuntimeError(
                    "stable-anchor rail sensors must define filter_prim_paths_expr"
                )
            force = torch.linalg.vector_norm(force_matrix_w, dim=-1)
            dangerous |= force.reshape(self.num_envs, -1).amax(dim=-1) > threshold
        return dangerous

    def _stable_anchor_completion(self) -> tuple[torch.Tensor, ...]:
        desired = (
            self.motion.stable_support_contact_indicator[self.time_steps] > 0.5
        )
        desired_support_rung_id = self.motion.stable_support_rung_id[self.time_steps]
        actual, wrong_surface = self._stable_anchor_actual_contacts(
            desired_support_rung_id
        )
        near_target = (
            torch.linalg.vector_norm(
                self.robot_limb_pos_w
                - self._stable_support_goal_positions_for_frames(self.time_steps),
                dim=-1,
            )
            <= self._limb_target_distance_thresholds
        )
        # Use the same pair-filtered rung sensors for both the actual contact
        # bit and target completion.  In particular, CR1 foot geometry contacts
        # through ankle_x while target position is measured at a fixed sole
        # point; the generic net-force sensor must not be used as a second,
        # inconsistent foot-contact source.
        target_contact = desired & actual & near_target

        target_frame = self.time_steps
        predecessor = self.motion.stable_motion_anchor_predecessor[target_frame]
        predecessor_desired = (
            self.motion.stable_support_contact_indicator[predecessor] > 0.5
        )
        predecessor_rung = self.motion.stable_support_rung_id[predecessor]
        target_rung = self.motion.stable_support_rung_id[target_frame]
        newly_required = desired & (
            (~predecessor_desired) | (target_rung != predecessor_rung)
        )
        (
            strict_contact_complete,
            relaxed_contact_complete,
            contact_match_fraction,
            target_contact_fraction,
        ) = stable_anchor_contact_completion(
            desired, actual, target_contact, newly_required
        )

        height_error = torch.abs(self.robot_anchor_pos_w[:, 2] - self.anchor_pos_w[:, 2])
        orientation_error = quat_error_magnitude(
            self.robot_anchor_quat_w, self.anchor_quat_w
        )
        linear_speed = torch.linalg.vector_norm(self.robot_anchor_lin_vel_w, dim=-1)
        angular_speed = torch.linalg.vector_norm(self.robot_anchor_ang_vel_w, dim=-1)
        safe = (
            (height_error <= self.cfg.stable_anchor_root_height_error_threshold)
            & (orientation_error <= self.cfg.stable_anchor_root_orientation_threshold)
            & (linear_speed <= self.cfg.stable_anchor_root_linear_speed_threshold)
            & (angular_speed <= self.cfg.stable_anchor_root_angular_speed_threshold)
            & ~wrong_surface
            & ~self._stable_anchor_has_dangerous_contact()
        )
        return (
            strict_contact_complete & safe,
            relaxed_contact_complete & safe,
            safe,
            contact_match_fraction,
            target_contact_fraction,
        )

    def _advance_reference_with_stable_anchor_gate(self) -> None:
        holding = self._stable_anchor_holding
        strict, relaxed, safe, contact_match, target_contact = (
            self._stable_anchor_completion()
        )
        relaxed_after_steps = round(
            self.cfg.stable_anchor_relaxed_after_s * self._control_fps
        )
        failure_after_steps = round(
            self.cfg.stable_anchor_failure_after_s * self._control_fps
        )
        (
            self._stable_anchor_hold_steps,
            self._stable_anchor_confirmation_count,
            release,
            timed_out,
        ) = stable_anchor_gate_transition(
            holding,
            self._stable_anchor_hold_steps,
            self._stable_anchor_confirmation_count,
            strict,
            relaxed,
            relaxed_after_steps=relaxed_after_steps,
            failure_after_steps=failure_after_steps,
            confirmation_steps=self.cfg.stable_anchor_confirmation_steps,
        )
        self._stable_anchor_completion_event.copy_(release)
        self._stable_anchor_completion_count.add_(release.long())
        self._stable_anchor_timeout_failed |= timed_out

        can_advance = ((~holding) | release) & ~self._stable_anchor_timeout_failed
        next_steps = torch.where(can_advance, self.time_steps + 1, self.time_steps)
        next_steps.clamp_(max=self.motion.time_step_total - 1)
        arrived = can_advance & self.motion.stable_motion_anchor_mask[next_steps]
        self.time_steps.copy_(next_steps)
        self._stable_anchor_holding = (holding & ~release) | arrived
        reset_counters = release | arrived
        self._stable_anchor_hold_steps[reset_counters] = 0
        self._stable_anchor_confirmation_count[reset_counters] = 0

        self.metrics["stable_anchor_holding"] = self._stable_anchor_holding.float()
        self.metrics["stable_anchor_hold_time_s"] = (
            self._stable_anchor_hold_steps.float() / self._control_fps
        )
        self.metrics["stable_anchor_strict_complete"] = strict.float()
        self.metrics["stable_anchor_relaxed_complete"] = relaxed.float()
        self.metrics["stable_anchor_safe"] = safe.float()
        self.metrics["stable_anchor_contact_match_fraction"] = contact_match
        self.metrics["stable_anchor_target_contact_fraction"] = target_contact
        self.metrics["stable_anchor_completion_event"] = release.float()
        self.metrics["stable_anchor_completion_count"] = (
            self._stable_anchor_completion_count.float()
        )
        self.metrics["stable_anchor_timeout_failed"] = (
            self._stable_anchor_timeout_failed.float()
        )

    def _update_motion_progress(self) -> None:
        """Advance stable-pose supervision only after observable completion."""

        if not self.cfg.motion_semantic_progress_enabled:
            return
        direction = self.motion.climb_direction[self.time_steps]
        direction_changed = direction != self._motion_progress_direction
        if torch.any(direction_changed):
            self._motion_target_frame[direction_changed] = (
                self.motion.initial_stable_motion_target_frame[
                    self.time_steps[direction_changed]
                ]
            )
            self._motion_progress_confirmation_count[direction_changed] = 0
            self._motion_progress_direction[direction_changed] = direction[direction_changed]

        valid = self.motion_semantic_target_valid
        joint_error = self.robot_joint_pos - self.motion_semantic_target_joint_pos
        joint_rms_error = torch.sqrt(torch.mean(joint_error.square(), dim=-1))
        orientation_error = quat_error_magnitude(
            self.robot_anchor_quat_w, self.motion_semantic_target_root_quat_w
        )
        anchor_linear_speed = torch.linalg.vector_norm(
            self.robot_anchor_lin_vel_w, dim=-1
        )
        anchor_angular_speed = torch.linalg.vector_norm(
            self.robot_anchor_ang_vel_w, dim=-1
        )
        joint_speed_rms = torch.sqrt(
            torch.mean(self.robot_joint_vel.square(), dim=-1)
        )

        target_contact_forces_w = self._contact_sensor.data.net_forces_w[
            :, self._contact_limb_body_indexes, :
        ]
        required, _, target_contact = classify_limb_contacts(
            self.motion_semantic_target_contact_indicator,
            self.robot_limb_pos_w,
            self.motion_semantic_target_contact_pos_w,
            target_contact_forces_w,
            self._limb_target_distance_thresholds,
            self.cfg.contact_force_threshold,
        )
        required_count = required.sum(dim=-1)
        target_contact_count = target_contact.sum(dim=-1)
        contact_complete = torch.all((~required) | target_contact, dim=-1)
        target_contact_fraction = torch.where(
            required_count > 0,
            target_contact_count.float() / required_count.clamp_min(1).float(),
            torch.ones_like(joint_rms_error),
        )

        completed = (
            valid
            & (joint_rms_error <= self.cfg.motion_progress_joint_rms_threshold)
            & (orientation_error <= self.cfg.motion_progress_orientation_threshold)
            & (
                anchor_linear_speed
                <= self.cfg.motion_progress_anchor_linear_speed_threshold
            )
            & (
                anchor_angular_speed
                <= self.cfg.motion_progress_anchor_angular_speed_threshold
            )
            & (
                joint_speed_rms
                <= self.cfg.motion_progress_joint_speed_rms_threshold
            )
            & contact_complete
        )
        self._motion_progress_confirmation_count = torch.where(
            completed,
            self._motion_progress_confirmation_count + 1,
            torch.zeros_like(self._motion_progress_confirmation_count),
        )
        confirmed = (
            self._motion_progress_confirmation_count
            >= self.cfg.motion_progress_confirmation_steps
        )
        current_target = self._motion_target_frame.clamp_min(0)
        next_target = self.motion.stable_motion_target_successor[current_target]
        advance = confirmed & valid & (next_target != self._motion_target_frame)
        self._motion_target_frame = torch.where(
            advance, next_target, self._motion_target_frame
        )
        self._motion_progress_confirmation_count[advance] = 0

        self.metrics["motion_semantic_target_frame_fraction"] = (
            self._motion_target_frame.clamp_min(0).float()
            / max(self.motion.time_step_total - 1, 1)
        )
        self.metrics["motion_semantic_joint_rms_error"] = joint_rms_error
        self.metrics["motion_semantic_orientation_error"] = orientation_error
        self.metrics["motion_semantic_target_contact_fraction"] = (
            target_contact_fraction
        )
        self.metrics["motion_semantic_completed"] = completed.float()

    def _advance_reference(self) -> None:
        if self.cfg.stable_anchor_gating_enabled:
            self._advance_reference_with_stable_anchor_gate()
            return
        finite_hold = self.stop_holding & (self._stop_hold_steps_remaining > 0)
        self._stop_hold_steps_remaining[finite_hold] -= 1
        resume = finite_hold & (self._stop_hold_steps_remaining == 0)
        if torch.any(resume):
            resume_ids = torch.nonzero(resume, as_tuple=False).squeeze(-1)
            self.resume_climb_command(resume_ids)

        scheduled_trigger = (
            (self._stop_state == STOP_STATE_RUN)
            & (self._stop_target_frame >= 0)
            & (self.time_steps >= self._stop_trigger_frame)
        )
        self._stop_state[scheduled_trigger] = STOP_STATE_SEEK

        advance = self._stop_state != STOP_STATE_HOLD
        next_steps = torch.where(advance, self.time_steps + 1, self.time_steps)
        next_steps.clamp_(max=self.motion.time_step_total - 1)
        seeking = self._stop_state == STOP_STATE_SEEK
        reached_target = seeking & (next_steps >= self._stop_target_frame)
        next_steps = torch.where(reached_target, self._stop_target_frame, next_steps)
        self.time_steps.copy_(next_steps)
        self._stop_state[reached_target] = STOP_STATE_HOLD

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        for limb_name in self._force_limit_sample_count:
            self._force_limit_sample_count[limb_name][env_ids] = 0.0
            self._force_limit_over_soft_count[limb_name][env_ids] = 0.0
            self._force_limit_previous_hard[limb_name][env_ids] = False
        for histogram in self._force_histograms.values():
            histogram[env_ids] = 0.0
        self._sample_ladder_geometry(env_ids)
        self._adaptive_sampling(env_ids)
        self._sample_stop_command_training(env_ids)
        self._reset_motion_progress(env_ids)
        self._reset_stable_anchor_gate(env_ids)

        self._episode_start_time_steps[env_ids] = self.time_steps[env_ids]
        self._tracking_path_initialized[env_ids] = False
        self._tracking_episode_steps[env_ids] = 0.0
        self._tracking_reference_motion_steps[env_ids] = 0.0
        self._tracking_stationary_motion_steps[env_ids] = 0.0

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        pose_lower, pose_upper, fixed_start_pose_randomized = pose_range_bounds(
            self.cfg.pose_range,
            len(env_ids),
            self.device,
            fixed_start_mask=self._fixed_start_episode[env_ids],
            fixed_start_pose_range=self.cfg.fixed_start_pose_range,
            fixed_start_pose_randomization_probability=(
                self.cfg.fixed_start_pose_randomization_probability
            ),
        )
        self._fixed_start_pose_randomized_episode[env_ids] = (
            fixed_start_pose_randomized
        )
        rand_samples = sample_uniform(
            pose_lower, pose_upper, (len(env_ids), 6), device=self.device
        )
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

        # Terminations and rewards run before the first post-reset command update.
        # Seed valid targets here so the first step never compares the robot against zero-filled buffers.
        self.body_pos_relative_w[env_ids] = self.body_pos_w[env_ids]
        self.body_quat_relative_w[env_ids] = self.body_quat_w[env_ids]
        self.metrics["reset_valid_fraction"][env_ids] = self._current_reset_valid_fraction[env_ids]

    def _update_command(self):
        if self.cfg.advance_reference:
            self._advance_reference()
        self._update_motion_progress()

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )
                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                self.rung_visualizer = VisualizationMarkers(self.cfg.rung_visualizer_cfg)
                self.contact_target_visualizer = VisualizationMarkers(
                    self.cfg.contact_target_visualizer_cfg
                )
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )
            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            self.rung_visualizer.set_visibility(True)
            self.contact_target_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)
        elif hasattr(self, "current_anchor_visualizer"):
            self.current_anchor_visualizer.set_visibility(False)
            self.goal_anchor_visualizer.set_visibility(False)
            self.rung_visualizer.set_visibility(False)
            self.contact_target_visualizer.set_visibility(False)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(False)
                self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)
        self.rung_visualizer.visualize(self.rung_positions_w.reshape(-1, 3))
        self.contact_target_visualizer.visualize(
            self.target_contact_pos_w.reshape(-1, 3),
            marker_indices=(self.contact_indicator <= 0.5).long().reshape(-1),
        )
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class LadderMotionCommandCfg(CommandTermCfg):
    class_type: type = LadderMotionCommand

    asset_name: str = MISSING
    motion_file: str = "__MISSING_LADDER_MOTION_FILE__.npz"
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    ladder_geometry: LadderGeometry = LADDER_A
    ladder_asset_name: str | None = None
    ladder_inclination_delta_range: tuple[float, float] = (0.0, 0.0)
    ladder_spacing_delta_range: tuple[float, float] = (0.0, 0.0)
    nominal_ladder_geometry_probability: float = 0.0
    use_geometry_conditioned_reset_masks: bool = True
    use_reset_valid_mask: bool = True
    ladder_fixed_box_names: tuple[str, ...] = ()
    ladder_rung_box_names: tuple[str, ...] = ()
    ladder_inclined_box_names: tuple[str, ...] = ()
    ladder_relative_observation_frame: str = "world"
    limb_names: tuple[str, ...] = LIMB_NAMES
    limb_body_names: dict[str, str] = DEFAULT_LIMB_BODY_NAMES
    limb_target_offsets: dict[str, tuple[float, float, float]] | None = None
    contact_sensor_name: str = "contact_forces"
    contact_force_threshold: float = 5.0
    contact_front_normal_fraction: float = 0.5
    limb_target_distance_thresholds: dict[str, float] | None = None
    contact_force_soft_limits: dict[str, float] = {}
    contact_force_hard_limits: dict[str, float] = {}
    fixed_reference_frame: int | None = None
    advance_reference: bool = True
    stop_command_training_enabled: bool = False
    stop_hold_reset_probability: float = 0.0
    stop_seek_episode_probability: float = 0.0
    stop_hold_time_range_s: tuple[float, float] = (1.0, 3.0)
    motion_semantic_progress_enabled: bool = False
    motion_progress_joint_rms_threshold: float = 0.20
    motion_progress_orientation_threshold: float = 0.25
    motion_progress_anchor_linear_speed_threshold: float = 0.10
    motion_progress_anchor_angular_speed_threshold: float = 0.50
    motion_progress_joint_speed_rms_threshold: float = 1.0
    motion_progress_confirmation_steps: int = 5
    stable_anchor_gating_enabled: bool = False
    stable_anchor_confirmation_steps: int = 4
    stable_anchor_relaxed_after_s: float = 2.0
    stable_anchor_failure_after_s: float = 5.5
    stable_anchor_root_height_error_threshold: float = 0.35
    stable_anchor_root_orientation_threshold: float = 0.70
    stable_anchor_root_linear_speed_threshold: float = 0.50
    stable_anchor_root_angular_speed_threshold: float = 1.50
    stable_anchor_danger_contact_force_threshold: float = 5.0
    stable_anchor_limb_contact_sensor_names: tuple[str, ...] = ()
    stable_anchor_ground_contact_sensor_names: tuple[str, ...] = ()
    stable_anchor_rail_contact_sensor_names: tuple[str, ...] = ()
    reset_by_stable_support_segments: bool = False
    fixed_start_frame: int = 0
    fixed_start_probability: float = 0.0
    secondary_fixed_start_frame: int | None = None
    secondary_fixed_start_probability: float = 0.0
    secondary_fixed_start_requires_nominal_ladder: bool = False
    uniform_start_probability: float = 0.0

    pose_range: dict[str, tuple[float, float]] = {}
    fixed_start_pose_range: dict[str, tuple[float, float]] = {}
    fixed_start_pose_randomization_probability: float = 1.0
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    rung_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/ladder_rungs",
        markers={
            "rung": sim_utils.SphereCfg(
                radius=0.018,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 1.0)),
            )
        },
    )
    contact_target_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/contact_targets",
        markers={
            "required": sim_utils.SphereCfg(
                radius=0.022,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.2)),
            ),
            "swing": sim_utils.SphereCfg(
                radius=0.014,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
            ),
        },
    )
