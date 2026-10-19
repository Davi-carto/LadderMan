"""Replay an annotated CR1 ladder reference motion with contact target markers."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from ladder.utils.ladder_presets import HOLOSOMA_CR1_WOOD_STEP
from ladder.utils.contact_schema import CR1_INITIAL_SOLE_TARGET_Z

DEFAULT_MOTION = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "restored_prefix_v300_stride2_slow50"
    / "smooth_start"
    / "climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_train_ready.npz"
)
DEFAULT_ROBOT = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)
DEFAULT_LADDER = REPO_ROOT / "assets" / "ladders" / "ladder_A_aframe" / "wood_step_ladder_physics.usda"


parser = argparse.ArgumentParser(description="Replay a Ladder motion npz and visualize CR1 contact targets.")
parser.add_argument("--motion_file", type=Path, default=DEFAULT_MOTION, help="Annotated Ladder motion npz.")
parser.add_argument("--robot_asset", type=Path, default=DEFAULT_ROBOT, help="CR1 25DOF URDF.")
parser.add_argument("--ladder_asset", type=Path, default=DEFAULT_LADDER, help="USD/USDA ladder asset to show.")
parser.add_argument("--ladder_x", type=float, default=None, help="Override displayed ladder X translation.")
parser.add_argument("--ladder_y", type=float, default=None, help="Override displayed ladder Y translation.")
parser.add_argument("--ladder_z", type=float, default=None, help="Override displayed ladder Z translation.")
parser.add_argument("--ladder_yaw_deg", type=float, default=None, help="Override displayed ladder yaw in degrees.")
parser.add_argument("--skip_ladder_asset", action="store_true", help="Do not spawn the ladder USD asset.")
parser.add_argument(
    "--skip_markers",
    action="store_true",
    help="Do not create or update contact markers; permits replaying an unannotated raw motion.",
)
parser.add_argument("--show_rung_centers", action="store_true", help="Show every geometry-derived rung center.")
parser.add_argument("--follow_camera", action="store_true", help="Move the camera with the replayed robot root.")
parser.add_argument("--skip_camera_update", action="store_true", help="Deprecated alias; camera updates are off by default.")
parser.add_argument(
    "--playback_fps", type=float, default=None, help="Viewer playback fps; defaults to the motion fps."
)
parser.add_argument(
    "--viewer_fps",
    type=float,
    default=60.0,
    help="GUI/render refresh rate, independent of motion playback fps (default: 60).",
)
parser.add_argument("--start_frame", type=int, default=0, help="First motion frame to display.")
parser.add_argument(
    "--end_frame",
    type=int,
    default=None,
    help="Inclusive final motion frame to display; defaults to the final file frame.",
)
parser.add_argument("--hold_frame", action="store_true", help="Keep displaying start_frame for close inspection.")
parser.add_argument(
    "--stable_frames_only",
    action="store_true",
    help="Replay only ground-aware stable-support anchors (one frame per four-limb support state).",
)
parser.add_argument("--smoke_steps", type=int, default=0, help="Run this many frames and exit; 0 keeps replaying.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import carb
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb.ladder_geometry import generate_ladder, limb_lateral_offsets_tensor, rung_positions_tensor

if not args_cli.headless:
    import omni.appwindow
    import omni.ui as ui


class ReplayController:
    """Interactive playback state and GUI controls for reference inspection."""

    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        motion_fps: float,
        paused: bool,
        enabled: bool,
        frame_sequence: list[int] | None = None,
    ):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_sequence = frame_sequence
        self._sequence_index = 0
        self.frame = frame_sequence[0] if frame_sequence else start_frame
        self.motion_fps = motion_fps
        self.paused = paused
        self.diagnostics_requested = False
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._window = None
        self._status_label = None
        self._frame_model = None
        self._updating_frame_model = False

        if enabled:
            self._create_ui()
            app_window = omni.appwindow.get_default_app_window()
            if app_window is not None:
                self._input = carb.input.acquire_input_interface()
                self._keyboard = app_window.get_keyboard()
                self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                    self._keyboard, self._on_keyboard_event
                )

    def close(self) -> None:
        if self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        if self._window is not None:
            self._window.visible = False
            self._window.destroy()
            self._window = None

    def advance(self) -> None:
        if not self.paused:
            if self.frame_sequence:
                self._sequence_index = (self._sequence_index + 1) % len(self.frame_sequence)
                self.frame = self.frame_sequence[self._sequence_index]
            else:
                self.frame = self.start_frame if self.frame >= self.end_frame else self.frame + 1

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.update_status()

    def seek(self, frame: int) -> None:
        frame = max(self.start_frame, min(self.end_frame, int(frame)))
        if self.frame_sequence:
            self._sequence_index = min(
                range(len(self.frame_sequence)),
                key=lambda index: abs(self.frame_sequence[index] - frame),
            )
            self.frame = self.frame_sequence[self._sequence_index]
        else:
            self.frame = frame
        self.paused = True
        self.update_status()

    def step(self, delta: int) -> None:
        if self.frame_sequence:
            self._sequence_index = max(
                0, min(len(self.frame_sequence) - 1, self._sequence_index + delta)
            )
            self.frame = self.frame_sequence[self._sequence_index]
            self.paused = True
            self.update_status()
        else:
            self.seek(self.frame + delta)

    def request_diagnostics(self) -> None:
        self.diagnostics_requested = True

    def consume_diagnostics_request(self) -> bool:
        requested = self.diagnostics_requested
        self.diagnostics_requested = False
        return requested

    def update_status(self) -> str:
        state = "PAUSED" if self.paused else "PLAYING"
        status = (
            f"Frame {self.frame} | Range [{self.start_frame}, {self.end_frame}] | "
            f"Reference time {self.frame / self.motion_fps:.2f} s | {state}"
        )
        if self.frame_sequence:
            status += (
                f" | Stable anchor {self._sequence_index + 1}/"
                f"{len(self.frame_sequence)}"
            )
        if self._status_label is not None:
            self._status_label.text = status
        if self._frame_model is not None:
            self._updating_frame_model = True
            self._frame_model.set_value(self.frame)
            self._updating_frame_model = False
        return status

    def _create_ui(self) -> None:
        self._window = ui.Window(
            "Reference Replay",
            width=560,
            height=125,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                self._status_label = ui.Label("", height=24)
                slider = ui.IntSlider(min=self.start_frame, max=self.end_frame, height=20)
                self._frame_model = slider.model
                self._frame_model.add_value_changed_fn(self._on_frame_model_changed)
                with ui.HStack(spacing=4, height=28):
                    ui.Button("|<", clicked_fn=lambda: self.seek(self.start_frame), tooltip="First range frame")
                    ui.Button("-50", clicked_fn=lambda: self.step(-50), tooltip="Back 50 frames")
                    ui.Button("<", clicked_fn=lambda: self.step(-1), tooltip="Previous frame")
                    ui.Button("Play/Pause", clicked_fn=self.toggle_pause)
                    ui.Button(">", clicked_fn=lambda: self.step(1), tooltip="Next frame")
                    ui.Button("+50", clicked_fn=lambda: self.step(50), tooltip="Forward 50 frames")
                    ui.Button(">|", clicked_fn=lambda: self.seek(self.end_frame), tooltip="Final range frame")
                    ui.Button("Info", clicked_fn=self.request_diagnostics, tooltip="Print contact diagnostics")
        self.update_status()

    def _on_frame_model_changed(self, model) -> None:
        if not self._updating_frame_model and model.as_int != self.frame:
            self.seek(model.as_int)

    def _on_keyboard_event(self, event, *args) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = event.input.name
        if key == "SPACE":
            self.toggle_pause()
        elif key == "LEFT":
            self.step(-1)
        elif key == "RIGHT":
            self.step(1)
        elif key == "PAGE_DOWN":
            self.step(-50)
        elif key == "PAGE_UP":
            self.step(50)
        elif key == "HOME":
            self.seek(self.start_frame)
        elif key == "END":
            self.seek(self.end_frame)
        elif key == "I":
            self.request_diagnostics()
        return True


def _check_file(path: Path, suffixes: tuple[str, ...]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")
    if resolved.suffix.lower() not in suffixes:
        raise ValueError(f"Expected one of {suffixes}, got: {resolved}")
    return resolved


def _as_string_list(values: np.ndarray) -> list[str]:
    items = np.asarray(values).reshape(-1).tolist()
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items]


def _yaw_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * math.radians(yaw_deg)
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def _load_motion(path: Path, *, require_contact_annotations: bool = True) -> dict[str, np.ndarray | list[str]]:
    required = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "body_names",
    )
    if require_contact_annotations:
        required += ("contact_indicator", "contact_rung_id", "limb_names", "limb_body_names")
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{path} is missing required keys: {missing}")
        motion = {key: data[key] for key in data.files}
    motion["joint_names"] = _as_string_list(motion["joint_names"])
    motion["body_names"] = _as_string_list(motion["body_names"])
    if "limb_names" in motion:
        motion["limb_names"] = _as_string_list(motion["limb_names"])
    if "limb_body_names" in motion:
        motion["limb_body_names"] = _as_string_list(motion["limb_body_names"])
    return motion


def make_robot_cfg(robot_asset: Path) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(robot_asset),
            fix_base=False,
            merge_fixed_joints=False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=400.0,
                velocity_limit_sim=50.0,
                stiffness=80.0,
                damping=4.0,
            )
        },
    )


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, CR1_INITIAL_SOLE_TARGET_Z)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot: ArticulationCfg = make_robot_cfg(_check_file(args_cli.robot_asset, (".urdf",)))


LIMB_COLORS = {
    "left_hand": (0.0, 0.85, 1.0),
    "right_hand": (0.2, 0.35, 1.0),
    "left_foot": (0.15, 1.0, 0.25),
    "right_foot": (1.0, 0.75, 0.05),
}


def _make_limb_marker(limb_names: list[str]) -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/LadderReference/limbs",
        markers={
            limb_name: sim_utils.SphereCfg(
                radius=0.055,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=LIMB_COLORS.get(limb_name, (0.1, 0.4, 1.0))
                ),
            )
            for limb_name in limb_names
        },
    )
    return VisualizationMarkers(cfg)


def _make_target_marker() -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/LadderReference/targets",
        markers={
            "inactive": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
            ),
            "phase_only": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.05)),
            ),
            "required": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            ),
            "support_only": sim_utils.SphereCfg(
                radius=0.055,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.15, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _make_rung_center_marker() -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/LadderReference/rung_centers",
            markers={
                "center": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                )
            },
        )
    )


def _motion_ladder_geometry(motion: dict[str, np.ndarray | list[str]]):
    params = np.asarray(motion["contact_annotation_ladder_params"], dtype=np.float32)
    origin = np.asarray(motion["contact_annotation_ladder_origin"], dtype=np.float32)
    num_rungs = int(np.asarray(motion.get("contact_annotation_ladder_num_rungs", [12])).reshape(-1)[0])
    return generate_ladder(
        phi=float(params[0]),
        rung_spacing=float(params[1]),
        rung_length=float(params[2]),
        rung_radius_or_width=float(params[3]),
        num_rungs=num_rungs,
        origin=tuple(float(x) for x in origin),
    )


def _target_contacts(motion: dict[str, np.ndarray | list[str]], device: str) -> torch.Tensor:
    geometry = _motion_ladder_geometry(motion)
    rung_positions = rung_positions_tensor(geometry, device=device)
    if "contact_annotation_limb_target_offsets" in motion:
        offsets = torch.tensor(
            np.asarray(motion["contact_annotation_limb_target_offsets"]), dtype=torch.float32, device=device
        )
    else:
        offsets = limb_lateral_offsets_tensor(geometry, motion["limb_names"], device=device)
    support_rung_ids = motion.get("stable_support_rung_id")
    if support_rung_ids is not None:
        support_rung_ids = np.asarray(support_rung_ids, dtype=np.int64)
        rung_ids_array = support_rung_ids - 1
    else:
        rung_ids_array = np.asarray(motion["contact_rung_id"], dtype=np.int64)
    if "contact_phase_indicator" in motion and "contact_phase_rung_id" in motion:
        support = np.asarray(motion["contact_indicator"]) > 0.5
        phase = np.asarray(motion["contact_phase_indicator"]) > 0.5
        phase_rung_ids = np.asarray(motion["contact_phase_rung_id"], dtype=np.int64)
        rung_ids_array = np.where(~support & phase, phase_rung_ids, rung_ids_array)
    rung_ids = torch.tensor(rung_ids_array, device=device)
    rung_ids = torch.clamp(rung_ids, 0, geometry.num_rungs - 1)
    targets = rung_positions[rung_ids] + offsets.unsqueeze(0)
    if support_rung_ids is not None:
        body_names = motion["body_names"]
        limb_body_indexes = np.asarray(
            motion.get(
                "limb_body_indexes",
                [body_names.index(name) for name in motion["limb_body_names"]],
            ),
            dtype=np.int64,
        )
        ground_targets = torch.tensor(
            np.asarray(motion["body_pos_w"], dtype=np.float32)[:, limb_body_indexes],
            dtype=torch.float32,
            device=device,
        )
        targets = torch.where(
            torch.tensor(support_rung_ids == 0, device=device).unsqueeze(-1),
            ground_targets,
            targets,
        )
    return targets


def _ladder_asset_pose(motion: dict[str, np.ndarray | list[str]]) -> tuple[tuple[float, float, float], float]:
    default_translation = HOLOSOMA_CR1_WOOD_STEP.asset_translation
    default_yaw = HOLOSOMA_CR1_WOOD_STEP.asset_yaw_deg
    translation = np.asarray(
        motion.get("contact_annotation_ladder_asset_translation", default_translation), dtype=np.float32
    ).reshape(3)
    yaw_deg = float(
        np.asarray(motion.get("contact_annotation_ladder_asset_yaw_deg", [default_yaw])).reshape(-1)[0]
    )
    overrides = (args_cli.ladder_x, args_cli.ladder_y, args_cli.ladder_z)
    translation = tuple(
        float(value if override is None else override) for value, override in zip(translation, overrides, strict=True)
    )
    return translation, yaw_deg if args_cli.ladder_yaw_deg is None else args_cli.ladder_yaw_deg


def _fallback_contact_nearest_distances(motion: dict[str, np.ndarray | list[str]]) -> np.ndarray:
    """Recompute diagnostic nearest-target distances for repaired motions.

    Reference repair tools intentionally remove stale derived contact fields. The
    replay viewer can still inspect those motions by deriving the same nearest
    rung distance from the saved body trajectory and annotation geometry.
    """

    geometry = _motion_ladder_geometry(motion)
    rung_positions = rung_positions_tensor(geometry, device="cpu").numpy()
    if "contact_annotation_limb_target_offsets" in motion:
        offsets = np.asarray(motion["contact_annotation_limb_target_offsets"], dtype=np.float32)
    else:
        offsets = limb_lateral_offsets_tensor(geometry, motion["limb_names"], device="cpu").numpy()
    body_names = motion["body_names"]
    limb_body_indexes = np.asarray(
        motion.get(
            "limb_body_indexes",
            [body_names.index(name) for name in motion["limb_body_names"]],
        ),
        dtype=np.int64,
    )
    limb_positions = np.asarray(motion["body_pos_w"], dtype=np.float32)[:, limb_body_indexes]
    target_points = rung_positions[None, :, :] + offsets[:, None, :]
    return np.linalg.norm(
        limb_positions[:, :, None, :] - target_points[None, :, :, :], axis=-1
    ).min(axis=-1)


def _print_frame_diagnostics(
    motion: dict[str, np.ndarray | list[str]], frame: int, target_contacts: torch.Tensor
) -> None:
    print(f"[FRAME {frame}] contact diagnostics", flush=True)
    if "climb_direction" in motion:
        direction = float(np.asarray(motion["climb_direction"])[frame].reshape(-1)[0])
        zero_valid = (
            bool(np.asarray(motion["zero_command_valid_mask"])[frame])
            if "zero_command_valid_mask" in motion
            else False
        )
        stop_keyframe = (
            bool(np.asarray(motion["zero_command_segment_min_base_speed_mask"])[frame])
            if "zero_command_segment_min_base_speed_mask" in motion
            else (
                bool(np.asarray(motion["zero_command_natural_contact_stable_mask"])[frame])
                if "zero_command_natural_contact_stable_mask" in motion
                else False
            )
        )
        ground_default = (
            bool(np.asarray(motion["zero_command_ground_default_standing_mask"])[frame])
            if "zero_command_ground_default_standing_mask" in motion
            else False
        )
        top_default = (
            bool(np.asarray(motion["zero_command_top_default_standing_mask"])[frame])
            if "zero_command_top_default_standing_mask" in motion
            else False
        )
        terminal_default = (
            bool(np.asarray(motion["zero_command_terminal_default_standing_mask"])[frame])
            if "zero_command_terminal_default_standing_mask" in motion
            else False
        )
        print(
            f"  command: climb_direction={direction:+.0f} zero_valid={int(zero_valid)} "
            f"four_limb_stop_keyframe={int(stop_keyframe)} "
            f"ground_default={int(ground_default)} top_default={int(top_default)} "
            f"terminal_default={int(terminal_default)}",
            flush=True,
        )
    nearest_distances = (
        np.asarray(motion["contact_nearest_distance"])
        if "contact_nearest_distance" in motion
        else _fallback_contact_nearest_distances(motion)
    )
    contact_speed = (
        np.asarray(motion["contact_speed"])
        if "contact_speed" in motion
        else np.zeros_like(nearest_distances)
    )
    raw_indicator = motion.get("contact_raw_indicator")
    hysteresis_indicator = motion.get("contact_hysteresis_filled_indicator")
    phase_indicator = motion.get("contact_phase_indicator")
    phase_rung_ids = motion.get("contact_phase_rung_id")
    surface_distances = motion.get("contact_surface_nearest_distance")
    surface_clearances = motion.get("contact_surface_nearest_clearance")
    stable_support_surface_distances = motion.get(
        "stable_support_surface_distance"
    )
    for limb_index, limb_name in enumerate(motion["limb_names"]):
        distance = float(nearest_distances[frame, limb_index])
        speed = float(contact_speed[frame, limb_index])
        raw = (
            str(int(np.asarray(raw_indicator)[frame, limb_index] > 0.5))
            if raw_indicator is not None
            else "n/a"
        )
        hysteresis = (
            str(int(np.asarray(hysteresis_indicator)[frame, limb_index] > 0.5))
            if hysteresis_indicator is not None
            else "n/a"
        )
        final = int(np.asarray(motion["contact_indicator"])[frame, limb_index] > 0.5)
        rung_id = int(np.asarray(motion["contact_rung_id"])[frame, limb_index])
        stable_support = (
            int(
                np.asarray(motion["stable_support_contact_indicator"])[
                    frame, limb_index
                ]
                > 0.5
            )
            if "stable_support_contact_indicator" in motion
            else final
        )
        stable_support_rung = (
            int(np.asarray(motion["stable_support_rung_id"])[frame, limb_index])
            if "stable_support_rung_id" in motion
            else (rung_id + 1 if final else -1)
        )
        phase = (
            int(np.asarray(phase_indicator)[frame, limb_index] > 0.5)
            if phase_indicator is not None
            else final
        )
        phase_rung = (
            int(np.asarray(phase_rung_ids)[frame, limb_index])
            if phase_rung_ids is not None
            else rung_id
        )
        surface = (
            f" surface_distance={float(np.asarray(surface_distances)[frame, limb_index]):.4f} m"
            if surface_distances is not None
            else ""
        )
        clearance = (
            f" clearance={float(np.asarray(surface_clearances)[frame, limb_index]):+.4f} m"
            if surface_clearances is not None
            else ""
        )
        support_surface = (
            " support_surface_distance="
            f"{float(np.asarray(stable_support_surface_distances)[frame, limb_index]):.4f} m"
            if stable_support_surface_distances is not None
            else ""
        )
        target = target_contacts[frame, limb_index].detach().cpu().numpy()
        print(
            f"  {limb_name:>10}: distance={distance:.4f} m speed={speed:.3f} m/s "
            f"raw={raw} hysteresis={hysteresis} support={final} rung={rung_id} "
            f"stable_support={stable_support} support_id={stable_support_rung} "
            f"surface_phase={phase} phase_rung={phase_rung}{surface}{clearance} "
            f"{support_surface} target={np.round(target, 3).tolist()}",
            flush=True,
        )


def _motion_tensors(motion: dict[str, np.ndarray | list[str]], device: str) -> dict[str, torch.Tensor]:
    keys = [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]
    if "contact_indicator" in motion:
        keys.append("contact_indicator")
    if "stable_support_contact_indicator" in motion:
        keys.append("stable_support_contact_indicator")
    if "contact_phase_indicator" in motion:
        keys.append("contact_phase_indicator")
    return {key: torch.tensor(np.asarray(motion[key]), dtype=torch.float32, device=device) for key in keys}


def _write_frame(robot, tensors: dict[str, torch.Tensor], frame: int, joint_ids: list[int]) -> torch.Tensor:
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = tensors["body_pos_w"][frame : frame + 1, 0]
    root_state[:, 3:7] = tensors["body_quat_w"][frame : frame + 1, 0]
    root_state[:, 7:10] = tensors["body_lin_vel_w"][frame : frame + 1, 0]
    root_state[:, 10:13] = tensors["body_ang_vel_w"][frame : frame + 1, 0]
    robot.write_root_state_to_sim(root_state)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_ids] = tensors["joint_pos"][frame : frame + 1]
    joint_vel[:, joint_ids] = tensors["joint_vel"][frame : frame + 1]
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    return root_state[0, :3]


def run_simulator(sim: SimulationContext, scene: InteractiveScene, motion: dict[str, np.ndarray | list[str]]) -> None:
    if args_cli.skip_ladder_asset:
        print("[INFO] Skipping displayed ladder asset.", flush=True)
    else:
        ladder_asset = _check_file(args_cli.ladder_asset, (".usd", ".usda"))
        ladder_translation, ladder_yaw_deg = _ladder_asset_pose(motion)
        print(f"[INFO] Spawning ladder asset: {ladder_asset}", flush=True)
        print(f"[INFO] Ladder pose: translation={ladder_translation}, yaw={ladder_yaw_deg:g} deg", flush=True)
        ladder_cfg = sim_utils.UsdFileCfg(usd_path=str(ladder_asset))
        ladder_cfg.func(
            "/World/Ladder",
            ladder_cfg,
            translation=ladder_translation,
            orientation=_yaw_quat_wxyz(ladder_yaw_deg),
        )
        print("[INFO] Ladder asset spawned.", flush=True)

    robot = scene["robot"]
    joint_names = motion["joint_names"]
    body_names = motion["body_names"]
    joint_ids = robot.find_joints(joint_names, preserve_order=True)[0]
    if len(joint_ids) != len(joint_names):
        raise RuntimeError(f"Expected {len(joint_names)} joints, found {len(joint_ids)}")

    if args_cli.skip_markers:
        limb_body_names = []
        limb_ids = []
        print("[INFO] Skipping contact markers.", flush=True)
        limb_marker = None
        target_marker = None
        target_contacts = None
        target_marker_contacts = None
        rung_center_marker = None
    else:
        limb_body_names = motion["limb_body_names"]
        limb_ids = [body_names.index(name) for name in limb_body_names]
        print("[INFO] Creating contact markers.", flush=True)
        limb_marker = _make_limb_marker(limb_body_names)
        target_marker = _make_target_marker()
        target_contacts = _target_contacts(motion, sim.device)
        target_marker_contacts = target_contacts.clone()
        plane_normal = torch.tensor(
            _motion_ladder_geometry(motion).plane_normal,
            dtype=target_marker_contacts.dtype,
            device=target_marker_contacts.device,
        )
        heel_direction = plane_normal.clone()
        heel_direction[2] = 0.0
        heel_direction /= torch.linalg.vector_norm(heel_direction)
        foot_limb_indexes = [
            index for index, limb_name in enumerate(motion["limb_names"]) if limb_name.endswith("_foot")
        ]
        target_marker_contacts[:, foot_limb_indexes, :] += 0.10 * heel_direction
        if args_cli.show_rung_centers:
            rung_center_marker = _make_rung_center_marker()
            rung_center_marker.visualize(rung_positions_tensor(_motion_ladder_geometry(motion), device=sim.device))
        else:
            rung_center_marker = None
    tensors = _motion_tensors(motion, sim.device)
    print("[INFO] Motion tensors cached.", flush=True)

    num_frames = int(tensors["joint_pos"].shape[0])
    if args_cli.start_frame < 0 or args_cli.start_frame >= num_frames:
        raise ValueError(f"--start_frame must be in [0, {num_frames}), got {args_cli.start_frame}")
    end_frame = num_frames - 1 if args_cli.end_frame is None else args_cli.end_frame
    if end_frame < args_cli.start_frame or end_frame >= num_frames:
        raise ValueError(
            f"--end_frame must be in [{args_cli.start_frame}, {num_frames}), got {end_frame}"
        )
    steps = 0
    sim_dt = sim.get_physics_dt()
    motion_fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    controller = ReplayController(
        start_frame=args_cli.start_frame,
        end_frame=end_frame,
        motion_fps=motion_fps,
        paused=args_cli.hold_frame,
        enabled=not args_cli.headless,
        frame_sequence=(
            [
                int(frame)
                for frame in np.asarray(
                    motion.get(
                        "stable_support_anchor_frames",
                        motion.get("zero_command_direct_hold_frames", []),
                    ),
                    dtype=np.int64,
                ).reshape(-1)
                if args_cli.start_frame <= int(frame) <= end_frame
            ]
            if args_cli.stable_frames_only
            else None
        ),
    )
    if args_cli.stable_frames_only and not controller.frame_sequence:
        raise ValueError(
            "--stable_frames_only requires stable_support_anchor_frames or "
            "zero_command_direct_hold_frames in the selected replay range"
        )
    playback_fps = (
        args_cli.playback_fps
        if args_cli.playback_fps is not None
        else motion_fps
    )
    if playback_fps <= 0.0:
        raise ValueError("--playback_fps must be positive")
    if args_cli.viewer_fps <= 0.0:
        raise ValueError("--viewer_fps must be positive")
    print(
        f"[INFO] Replay frames={num_frames}, range=[{args_cli.start_frame}, {end_frame}], "
        f"motion_fps={motion_fps:g}, playback_fps={playback_fps:g}, "
        f"viewer_fps={args_cli.viewer_fps:g}",
        flush=True,
    )
    if controller.frame_sequence:
        print(
            f"[INFO] Stable anchors ({len(controller.frame_sequence)}): "
            f"{controller.frame_sequence}",
            flush=True,
        )
    if not args_cli.skip_markers:
        print(f"[INFO] Limb bodies: {', '.join(limb_body_names)}", flush=True)
    if not args_cli.headless:
        print(
            "[INFO] Controls: Space=play/pause, Left/Right=step, PageDown/PageUp=-/+50, "
            "Home/End=first/final, I=contact diagnostics",
            flush=True,
        )
    if not args_cli.skip_markers:
        print("[INFO] Limb colors: left hand=cyan, right hand=blue, left foot=green, right foot=yellow", flush=True)
        print(
            "[INFO] Target colors: yellow/orange=contact phase (including stable support), "
            "purple=legacy support only, gray=inactive",
            flush=True,
        )
        print("[INFO] Foot target markers: display-only 0.10 m heelward offset; saved targets unchanged", flush=True)
        if args_cli.show_rung_centers:
            print("[INFO] White small markers=rung/step centers", flush=True)
    if args_cli.hold_frame and target_contacts is not None:
        _print_frame_diagnostics(motion, controller.frame, target_contacts)

    initial_root_pos = tensors["body_pos_w"][controller.frame, 0]
    sim.set_camera_view(
        (initial_root_pos + torch.tensor([2.2, 2.2, 0.7], device=sim.device)).cpu().numpy(),
        initial_root_pos.cpu().numpy(),
    )
    frame_period = 1.0 / playback_fps
    viewer_period = 1.0 / args_cli.viewer_fps
    next_render_time = time.perf_counter()
    last_motion_advance_time = next_render_time
    displayed_frame = None

    try:
        while simulation_app.is_running():
            now = time.perf_counter()
            if controller.paused:
                last_motion_advance_time = now
            elif now - last_motion_advance_time >= frame_period:
                controller.advance()
                last_motion_advance_time = now

            frame = controller.frame
            if frame != displayed_frame:
                root_pos = _write_frame(robot, tensors, frame, joint_ids)
                scene.write_data_to_sim()

                if not args_cli.skip_markers:
                    limb_marker.visualize(
                        tensors["body_pos_w"][frame, limb_ids],
                        marker_indices=list(range(len(limb_ids))),
                    )
                    support = tensors.get(
                        "stable_support_contact_indicator",
                        tensors["contact_indicator"],
                    )[frame] > 0.5
                    phase = tensors.get(
                        "contact_phase_indicator", tensors["contact_indicator"]
                    )[frame] > 0.5
                    marker_indices = torch.where(
                        phase,
                        torch.ones_like(support, dtype=torch.long),
                        torch.where(
                            support,
                            torch.full_like(support, 3, dtype=torch.long),
                            torch.zeros_like(support, dtype=torch.long),
                        ),
                    )
                    target_marker.visualize(
                        target_marker_contacts[frame], marker_indices=marker_indices
                    )
                if args_cli.follow_camera and not args_cli.skip_camera_update:
                    sim.set_camera_view(
                        (
                            root_pos
                            + torch.tensor([2.2, 2.2, 0.7], device=sim.device)
                        ).cpu().numpy(),
                        root_pos.cpu().numpy(),
                    )
                status = controller.update_status()
                print(f"\r[REPLAY] {status}", end="", flush=True)
                displayed_frame = frame
            if controller.consume_diagnostics_request() and target_contacts is not None:
                print()
                _print_frame_diagnostics(motion, frame, target_contacts)
            sim.render()
            scene.update(sim_dt)

            steps += 1
            if args_cli.smoke_steps > 0 and steps >= args_cli.smoke_steps:
                print(f"\n[INFO] Smoke replay completed for {steps} frames.")
                break
            next_render_time += viewer_period
            current_time = time.perf_counter()
            if next_render_time < current_time - viewer_period:
                next_render_time = current_time
            time.sleep(max(0.0, next_render_time - current_time))
    finally:
        print()
        controller.close()


def main() -> None:
    motion_file = _check_file(args_cli.motion_file, (".npz",))
    motion = _load_motion(motion_file, require_contact_annotations=not args_cli.skip_markers)
    if args_cli.viewer_fps <= 0.0:
        raise ValueError("--viewer_fps must be positive")
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / args_cli.viewer_fps, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    print("[INFO] Creating replay scene.", flush=True)
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    print("[INFO] Resetting simulation.", flush=True)
    sim.reset()
    print("[INFO] Simulation reset complete.", flush=True)
    run_simulator(sim, scene, motion)


if __name__ == "__main__":
    main()
    simulation_app.close()
