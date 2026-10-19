"""Live camera diagnostics for LadderMan policy replay.

This module is deliberately separate from policy observations. It lets a
nonvisual checkpoint render and inspect the camera inputs before those inputs
are added to a training task.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ladder.utils.robot_semantics import spawn_cr1_with_link_semantics


CAMERA_DIAGNOSTIC_NAME = "camera_diagnostic"


@dataclass(frozen=True)
class CameraDiagnosticSettings:
    """Display and capture settings for one replay camera."""

    output_dir: Path
    save_interval: int = 200
    display_scale: float = 1.0
    show: bool = True
    min_depth: float = 0.1
    max_depth: float = 2.0


def rung_ids_from_semantic_info(semantic_info: Mapping[str, Any] | None) -> set[int]:
    """Return IDs whose Replicator ``class`` label denotes a ladder rung."""

    if not isinstance(semantic_info, Mapping):
        return set()
    id_to_labels = semantic_info.get("idToLabels", {})
    if not isinstance(id_to_labels, Mapping):
        return set()

    rung_ids = set()
    for semantic_id, labels in id_to_labels.items():
        if not isinstance(labels, Mapping):
            continue
        class_label = labels.get("class")
        if class_label is not None and "rung" in str(class_label).lower():
            rung_ids.add(int(semantic_id))
    return rung_ids


def robot_ids_from_semantic_info(semantic_info: Mapping[str, Any] | None) -> set[int]:
    """Return IDs whose Replicator ``class`` label denotes the robot."""

    if not isinstance(semantic_info, Mapping):
        return set()
    id_to_labels = semantic_info.get("idToLabels", {})
    if not isinstance(id_to_labels, Mapping):
        return set()

    robot_ids = set()
    for semantic_id, labels in id_to_labels.items():
        if not isinstance(labels, Mapping):
            continue
        class_label = labels.get("class")
        if class_label is not None and str(class_label).strip().lower() == "robot":
            robot_ids.add(int(semantic_id))
    return robot_ids


def semantic_label_mapping(semantic_info: Mapping[str, Any] | None) -> dict[int, dict[str, str]]:
    """Normalize Replicator's semantic mapping for printing and JSON capture."""

    if not isinstance(semantic_info, Mapping):
        return {}
    id_to_labels = semantic_info.get("idToLabels", {})
    if not isinstance(id_to_labels, Mapping):
        return {}

    mapping = {}
    for semantic_id, labels in id_to_labels.items():
        if isinstance(labels, Mapping):
            mapping[int(semantic_id)] = {str(key): str(value) for key, value in labels.items()}
    return mapping


def rung_mask_from_semantic_info(
    semantic_image: np.ndarray, semantic_info: Mapping[str, Any] | None
) -> np.ndarray:
    """Build a 2-D rung mask for one non-colorized semantic image."""

    semantic = np.asarray(semantic_image)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    if semantic.ndim != 2:
        raise ValueError(
            "Expected one semantic image with shape (height, width) or "
            f"(height, width, 1), got {semantic.shape}"
        )

    rung_ids = sorted(rung_ids_from_semantic_info(semantic_info))
    if not rung_ids:
        return np.zeros(semantic.shape, dtype=bool)
    return np.isin(semantic, np.asarray(rung_ids, dtype=semantic.dtype))


def rfm_keep_mask_from_semantic_info(
    semantic_image: np.ndarray, semantic_info: Mapping[str, Any] | None
) -> np.ndarray:
    """Build the paper-aligned RFM mask that keeps ladder rungs and the robot."""

    rung_mask = rung_mask_from_semantic_info(semantic_image, semantic_info)
    robot_ids = sorted(robot_ids_from_semantic_info(semantic_info))
    if not robot_ids:
        return rung_mask

    semantic = np.asarray(semantic_image)
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    return rung_mask | np.isin(semantic, np.asarray(robot_ids, dtype=semantic.dtype))


def attach_cr1_camera_diagnostic(
    scene_cfg: Any,
    *,
    width: int,
    height: int,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    convention: str,
    min_depth: float,
    max_depth: float,
) -> None:
    """Attach a diagnostic-only D435-style camera to an existing CR1 scene."""

    if hasattr(scene_cfg, CAMERA_DIAGNOSTIC_NAME):
        raise ValueError(f"Scene already contains a sensor named {CAMERA_DIAGNOSTIC_NAME!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Diagnostic camera dimensions must be positive, got {width}x{height}")
    if not 0.0 <= min_depth < max_depth:
        raise ValueError(f"Invalid diagnostic depth interval [{min_depth}, {max_depth}]")

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    scene_cfg.robot.spawn.semantic_tags = [("class", "robot")]
    scene_cfg.robot.spawn.func = spawn_cr1_with_link_semantics
    setattr(
        scene_cfg,
        CAMERA_DIAGNOSTIC_NAME,
        TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/d435_head_link/CameraDiagnostic",
            update_period=0.02,
            height=height,
            width=width,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=45.5,
                clipping_range=(min_depth, max_depth),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=position,
                rot=rotation,
                convention=convention,
            ),
            depth_clipping_behavior="max",
            semantic_filter=["class"],
            colorize_semantic_segmentation=False,
        ),
    )


class CameraDiagnostics:
    """Render, print, and optionally save diagnostic frames from one camera."""

    _WINDOW_RGB = "CR1 camera RGB"
    _WINDOW_DEPTH = "CR1 camera depth (m)"
    _WINDOW_MASKED_DEPTH = "CR1 camera RFM depth (rungs+robot)"
    _WINDOW_MASK = "CR1 camera RFM keep mask (rungs+robot)"
    _WINDOW_OVERLAY = "CR1 camera RGB+rungs"
    _WINDOW_NAMES = (
        _WINDOW_RGB,
        _WINDOW_DEPTH,
        _WINDOW_MASKED_DEPTH,
        _WINDOW_MASK,
        _WINDOW_OVERLAY,
    )

    def __init__(self, settings: CameraDiagnosticSettings):
        if settings.save_interval < 0:
            raise ValueError("save_interval must be non-negative")
        if settings.display_scale <= 0.0:
            raise ValueError("display_scale must be positive")
        if not 0.0 <= settings.min_depth < settings.max_depth:
            raise ValueError("Diagnostic depth bounds must satisfy 0 <= min < max")

        import cv2

        self.settings = settings
        self._cv2 = cv2
        self._display_backend = (
            "opencv" if _opencv_highgui_available(cv2.getBuildInformation()) else "omni"
        )
        self._omni_ui = None
        self._omni_windows = {}
        self._omni_image_providers = {}
        if settings.show and self._display_backend == "omni":
            import omni.ui

            self._omni_ui = omni.ui
            print(
                "[CAMERA] OpenCV HighGUI is unavailable; displaying camera images "
                "in Isaac Sim windows.",
                flush=True,
            )
        self._printed_mapping = None
        self._printed_frame_properties = False
        self._capture_index = 0

    def update(self, sensor: Any, control_step: int) -> bool:
        """Display the first-environment frame and return whether to quit."""

        rgb = _first_image(sensor.data.output["rgb"])
        depth = _first_image(sensor.data.output["distance_to_image_plane"])
        semantic = _first_image(sensor.data.output["semantic_segmentation"])
        semantic_info = sensor.data.info.get("semantic_segmentation")
        mapping = semantic_label_mapping(semantic_info)
        rung_ids = rung_ids_from_semantic_info(semantic_info)
        rung_mask = rung_mask_from_semantic_info(semantic, semantic_info)
        robot_ids = robot_ids_from_semantic_info(semantic_info)
        rfm_keep_mask = rfm_keep_mask_from_semantic_info(semantic, semantic_info)

        rgb = _as_uint8_rgb(rgb)
        depth = _as_depth_2d(depth)
        self._print_frame_properties(
            sensor,
            rgb,
            depth,
            semantic,
            mapping,
            rung_ids,
            robot_ids,
            rung_mask,
            rfm_keep_mask,
        )

        rgb_bgr = rgb[..., ::-1]
        depth_color = self._colorize_depth(depth)
        mask_image = rfm_keep_mask.astype(np.uint8) * 255
        masked_depth_color = _mask_excluded_pixels(depth_color, rfm_keep_mask)
        overlay = rgb_bgr.copy()
        overlay[rung_mask] = np.round(
            0.45 * overlay[rung_mask] + 0.55 * np.asarray((0, 0, 255), dtype=np.float32)
        ).astype(np.uint8)

        if self.settings.show:
            self._show(self._WINDOW_RGB, rgb_bgr)
            self._show(self._WINDOW_DEPTH, depth_color)
            self._show(self._WINDOW_MASKED_DEPTH, masked_depth_color)
            self._show(self._WINDOW_MASK, mask_image)
            self._show(self._WINDOW_OVERLAY, overlay)
            if self._display_backend == "opencv":
                key = self._cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("[CAMERA] quit requested from OpenCV window", flush=True)
                    return True

        if self._should_save(control_step):
            self._save_capture(
                control_step,
                rgb_bgr,
                depth,
                depth_color,
                masked_depth_color,
                semantic,
                rung_mask,
                mask_image,
                overlay,
                mapping,
                rung_ids,
                robot_ids,
                sensor,
            )
        return False

    def close(self) -> None:
        """Close only the diagnostic windows created by this object."""

        if not self.settings.show:
            return
        if self._display_backend == "opencv":
            for window_name in self._WINDOW_NAMES:
                try:
                    self._cv2.destroyWindow(window_name)
                except self._cv2.error:
                    pass
        else:
            for window in self._omni_windows.values():
                window.visible = False
                window.destroy()
            self._omni_windows.clear()
            self._omni_image_providers.clear()

    def _print_frame_properties(
        self,
        sensor: Any,
        rgb: np.ndarray,
        depth: np.ndarray,
        semantic: np.ndarray,
        mapping: dict[int, dict[str, str]],
        rung_ids: set[int],
        robot_ids: set[int],
        rung_mask: np.ndarray,
        rfm_keep_mask: np.ndarray,
    ) -> None:
        if not self._printed_frame_properties:
            intrinsics = _as_numpy(sensor.data.intrinsic_matrices[0]).tolist()
            finite_depth = depth[np.isfinite(depth)]
            depth_range = (
                (float(finite_depth.min()), float(finite_depth.max()))
                if finite_depth.size
                else (float("nan"), float("nan"))
            )
            raw_ids = np.unique(np.squeeze(semantic)).astype(np.int64).tolist()
            print("[CAMERA] diagnostic sensor initialized", flush=True)
            print(
                f"  rgb_shape={rgb.shape} depth_shape={depth.shape} semantic_shape={semantic.shape} "
                f"depth_range_m={depth_range} intrinsics={intrinsics}",
                flush=True,
            )
            print(
                f"  raw_semantic_ids={raw_ids} rung_ids={sorted(rung_ids)} "
                f"robot_ids={sorted(robot_ids)} rung_pixel_fraction={float(rung_mask.mean()):.6f} "
                f"rfm_keep_pixel_fraction={float(rfm_keep_mask.mean()):.6f}",
                flush=True,
            )
            self._printed_frame_properties = True

        if mapping != self._printed_mapping:
            print(f"[CAMERA] semantic idToLabels={mapping}", flush=True)
            if not rung_ids:
                print(
                    "[CAMERA][WARNING] no semantic ID is labeled class=rung; "
                    "the RFM views will contain only robot pixels.",
                    flush=True,
                )
            if not robot_ids:
                print(
                    "[CAMERA][WARNING] no semantic ID is labeled class=robot; "
                    "the RFM views cannot retain the robot.",
                    flush=True,
                )
            self._printed_mapping = mapping

    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        clipped = np.clip(depth, self.settings.min_depth, self.settings.max_depth)
        normalized = (clipped - self.settings.min_depth) / (
            self.settings.max_depth - self.settings.min_depth
        )
        image = np.round(normalized * 255.0).astype(np.uint8)
        image[~np.isfinite(depth)] = 0
        return self._cv2.applyColorMap(image, self._cv2.COLORMAP_TURBO)

    def _show(self, name: str, image: np.ndarray) -> None:
        resized = self._cv2.resize(
            image,
            dsize=None,
            fx=self.settings.display_scale,
            fy=self.settings.display_scale,
            interpolation=self._cv2.INTER_NEAREST,
        )
        if self._display_backend == "opencv":
            self._cv2.imshow(name, resized)
            return

        rgba = _bgr_or_gray_to_rgba(resized)
        height, width = rgba.shape[:2]
        provider = self._omni_image_providers.get(name)
        if provider is None:
            provider = self._omni_ui.ByteImageProvider()
            window_index = self._WINDOW_NAMES.index(name)
            window = self._omni_ui.Window(
                name,
                width=width,
                height=height + 30,
                position_x=(window_index % 3) * width,
                position_y=10 + (window_index // 3) * (height + 30),
                visible=True,
            )
            with window.frame:
                self._omni_ui.ImageWithProvider(
                    provider,
                    fill_policy=self._omni_ui.IwpFillPolicy.IWP_STRETCH,
                )
            self._omni_windows[name] = window
            self._omni_image_providers[name] = provider
        provider.set_bytes_data(rgba.flatten().data, [width, height])

    def _should_save(self, control_step: int) -> bool:
        return self.settings.save_interval > 0 and control_step % self.settings.save_interval == 0

    def _save_capture(
        self,
        control_step: int,
        rgb_bgr: np.ndarray,
        depth: np.ndarray,
        depth_color: np.ndarray,
        masked_depth_color: np.ndarray,
        semantic: np.ndarray,
        rung_mask: np.ndarray,
        mask_image: np.ndarray,
        overlay: np.ndarray,
        mapping: dict[int, dict[str, str]],
        rung_ids: set[int],
        robot_ids: set[int],
        sensor: Any,
    ) -> None:
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"capture_{self._capture_index:04d}_step_{control_step:06d}"
        self._cv2.imwrite(str(self.settings.output_dir / f"{stem}_rgb.png"), rgb_bgr)
        self._cv2.imwrite(str(self.settings.output_dir / f"{stem}_depth.png"), depth_color)
        self._cv2.imwrite(
            str(self.settings.output_dir / f"{stem}_rfm_depth.png"),
            masked_depth_color,
        )
        self._cv2.imwrite(str(self.settings.output_dir / f"{stem}_rfm_keep_mask.png"), mask_image)
        self._cv2.imwrite(str(self.settings.output_dir / f"{stem}_overlay.png"), overlay)
        np.save(self.settings.output_dir / f"{stem}_depth_m.npy", depth)
        np.save(self.settings.output_dir / f"{stem}_semantic.npy", semantic)
        metadata = {
            "control_step": control_step,
            "depth_display_range_m": [self.settings.min_depth, self.settings.max_depth],
            "intrinsic_matrix": _as_numpy(sensor.data.intrinsic_matrices[0]).tolist(),
            "semantic_id_to_labels": {str(key): value for key, value in mapping.items()},
            "rung_semantic_ids": sorted(rung_ids),
            "robot_semantic_ids": sorted(robot_ids),
            "rung_pixel_fraction": float(rung_mask.mean()),
            "rfm_keep_pixel_fraction": float(np.count_nonzero(mask_image) / mask_image.size),
        }
        metadata_path = self.settings.output_dir / f"{stem}_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
        print(f"[CAMERA] saved {stem} to {self.settings.output_dir}", flush=True)
        self._capture_index += 1


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _opencv_highgui_available(build_information: str) -> bool:
    """Return whether an OpenCV build reports a usable window backend."""

    for line in build_information.splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", maxsplit=1)[1].strip().upper() != "NONE"
    return True


def _bgr_or_gray_to_rgba(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV display data into the RGBA layout expected by omni.ui."""

    source = np.asarray(image)
    if source.ndim == 2:
        rgb = np.repeat(source[..., None], 3, axis=-1)
    elif source.ndim == 3 and source.shape[-1] == 3:
        rgb = source[..., ::-1]
    else:
        raise ValueError(f"Expected BGR or grayscale image, got {source.shape}")

    rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    return np.ascontiguousarray(rgba)


def _mask_excluded_pixels(depth_color: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Keep selected depth colors and render all other pixels black."""

    image = np.asarray(depth_color)
    mask = np.asarray(keep_mask, dtype=bool)
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Depth image and keep mask shapes do not match: {image.shape} vs {mask.shape}")
    masked = np.zeros_like(image)
    masked[mask] = image[mask]
    return masked


def _first_image(value: Any) -> np.ndarray:
    image = _as_numpy(value)
    if image.ndim < 3:
        raise ValueError(f"Camera output must include environment and image dimensions, got {image.shape}")
    return image[0]


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"Expected RGB image with at least three channels, got {rgb.shape}")
    return np.ascontiguousarray(rgb[..., :3].astype(np.uint8, copy=False))


def _as_depth_2d(image: np.ndarray) -> np.ndarray:
    depth = np.asarray(image)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth image with shape (height, width, 1), got {depth.shape}")
    return depth.astype(np.float32, copy=False)


__all__ = [
    "CAMERA_DIAGNOSTIC_NAME",
    "CameraDiagnosticSettings",
    "CameraDiagnostics",
    "attach_cr1_camera_diagnostic",
    "rfm_keep_mask_from_semantic_info",
    "robot_ids_from_semantic_info",
    "rung_ids_from_semantic_info",
    "rung_mask_from_semantic_info",
    "semantic_label_mapping",
]
