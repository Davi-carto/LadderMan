"""Annotate a single LadderMan reference motion with contact labels."""

from __future__ import annotations

import argparse
import os
import sys


SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source", "ladder"))
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

from ladder.utils.contact_annotation import ContactAnnotationCfg, annotate_motion_file, make_contact_annotation_cfg
from ladder.utils.contact_schema import (
    CR1_CONTACT_DISTANCE_THRESHOLDS,
    CR1_CONTACT_TARGET_OFFSETS,
    CR1_INITIAL_SOLE_TARGET_Z,
    CR1_LIMB_BODY_NAMES,
    DEFAULT_LIMB_BODY_NAMES,
    LIMB_NAMES,
)
from ladder.utils.ladder_presets import HOLOSOMA_CR1_WOOD_STEP, LADDER_A_CONTACT_PRESET


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate LadderMan contact labels for one upward reference motion.")
    parser.add_argument("--input_file", type=str, required=True, help="Input WBT/IsaacLab-style motion.npz.")
    parser.add_argument("--output_file", type=str, required=True, help="Output motion npz with contact labels.")
    parser.add_argument("--ladder_phi", type=float, default=None, help="Ladder inclination in degrees or radians.")
    parser.add_argument("--rung_spacing", type=float, default=None, help="Rung spacing in meters.")
    parser.add_argument("--rung_length", type=float, default=None, help="Rung length in meters.")
    parser.add_argument("--rung_width", type=float, default=None, help="Rung width/radius in meters.")
    parser.add_argument("--num_rungs", type=int, default=None, help="Number of ladder rungs.")
    parser.add_argument(
        "--ladder_origin",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Ladder origin used by contact target generation.",
    )
    parser.add_argument(
        "--ladder_asset_translation",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Displayed/physical ladder asset translation stored in the annotated motion.",
    )
    parser.add_argument(
        "--ladder_asset_yaw_deg",
        type=float,
        default=None,
        help="Displayed/physical ladder asset yaw stored in the annotated motion.",
    )
    parser.add_argument("--robot", choices=("g1", "cr1"), default="g1", help="Default limb body names to use.")
    parser.add_argument(
        "--limb_body_names",
        type=str,
        nargs=len(LIMB_NAMES),
        default=None,
        metavar=("LEFT_HAND", "RIGHT_HAND", "LEFT_FOOT", "RIGHT_FOOT"),
        help="Optional limb body names overriding --robot defaults.",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=None,
        help="Uniform max limb-to-target distance; overrides the CR1 per-limb defaults.",
    )
    parser.add_argument(
        "--limb_distance_thresholds",
        type=float,
        nargs=len(LIMB_NAMES),
        default=None,
        metavar=("LEFT_HAND", "RIGHT_HAND", "LEFT_FOOT", "RIGHT_FOOT"),
        help="Optional per-limb contact distance thresholds.",
    )
    parser.add_argument(
        "--limb_release_distance_thresholds",
        type=float,
        nargs=len(LIMB_NAMES),
        default=None,
        metavar=("LEFT_HAND", "RIGHT_HAND", "LEFT_FOOT", "RIGHT_FOOT"),
        help="Optional wider same-rung release thresholds for brief target-distance excursions.",
    )
    parser.add_argument(
        "--limb_target_offsets",
        type=float,
        nargs=3 * len(LIMB_NAMES),
        default=None,
        help=(
            "Optional per-limb XYZ target offsets in left-hand, right-hand, left-foot, right-foot order. "
            "Overrides the robot defaults for this output file only."
        ),
    )
    parser.add_argument("--speed_threshold", type=float, default=0.35, help="Max limb speed for contact.")
    parser.add_argument("--min_contact_frames", type=int, default=4, help="Remove shorter contact runs.")
    parser.add_argument("--max_gap_frames", type=int, default=2, help="Fill shorter gaps inside contact runs.")
    parser.add_argument(
        "--ground_initial_robot",
        action="store_true",
        help="Explicitly shift the full CR1 body trajectory so the lowest initial sole point reaches the target Z.",
    )
    parser.add_argument(
        "--initial_foot_target_z",
        type=float,
        default=None,
        help="Target world Z used with --ground_initial_robot; CR1 defaults to -0.03 m.",
    )
    parser.add_argument(
        "--limb_body_indexes",
        type=int,
        nargs=len(LIMB_NAMES),
        default=None,
        metavar=("LEFT_HAND", "RIGHT_HAND", "LEFT_FOOT", "RIGHT_FOOT"),
        help="Optional body_pos_w indexes for left hand, right hand, left foot, right foot.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_file if it exists.")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.distance_threshold is not None and args.limb_distance_thresholds is not None:
        raise ValueError("use either --distance_threshold or --limb_distance_thresholds, not both")
    if args.initial_foot_target_z is not None and not args.ground_initial_robot:
        raise ValueError("--initial_foot_target_z requires --ground_initial_robot")
    ladder_preset = HOLOSOMA_CR1_WOOD_STEP if args.robot == "cr1" else LADDER_A_CONTACT_PRESET
    if args.limb_body_names is not None:
        limb_body_names = tuple(args.limb_body_names)
    elif args.robot == "cr1":
        limb_body_names = tuple(CR1_LIMB_BODY_NAMES[name] for name in LIMB_NAMES)
    else:
        limb_body_names = tuple(DEFAULT_LIMB_BODY_NAMES[name] for name in LIMB_NAMES)
    if args.limb_target_offsets is not None:
        limb_target_offsets = tuple(
            tuple(args.limb_target_offsets[3 * index : 3 * index + 3])
            for index in range(len(LIMB_NAMES))
        )
    else:
        limb_target_offsets = (
            tuple(CR1_CONTACT_TARGET_OFFSETS[name] for name in LIMB_NAMES)
            if args.robot == "cr1"
            else None
        )
    distance_threshold = (
        args.distance_threshold
        if args.distance_threshold is not None
        else (
            max(CR1_CONTACT_DISTANCE_THRESHOLDS.values())
            if args.robot == "cr1"
            else ContactAnnotationCfg().distance_threshold
        )
    )
    limb_distance_thresholds = (
        tuple(args.limb_distance_thresholds)
        if args.limb_distance_thresholds is not None
        else (
            tuple(CR1_CONTACT_DISTANCE_THRESHOLDS[name] for name in LIMB_NAMES)
            if args.robot == "cr1" and args.distance_threshold is None
            else None
        )
    )

    cfg = make_contact_annotation_cfg(
        ladder_phi=args.ladder_phi if args.ladder_phi is not None else ladder_preset.phi_deg,
        rung_spacing=args.rung_spacing if args.rung_spacing is not None else ladder_preset.rung_spacing,
        rung_length=args.rung_length if args.rung_length is not None else ladder_preset.rung_length,
        rung_width=args.rung_width if args.rung_width is not None else ladder_preset.rung_width,
        num_rungs=args.num_rungs if args.num_rungs is not None else ladder_preset.num_rungs,
        ladder_origin=tuple(args.ladder_origin) if args.ladder_origin is not None else ladder_preset.target_origin,
        limb_body_names=limb_body_names,
        limb_target_offsets=limb_target_offsets,
        distance_threshold=distance_threshold,
        limb_distance_thresholds=limb_distance_thresholds,
        limb_release_distance_thresholds=(
            tuple(args.limb_release_distance_thresholds)
            if args.limb_release_distance_thresholds is not None
            else None
        ),
        speed_threshold=args.speed_threshold,
        min_contact_frames=args.min_contact_frames,
        max_gap_frames=args.max_gap_frames,
        ladder_asset_translation=(
            tuple(args.ladder_asset_translation)
            if args.ladder_asset_translation is not None
            else ladder_preset.asset_translation
        ),
        ladder_asset_yaw_deg=(
            args.ladder_asset_yaw_deg
            if args.ladder_asset_yaw_deg is not None
            else ladder_preset.asset_yaw_deg
        ),
        ground_initial_feet=args.robot == "cr1" and args.ground_initial_robot,
        initial_foot_target_z=(
            args.initial_foot_target_z
            if args.initial_foot_target_z is not None
            else CR1_INITIAL_SOLE_TARGET_Z if args.robot == "cr1" else 0.0
        ),
    )
    labels = annotate_motion_file(
        args.input_file,
        args.output_file,
        cfg,
        limb_body_indexes=args.limb_body_indexes,
        overwrite=args.overwrite,
    )

    print(f"[INFO] Wrote annotated motion: {args.output_file}")
    print(f"[INFO] Ladder preset: {ladder_preset.name}")
    if "contact_annotation_robot_grounding_z_offset" in labels:
        grounding_offset = float(labels["contact_annotation_robot_grounding_z_offset"][0])
        target_z = float(labels["contact_annotation_robot_initial_foot_target_z"][0])
        print(f"[INFO] Initial robot Z offset: {grounding_offset:.6f} m (lowest sole target Z: {target_z:.6f} m)")
    print("[INFO] Limb order:", ", ".join(LIMB_NAMES))
    print("[INFO] Limb bodies:", ", ".join(limb_body_names))
    if limb_target_offsets is not None:
        print("[INFO] Limb target offsets:", limb_target_offsets)
    if limb_distance_thresholds is None:
        distance_summary = f"distance={distance_threshold:.3f} m"
    else:
        distance_summary = "distance=" + "/".join(f"{value:.3f}" for value in limb_distance_thresholds) + " m"
    print(f"[INFO] Contact thresholds: {distance_summary}, speed={args.speed_threshold:.3f} m/s")
    if args.limb_release_distance_thresholds is not None:
        print(
            "[INFO] Same-rung release distances: "
            + "/".join(f"{value:.3f}" for value in args.limb_release_distance_thresholds)
            + " m"
        )
    if args.limb_body_indexes is None:
        print("[INFO] Limb body indexes resolved from motion body_names or the default local G1 body order.")
    else:
        print("[INFO] Limb body indexes:", args.limb_body_indexes)

    indicator = labels["contact_indicator"]
    rung_ids = labels["contact_rung_id"]
    for limb_index, limb_name in enumerate(LIMB_NAMES):
        active = indicator[:, limb_index] > 0.5
        rung_summary = sorted(set(rung_ids[active, limb_index].tolist())) if active.any() else []
        fraction = float(active.mean())
        print(f"[INFO] {limb_name}: contact_fraction={fraction:.3f}, contact_rungs={rung_summary}")


if __name__ == "__main__":
    main()
