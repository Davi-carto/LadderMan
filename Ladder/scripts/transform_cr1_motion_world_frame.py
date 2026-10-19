"""Align a Ladder CR1 motion between two ladder world poses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from ladder.utils.motion_world_transform import align_motion_ladder_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--source_ladder_position", type=float, nargs=3, required=True)
    parser.add_argument("--source_ladder_yaw_deg", type=float, required=True)
    parser.add_argument("--target_ladder_position", type=float, nargs=3, required=True)
    parser.add_argument("--target_ladder_yaw_deg", type=float, required=True)
    parser.add_argument(
        "--robot_translation",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Approved world translation applied to the robot after ladder-frame alignment.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_file = args.input_file.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input motion not found: {input_file}")
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_file}. Pass --overwrite to replace it.")

    with np.load(input_file, allow_pickle=False) as data:
        motion = {key: data[key] for key in data.files}
    output = align_motion_ladder_frame(
        motion,
        source_ladder_position=args.source_ladder_position,
        source_ladder_yaw_deg=args.source_ladder_yaw_deg,
        target_ladder_position=args.target_ladder_position,
        target_ladder_yaw_deg=args.target_ladder_yaw_deg,
        robot_translation=args.robot_translation,
    )
    output["world_alignment_source_file"] = np.asarray([str(input_file)])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **output)
    print(f"[INFO] Wrote world-aligned motion: {output_file}")
    print(
        "[INFO] Ladder pose: "
        f"{tuple(args.source_ladder_position)}, yaw={args.source_ladder_yaw_deg:g} deg -> "
        f"{tuple(args.target_ladder_position)}, yaw={args.target_ladder_yaw_deg:g} deg"
    )
    print(f"[INFO] Robot post-alignment translation: {tuple(args.robot_translation)} m")
    print(f"[INFO] Frames: {output['joint_pos'].shape[0]}, fps: {float(output['fps'][0]):g}")


if __name__ == "__main__":
    main()
