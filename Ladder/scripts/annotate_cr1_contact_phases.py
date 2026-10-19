"""Add finite-tread surface contact-phase labels to an annotated CR1 motion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


SOURCE_DIR = Path(__file__).resolve().parents[1] / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from ladder.tasks.ladder_climb.ladder_geometry import generate_ladder
from ladder.utils.contact_annotation import (
    ContactPhaseAnnotationCfg,
    DirectionalContactPhaseCfg,
    annotate_directional_ladder_contact_phases,
    annotate_ladder_contact_phases,
)
from ladder.utils.contact_schema import LIMB_NAMES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add an independent finite-tread contact-phase preview without changing stable contacts."
    )
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument(
        "--entry_distances",
        type=float,
        nargs=4,
        default=(0.06, 0.06, 0.06, 0.06),
        metavar=("LH", "RH", "LF", "RF"),
    )
    parser.add_argument(
        "--release_distances",
        type=float,
        nargs=4,
        default=(0.08, 0.08, 0.10, 0.10),
        metavar=("LH", "RH", "LF", "RF"),
    )
    parser.add_argument(
        "--entry_min_clearances",
        type=float,
        nargs=4,
        default=(-0.01, -0.01, -0.01, -0.01),
        metavar=("LH", "RH", "LF", "RF"),
    )
    parser.add_argument(
        "--release_min_clearances",
        type=float,
        nargs=4,
        default=(-0.015, -0.015, -0.015, -0.015),
        metavar=("LH", "RH", "LF", "RF"),
    )
    parser.add_argument(
        "--contact_point_z_offsets",
        type=float,
        nargs=4,
        default=(-0.045, -0.045, 0.0, 0.0),
        metavar=("LH", "RH", "LF", "RF"),
        help="World-Z offsets from stored limb body points; -45 mm maps sphere centers to their bottoms.",
    )
    parser.add_argument("--min_contact_frames", type=int, default=4)
    parser.add_argument("--max_gap_frames", type=int, default=5)
    parser.add_argument(
        "--directional_limbs",
        nargs="+",
        choices=LIMB_NAMES,
        default=None,
        help="Selected limbs that follow monotonic up/down rung order and departure locking.",
    )
    parser.add_argument("--top_neutral_start_frame", type=int, default=None)
    parser.add_argument("--descent_start_frame", type=int, default=None)
    parser.add_argument("--directional_core_distance", type=float, default=0.01)
    parser.add_argument("--directional_departure_distance", type=float, default=0.01)
    parser.add_argument("--directional_departure_rate", type=float, default=0.10)
    parser.add_argument("--directional_departure_normal_velocity", type=float, default=0.20)
    parser.add_argument("--directional_confirmation_frames", type=int, default=2)
    parser.add_argument(
        "--promote_to_contact_contract",
        action="store_true",
        help=(
            "Write the reviewed contact phase to contact_indicator/contact_rung_id "
            "and omit the duplicate contact_phase_indicator/contact_phase_rung_id keys."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _as_strings(values: np.ndarray) -> tuple[str, ...]:
    items = np.asarray(values).reshape(-1).tolist()
    return tuple(item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items)


def _active_runs(mask: np.ndarray, rung_ids: np.ndarray) -> list[tuple[int, int, int]]:
    runs = []
    start = None
    active_rung = -1
    for frame, active in enumerate(mask.tolist() + [False]):
        rung_id = int(rung_ids[frame]) if frame < len(rung_ids) else -1
        if active and (start is None or rung_id != active_rung):
            if start is not None:
                runs.append((start, frame - 1, active_rung))
            start = frame
            active_rung = rung_id
        elif not active and start is not None:
            runs.append((start, frame - 1, active_rung))
            start = None
            active_rung = -1
    return runs


def main() -> None:
    args = _parse_args()
    input_file = args.input_file.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_file}; pass --overwrite to replace it")

    with np.load(input_file, allow_pickle=False) as data:
        motion = {key: data[key] for key in data.files}

    required = (
        "body_pos_w",
        "body_names",
        "limb_names",
        "limb_body_names",
        "contact_indicator",
        "contact_rung_id",
        "contact_annotation_ladder_params",
        "contact_annotation_ladder_origin",
        "contact_annotation_ladder_num_rungs",
    )
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"Input motion is missing required fields: {missing}")

    limb_names = _as_strings(motion["limb_names"])
    if limb_names != tuple(LIMB_NAMES):
        raise ValueError(f"Expected limb order {tuple(LIMB_NAMES)}, got {limb_names}")
    limb_body_names = _as_strings(motion["limb_body_names"])
    params = np.asarray(motion["contact_annotation_ladder_params"], dtype=np.float64).reshape(4)
    origin = np.asarray(motion["contact_annotation_ladder_origin"], dtype=np.float64).reshape(3)
    num_rungs = int(np.asarray(motion["contact_annotation_ladder_num_rungs"]).reshape(-1)[0])
    geometry = generate_ladder(
        phi=float(params[0]),
        rung_spacing=float(params[1]),
        rung_length=float(params[2]),
        rung_radius_or_width=float(params[3]),
        num_rungs=num_rungs,
        origin=tuple(float(value) for value in origin),
    )
    cfg = ContactPhaseAnnotationCfg(
        ladder_geometry=geometry,
        limb_names=limb_names,
        limb_body_names=limb_body_names,
        entry_distance_thresholds=tuple(args.entry_distances),
        release_distance_thresholds=tuple(args.release_distances),
        entry_min_clearances=tuple(args.entry_min_clearances),
        release_min_clearances=tuple(args.release_min_clearances),
        contact_point_z_offsets=tuple(args.contact_point_z_offsets),
        min_contact_frames=args.min_contact_frames,
        max_gap_frames=args.max_gap_frames,
    )
    if args.directional_limbs is None:
        if args.top_neutral_start_frame is not None or args.descent_start_frame is not None:
            raise ValueError("direction frames require --directional_limbs")
        labels = annotate_ladder_contact_phases(motion, cfg)
    else:
        if args.top_neutral_start_frame is None or args.descent_start_frame is None:
            raise ValueError(
                "--directional_limbs requires --top_neutral_start_frame and --descent_start_frame"
            )
        count = len(args.directional_limbs)
        directional_cfg = DirectionalContactPhaseCfg(
            limb_names=tuple(args.directional_limbs),
            top_neutral_start_frame=args.top_neutral_start_frame,
            descent_start_frame=args.descent_start_frame,
            core_distance_thresholds=(args.directional_core_distance,) * count,
            departure_distance_thresholds=(args.directional_departure_distance,) * count,
            departure_rate_thresholds=(args.directional_departure_rate,) * count,
            departure_normal_velocity_thresholds=(
                args.directional_departure_normal_velocity,
            )
            * count,
            confirmation_frames=args.directional_confirmation_frames,
        )
        labels = annotate_directional_ladder_contact_phases(motion, cfg, directional_cfg)
    output = dict(motion)
    output.update(labels)
    output["contact_phase_annotation_source_file"] = np.asarray(str(input_file))
    if args.promote_to_contact_contract:
        output["contact_indicator"] = output.pop("contact_phase_indicator")
        output["contact_rung_id"] = output.pop("contact_phase_rung_id")
        output["contact_contract"] = np.asarray("finite_tread_contact_phase")
        output["contact_contract_promoted_from"] = np.asarray(
            "contact_phase_indicator/contact_phase_rung_id"
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **output)

    print(f"[INFO] Wrote surface contact-phase preview: {output_file}")
    print(f"[INFO] Source preserved unchanged: {input_file}")
    print("[INFO] Limb order:", ", ".join(limb_names))
    print("[INFO] Entry distances:", "/".join(f"{value:.3f}" for value in args.entry_distances), "m")
    print("[INFO] Release distances:", "/".join(f"{value:.3f}" for value in args.release_distances), "m")
    print(f"[INFO] Same-rung gaps filled: <= {args.max_gap_frames} frames")
    if args.directional_limbs is not None:
        print("[INFO] Directional limbs:", ", ".join(args.directional_limbs))
        effective_confirmation = int(
            labels["contact_phase_directional_effective_departure_confirmation_frames"][0]
        )
        print(
            f"[INFO] Directional departure confirmation: {effective_confirmation} consecutive frames"
        )
        print(
            f"[INFO] Travel segments: up=[0,{args.top_neutral_start_frame - 1}], "
            f"neutral=[{args.top_neutral_start_frame},{args.descent_start_frame - 1}], "
            f"down=[{args.descent_start_frame},{len(motion['body_pos_w']) - 1}]"
        )
    if args.promote_to_contact_contract:
        print(
            "[INFO] Promoted phase labels to the runtime "
            "contact_indicator/contact_rung_id contract."
        )
    support = np.asarray(motion["contact_indicator"]) > 0.5
    phase = labels["contact_phase_indicator"] > 0.5
    rung_ids = labels["contact_phase_rung_id"]
    for limb_index, limb_name in enumerate(limb_names):
        runs = _active_runs(phase[:, limb_index], rung_ids[:, limb_index])
        print(
            f"[INFO] {limb_name}: phase_frames={int(phase[:, limb_index].sum())} "
            f"support_frames={int(support[:, limb_index].sum())} "
            f"phase_only_frames={int((phase[:, limb_index] & ~support[:, limb_index]).sum())} "
            f"runs={runs}"
        )


if __name__ == "__main__":
    main()
