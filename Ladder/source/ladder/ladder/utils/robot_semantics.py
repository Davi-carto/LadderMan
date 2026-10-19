"""Semantic tagging helpers for CR1 camera tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def spawn_cr1_with_link_semantics(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Any:
    """Spawn CR1 and tag its link prims before PhysX tensor views are created."""

    import isaacsim.core.utils.stage as stage_utils
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf
    from isaaclab.sim.utils import find_matching_prims

    from ladder.robots.cr1 import CR1_COLLISION_BODY_NAMES

    try:
        import Semantics
    except ModuleNotFoundError:
        from pxr import Semantics

    robot_prim = spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation)
    robot_root_paths = [str(prim.GetPath()) for prim in find_matching_prims(str(prim_path))]
    link_paths = robot_link_prim_paths(robot_root_paths, CR1_COLLISION_BODY_NAMES)
    stage = stage_utils.get_current_stage()
    link_prims = [stage.GetPrimAtPath(path) for path in link_paths]
    invalid_paths = [path for path, prim in zip(link_paths, link_prims) if not prim.IsValid()]
    if invalid_paths:
        raise RuntimeError(f"Cannot apply robot semantics to invalid link prims: {invalid_paths}")

    for prim in link_prims:
        semantic = Semantics.SemanticsAPI.Apply(prim, "class_robot")
        semantic.CreateSemanticTypeAttr()
        semantic.CreateSemanticDataAttr()
        semantic.GetSemanticTypeAttr().Set("class")
        semantic.GetSemanticDataAttr().Set("robot")

    print(
        f"[CAMERA] applied class=robot to {len(link_prims)} CR1 link prims before PhysX initialization",
        flush=True,
    )
    return robot_prim


def robot_link_prim_paths(
    robot_root_paths: Sequence[str], robot_link_names: Sequence[str]
) -> tuple[str, ...]:
    """Build unique robot link paths for every matched environment root."""

    link_paths = tuple(
        dict.fromkeys(
            f"{root_path}/{link_name}"
            for root_path in robot_root_paths
            for link_name in robot_link_names
        )
    )
    if not link_paths:
        raise RuntimeError("No CR1 link prim paths were resolved for semantic tagging")
    return link_paths


__all__ = ["robot_link_prim_paths", "spawn_cr1_with_link_semantics"]
