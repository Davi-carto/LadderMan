"""Generate the CR1 training URDF with Holosoma's primitive collision geometry."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree


ASSET_DIR = Path(__file__).resolve().parent
SOURCE_URDF = ASSET_DIR / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
PRIMITIVE_URDF = ASSET_DIR / "CR1-B2-pro_simplify_collision.urdf"
OUTPUT_URDF = ASSET_DIR / "CR1-B2-pro_free_wrist_fixed_spherehand_holosoma_collision.urdf"

SPHERE_HAND_LINKS = {"left_sphere_hand_link", "right_sphere_hand_link"}
SKIPPED_PRIMITIVE_LINKS = {"left_hand_link", "right_hand_link"}
FOOT_COLLIDERS = {
    "left": (
        ("foot_sphere_1", (-0.06, 0.04, -0.06)),
        ("foot_sphere_2", (-0.06, -0.04, -0.06)),
        ("foot_sphere_3", (0.14, 0.04, -0.06)),
        ("foot_sphere_4", (0.14, -0.04, -0.06)),
        ("toe", (0.16, 0.0, -0.055)),
    ),
    "right": (
        ("foot_sphere_1", (-0.06, 0.04, -0.06)),
        ("foot_sphere_2", (-0.06, -0.04, -0.06)),
        ("foot_sphere_3", (0.14, 0.04, -0.06)),
        ("foot_sphere_4", (0.14, -0.04, -0.06)),
        ("toe", (0.16, 0.0, -0.055)),
    ),
}


def _add_fixed_sphere(
    root: ElementTree.Element,
    link_name: str,
    parent_name: str,
    position: tuple[float, float, float],
) -> None:
    link = ElementTree.SubElement(root, "link", {"name": link_name})
    inertial = ElementTree.SubElement(link, "inertial")
    ElementTree.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ElementTree.SubElement(inertial, "mass", {"value": "0.001"})
    ElementTree.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "0.0000001",
            "ixy": "0",
            "ixz": "0",
            "iyy": "0.0000001",
            "iyz": "0",
            "izz": "0.0000001",
        },
    )
    collision = ElementTree.SubElement(link, "collision")
    ElementTree.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    geometry = ElementTree.SubElement(collision, "geometry")
    ElementTree.SubElement(geometry, "sphere", {"radius": "0.005"})

    joint = ElementTree.SubElement(root, "joint", {"name": f"{link_name}_joint", "type": "fixed"})
    ElementTree.SubElement(
        joint,
        "origin",
        {"xyz": " ".join(f"{value:g}" for value in position), "rpy": "0 0 0"},
    )
    ElementTree.SubElement(joint, "parent", {"link": parent_name})
    ElementTree.SubElement(joint, "child", {"link": link_name})


def main() -> None:
    source_root = ElementTree.parse(SOURCE_URDF).getroot()
    primitive_root = ElementTree.parse(PRIMITIVE_URDF).getroot()
    compiler = source_root.find("./mujoco/compiler")
    if compiler is not None:
        compiler.set("meshdir", ".")
    source_links = {link.attrib["name"]: link for link in source_root.findall("link")}

    for link_name, link in source_links.items():
        if link_name in SPHERE_HAND_LINKS:
            continue
        for collision in list(link.findall("collision")):
            link.remove(collision)

    for primitive_link in primitive_root.findall("link"):
        link_name = primitive_link.attrib["name"]
        if link_name in SKIPPED_PRIMITIVE_LINKS:
            continue
        collisions = primitive_link.findall("collision")
        if not collisions:
            continue
        if link_name not in source_links:
            raise KeyError(f"primitive collision link is missing from sphere-hand model: {link_name}")
        for collision in collisions:
            source_links[link_name].append(deepcopy(collision))

    for side, colliders in FOOT_COLLIDERS.items():
        for suffix, position in colliders:
            _add_fixed_sphere(
                source_root,
                link_name=f"{side}_{suffix}_link",
                parent_name=f"{side}_ankle_x_link",
                position=position,
            )

    collision_shapes = [
        next(iter(collision.find("geometry"))).tag
        for collision in source_root.findall("./link/collision")
    ]
    if len(collision_shapes) != 46:
        raise ValueError(f"expected 46 Holosoma collision shapes, got {len(collision_shapes)}")
    if "mesh" in collision_shapes:
        raise ValueError("Holosoma-aligned training collisions must contain no meshes")

    source_root.insert(
        0,
        ElementTree.Comment(
            " Generated from the sphere-hand model with Holosoma primitive robot collisions. "
        ),
    )
    ElementTree.indent(source_root, space="  ")
    ElementTree.ElementTree(source_root).write(
        OUTPUT_URDF,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    )
    print(f"Wrote {OUTPUT_URDF}")
    print(f"Collision shapes: {len(collision_shapes)} ({sorted(set(collision_shapes))})")


if __name__ == "__main__":
    main()
