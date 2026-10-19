import math
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_URDF = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand_holosoma_collision.urdf"
)
HOLOSOMA_SCENE = (
    REPO_ROOT
    / "holosoma"
    / "src"
    / "holosoma_retargeting"
    / "holosoma_retargeting"
    / "demo_data"
    / "climb_ladder"
    / "climing_ladder_mirror_x_trim_v150_650_down008"
    / "CR1-B2-pro_free_wrist_fixed_spherehand_w_ladder.xml"
)


def _numbers(value: str, count: int) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    assert values.shape == (count,)
    return values


def _rpy_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = 0.5 * rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _canonical_quat(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat)
    first_nonzero = np.flatnonzero(np.abs(quat) > 1.0e-12)
    return -quat if first_nonzero.size and quat[first_nonzero[0]] < 0.0 else quat


def _urdf_collision_records() -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    root = ElementTree.parse(TRAINING_URDF).getroot()
    records = []
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            position = _numbers(origin.attrib.get("xyz", "0 0 0"), 3)
            quat = _rpy_to_quat_wxyz(_numbers(origin.attrib.get("rpy", "0 0 0"), 3))
            shape = next(iter(collision.find("geometry")))
            if shape.tag == "box":
                size = 0.5 * _numbers(shape.attrib["size"], 3)
            elif shape.tag == "cylinder":
                size = np.asarray([float(shape.attrib["radius"]), 0.5 * float(shape.attrib["length"])])
            elif shape.tag == "sphere":
                size = np.asarray([float(shape.attrib["radius"])])
            else:
                raise AssertionError(f"unexpected training collision type: {shape.tag}")
            records.append((link.attrib["name"], shape.tag, size, position, _canonical_quat(quat)))
    return records


def _holosoma_collision_records() -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    root = ElementTree.parse(HOLOSOMA_SCENE).getroot()
    robot_root = root.find("./worldbody/body[@name='base_link']")
    records = []

    def visit(body: ElementTree.Element) -> None:
        for geom in body.findall("geom"):
            if geom.attrib.get("contype", "1") == "0" or geom.attrib.get("conaffinity", "1") == "0":
                continue
            shape = geom.attrib.get("type", "sphere")
            size_count = {"box": 3, "cylinder": 2, "sphere": 1}[shape]
            records.append(
                (
                    body.attrib["name"],
                    shape,
                    _numbers(geom.attrib["size"], size_count),
                    _numbers(geom.attrib.get("pos", "0 0 0"), 3),
                    _canonical_quat(_numbers(geom.attrib.get("quat", "1 0 0 0"), 4)),
                )
            )
        for child in body.findall("body"):
            visit(child)

    visit(robot_root)
    return records


def _urdf_link_poses(link_names: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = ElementTree.parse(TRAINING_URDF).getroot()
    joints_by_child = {joint.find("child").attrib["link"]: joint for joint in root.findall("joint")}
    poses = {}
    for link_name in link_names - {"base_link"}:
        origin = joints_by_child[link_name].find("origin")
        poses[link_name] = (
            _numbers(origin.attrib.get("xyz", "0 0 0"), 3),
            _canonical_quat(_rpy_to_quat_wxyz(_numbers(origin.attrib.get("rpy", "0 0 0"), 3))),
        )
    return poses


def _holosoma_link_poses(link_names: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = ElementTree.parse(HOLOSOMA_SCENE).getroot()
    robot_root = root.find("./worldbody/body[@name='base_link']")
    poses = {}

    def visit(body: ElementTree.Element) -> None:
        link_name = body.attrib["name"]
        if link_name in link_names and link_name != "base_link":
            poses[link_name] = (
                _numbers(body.attrib.get("pos", "0 0 0"), 3),
                _canonical_quat(_numbers(body.attrib.get("quat", "1 0 0 0"), 4)),
            )
        for child in body.findall("body"):
            visit(child)

    visit(robot_root)
    return poses


def test_training_collision_shapes_match_holosoma_retarget_scene():
    training = sorted(_urdf_collision_records(), key=lambda record: record[0])
    holosoma = sorted(_holosoma_collision_records(), key=lambda record: record[0])

    assert len(training) == len(holosoma) == 46
    for training_record, holosoma_record in zip(training, holosoma, strict=True):
        assert training_record[:2] == holosoma_record[:2]
        for training_value, holosoma_value in zip(training_record[2:], holosoma_record[2:], strict=True):
            assert np.allclose(training_value, holosoma_value, atol=1.0e-5)

    collision_link_names = {record[0] for record in training}
    training_poses = _urdf_link_poses(collision_link_names)
    holosoma_poses = _holosoma_link_poses(collision_link_names)
    assert training_poses.keys() == holosoma_poses.keys()
    for link_name in training_poses:
        for training_value, holosoma_value in zip(
            training_poses[link_name], holosoma_poses[link_name], strict=True
        ):
            assert np.allclose(training_value, holosoma_value, atol=1.0e-5)
