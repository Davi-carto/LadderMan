from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape


PARAMS = {
    "name": "wood_step_ladder",
    "total_height_m": 1.60,
    "front_run_m": 1.04,
    "front_incline_deg": math.degrees(math.atan2(1.60, 1.04)),
    "base_platform_height_m": 0.16,
    "base_platform_width_m": 0.64,
    "base_platform_length_m": 1.04,
    "step_count": 7,
    "step_width_m": 0.64,
    "step_depth_m": 0.13,
    "step_thickness_m": 0.03,
    "step_top_spacing_m": 0.23,
    "step_y_spacing_m": 0.135,
    "side_board_thickness_m": 0.06,
    "side_board_depth_m": 0.14,
    "rear_spread_m": 1.04,
    "rear_leg_width_m": 0.08,
    "rear_leg_depth_m": 0.08,
    "rear_cross_beam_height_m": 0.72,
    "top_post_extra_height_m": 0.22,
    "top_platform_depth_m": 0.13,
    "top_platform_thickness_m": 0.03,
    "origin": "ground projection of the front climbing face",
    "frame": {
        "x": "left-right across the ladder",
        "y": "front-back depth; robot approaches from positive y",
        "z": "world up",
    },
}


def add(a, b):
    return [a[i] + b[i] for i in range(3)]


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def scale(v, s):
    return [x * s for x in v]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def norm(v):
    n = math.sqrt(dot(v, v))
    if n == 0:
        raise ValueError("zero-length vector")
    return [x / n for x in v]


def matvec(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def basis_from_z(z_axis, y_hint=(0.0, 1.0, 0.0)):
    z = norm(z_axis)
    x = norm(cross(y_hint, z))
    y = norm(cross(z, x))
    return [
        [x[0], y[0], z[0]],
        [x[1], y[1], z[1]],
        [x[2], y[2], z[2]],
    ]


IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def matrix_to_rpy(rot):
    sy = math.sqrt(rot[0][0] * rot[0][0] + rot[1][0] * rot[1][0])
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(rot[2][1], rot[2][2])
        pitch = math.atan2(-rot[2][0], sy)
        yaw = math.atan2(rot[1][0], rot[0][0])
    else:
        roll = math.atan2(-rot[1][2], rot[1][1])
        pitch = math.atan2(-rot[2][0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def matrix_to_quat_wxyz(rot):
    trace = rot[0][0] + rot[1][1] + rot[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2][1] - rot[1][2]) / s
        y = (rot[0][2] - rot[2][0]) / s
        z = (rot[1][0] - rot[0][1]) / s
    elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
        s = math.sqrt(1.0 + rot[0][0] - rot[1][1] - rot[2][2]) * 2.0
        w = (rot[2][1] - rot[1][2]) / s
        x = 0.25 * s
        y = (rot[0][1] + rot[1][0]) / s
        z = (rot[0][2] + rot[2][0]) / s
    elif rot[1][1] > rot[2][2]:
        s = math.sqrt(1.0 + rot[1][1] - rot[0][0] - rot[2][2]) * 2.0
        w = (rot[0][2] - rot[2][0]) / s
        x = (rot[0][1] + rot[1][0]) / s
        y = 0.25 * s
        z = (rot[1][2] + rot[2][1]) / s
    else:
        s = math.sqrt(1.0 + rot[2][2] - rot[0][0] - rot[1][1]) * 2.0
        w = (rot[1][0] - rot[0][1]) / s
        x = (rot[0][2] + rot[2][0]) / s
        y = (rot[1][2] + rot[2][1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return w / n, x / n, y / n, z / n


def box_mesh(center, size, rot=IDENTITY):
    hx, hy, hz = [s * 0.5 for s in size]
    local = [
        [-hx, -hy, -hz],
        [hx, -hy, -hz],
        [hx, hy, -hz],
        [-hx, hy, -hz],
        [-hx, -hy, hz],
        [hx, -hy, hz],
        [hx, hy, hz],
        [-hx, hy, hz],
    ]
    vertices = [add(center, matvec(rot, p)) for p in local]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    ]
    return vertices, faces


def triangle_normal(a, b, c):
    return norm(cross(sub(b, a), sub(c, a)))


def front_center_from_height(z, params):
    # Front climbing side leans backward as height increases. The measured
    # right-triangle starts at the front bottom corner of the first step.
    slope_z = z - params["base_platform_height_m"]
    y = -slope_z * params["front_run_m"] / params["total_height_m"]
    return [0.0, y, z]


def step_top_height(index, params):
    return params["base_platform_height_m"] + params["step_thickness_m"] + index * params["step_top_spacing_m"]


def step_front_y(index, params):
    return -index * params["step_y_spacing_m"]


def step_center_y(index, params):
    return step_front_y(index, params) - params["step_depth_m"] / 2.0


def slope_triangle_points(params):
    x = params["step_width_m"] / 2.0 + params["side_board_thickness_m"] + 0.08
    base_z = params["base_platform_height_m"]
    top_z = base_z + params["total_height_m"]
    front = [x, 0.0, base_z]
    rear = [x, -params["front_run_m"], base_z]
    top = [x, -params["front_run_m"], top_z]
    return front, rear, top


def rod_between(name, start, end, thickness, material):
    delta = sub(end, start)
    length = math.sqrt(dot(delta, delta))
    return {
        "name": name,
        "center": scale(add(start, end), 0.5),
        "size": [thickness, thickness, length],
        "rot": basis_from_z(delta, y_hint=(1.0, 0.0, 0.0)),
        "material": material,
    }


def slope_triangle_visual_helpers(params):
    front, rear, top = slope_triangle_points(params)
    thickness = 0.018
    return [
        rod_between("slope_triangle_run_green", front, rear, thickness, "helper_run"),
        rod_between("slope_triangle_height_blue", rear, top, thickness, "helper_height"),
        rod_between("slope_triangle_hypotenuse_red", front, top, thickness, "helper_slope"),
    ]


def ladder_parts(params):
    parts = []
    width = params["step_width_m"]
    depth = params["step_depth_m"]
    thickness = params["step_thickness_m"]
    count = params["step_count"]
    total_h = params["total_height_m"]
    base_h = params["base_platform_height_m"]
    top_z = base_h + total_h
    side_board_x = width / 2.0 + params["side_board_thickness_m"] / 2.0
    front_top = front_center_from_height(top_z, params)
    front_bottom = front_center_from_height(base_h, params)
    front_dir = norm(sub(front_top, front_bottom))
    front_rot = basis_from_z(front_dir)

    first_step_top = step_top_height(0, params)
    parts.append(
        {
            "name": "base_platform",
            "center": [0.0, -params["base_platform_length_m"] / 2.0, params["base_platform_height_m"] / 2.0],
            "size": [params["base_platform_width_m"], params["base_platform_length_m"], params["base_platform_height_m"]],
            "rot": IDENTITY,
            "material": "wood_rear",
        }
    )

    # Seven horizontal wide steps. Their top surfaces are vertically spaced by
    # step_top_spacing_m; the first step bottom sits on the base platform.
    for i in range(count):
        step_top_z = step_top_height(i, params)
        center = [0.0, step_center_y(i, params), step_top_z - thickness / 2.0]
        parts.append(
            {
                "name": f"step_{i + 1:02d}",
                "center": center,
                "size": [width, depth, thickness],
                "rot": IDENTITY,
                "material": "wood_step",
            }
        )

    # Slanted side boards on the climbing side. These are simple solids; the visual
    # stair-step shape comes from the protruding treads mounted on them.
    rail_mid = scale(add(front_top, front_bottom), 0.5)
    rail_len = math.sqrt(dot(sub(front_top, front_bottom), sub(front_top, front_bottom))) + 0.40
    for side, x in [("left", -side_board_x), ("right", side_board_x)]:
        center = [x, rail_mid[1], rail_mid[2]]
        parts.append(
            {
                "name": f"front_zigzag_side_board_{side}",
                "center": center,
                "size": [params["side_board_thickness_m"], params["side_board_depth_m"], rail_len],
                "rot": front_rot,
                "material": "wood_side",
            }
        )

    # Top platform / cross beam.
    top_center = front_center_from_height(top_z, params)
    top_center[1] -= params["top_platform_depth_m"] * 0.15
    top_center[2] = top_z - params["top_platform_thickness_m"] / 2.0
    parts.append(
        {
            "name": "top_platform",
            "center": top_center,
            "size": [width + 0.12, params["top_platform_depth_m"], params["top_platform_thickness_m"]],
            "rot": IDENTITY,
            "material": "wood_step",
        }
    )

    # High wooden posts above both sides.
    post_height = params["top_post_extra_height_m"]
    for side, x in [("left", -side_board_x), ("right", side_board_x)]:
        post_center = [x, top_center[1], total_h + post_height / 2.0]
        post_center[2] = top_z + post_height / 2.0
        parts.append(
            {
                "name": f"top_high_post_{side}",
                "center": post_center,
                "size": [0.09, 0.12, post_height],
                "rot": IDENTITY,
                "material": "wood_side",
            }
        )

    # Rear A-frame support legs.
    rear_bottom_y = front_center_from_height(base_h, params)[1] - params["rear_spread_m"]
    rear_bottom_z = 0.02
    rear_top = [0.0, top_center[1], top_z]
    rear_bottom = [0.0, rear_bottom_y, rear_bottom_z]
    rear_dir = norm(sub(rear_top, rear_bottom))
    rear_rot = basis_from_z(rear_dir)
    rear_len = math.sqrt(dot(sub(rear_top, rear_bottom), sub(rear_top, rear_bottom))) + 0.16
    rear_mid = scale(add(rear_top, rear_bottom), 0.5)
    rear_side_x = width / 2.0 + 0.05
    for side, x in [("left", -rear_side_x), ("right", rear_side_x)]:
        parts.append(
            {
                "name": f"rear_support_leg_{side}",
                "center": [x, rear_mid[1], rear_mid[2]],
                "size": [params["rear_leg_width_m"], params["rear_leg_depth_m"], rear_len],
                "rot": rear_rot,
                "material": "wood_rear",
            }
        )

    # Rear middle cross beam.
    beam_h = params["rear_cross_beam_height_m"]
    rear_beam_center_y = rear_bottom_y + (top_center[1] - rear_bottom_y) * (beam_h / total_h)
    parts.append(
        {
            "name": "rear_middle_cross_beam",
            "center": [0.0, rear_beam_center_y, beam_h],
            "size": [width + 0.12, 0.07, 0.08],
            "rot": IDENTITY,
            "material": "wood_rear",
        }
    )

    # Side limit bars, simplified as rectangular rods.
    chain_z = 0.82
    front_chain_y = front_center_from_height(chain_z, params)[1]
    rear_chain_y = rear_bottom_y + (top_center[1] - rear_bottom_y) * (chain_z / total_h)
    chain_len = abs(front_chain_y - rear_chain_y)
    for side, x in [("left", -rear_side_x), ("right", rear_side_x)]:
        parts.append(
            {
                "name": f"side_limit_bar_{side}",
                "center": [x, (front_chain_y + rear_chain_y) / 2.0, chain_z],
                "size": [0.025, chain_len, 0.025],
                "rot": IDENTITY,
                "material": "metal",
            }
        )

    params["derived"] = {
        "front_top_y_m": front_center_from_height(top_z, params)[1],
        "front_triangle_bottom_m": [0.0, 0.0, base_h],
        "front_triangle_rear_m": [0.0, -params["front_run_m"], base_h],
        "front_triangle_top_m": [0.0, -params["front_run_m"], top_z],
        "rear_bottom_y_m": rear_bottom_y,
        "top_platform_center_m": top_center,
        "first_step_top_height_m": first_step_top,
        "highest_step_top_height_m": step_top_height(count - 1, params),
        "highest_step_center_height_m": step_top_height(count - 1, params) - thickness / 2.0,
        "step_top_heights_m": [step_top_height(i, params) for i in range(count)],
        "step_front_y_positions_m": [step_front_y(i, params) for i in range(count)],
        "step_center_y_positions_m": [step_center_y(i, params) for i in range(count)],
        "total_depth_m": abs(rear_bottom_y) + depth,
        "front_slope_length_m": math.hypot(params["front_run_m"], total_h),
    }
    return parts


def write_yaml(path, params):
    derived = params["derived"]
    step_top_heights = ", ".join(f"{height:.6f}" for height in derived["step_top_heights_m"])
    step_front_y_positions = ", ".join(f"{y:.6f}" for y in derived["step_front_y_positions_m"])
    step_center_y_positions = ", ".join(f"{y:.6f}" for y in derived["step_center_y_positions_m"])
    triangle_bottom = ", ".join(f"{v:.6f}" for v in derived["front_triangle_bottom_m"])
    triangle_rear = ", ".join(f"{v:.6f}" for v in derived["front_triangle_rear_m"])
    triangle_top = ", ".join(f"{v:.6f}" for v in derived["front_triangle_top_m"])
    lines = [
        f"name: {params['name']}",
        f"total_height_m: {params['total_height_m']}",
        f"front_run_m: {params['front_run_m']}",
        f"front_incline_deg: {params['front_incline_deg']}",
        f"base_platform_height_m: {params['base_platform_height_m']}",
        f"base_platform_width_m: {params['base_platform_width_m']}",
        f"base_platform_length_m: {params['base_platform_length_m']}",
        f"step_count: {params['step_count']}",
        f"step_width_m: {params['step_width_m']}",
        f"step_depth_m: {params['step_depth_m']}",
        f"step_thickness_m: {params['step_thickness_m']}",
        f"step_top_spacing_m: {params['step_top_spacing_m']}",
        f"step_y_spacing_m: {params['step_y_spacing_m']}",
        f"rear_spread_m: {params['rear_spread_m']}",
        f"origin: {params['origin']}",
        "frame:",
        f"  x: \"{params['frame']['x']}\"",
        f"  y: \"{params['frame']['y']}\"",
        f"  z: \"{params['frame']['z']}\"",
        "derived:",
        f"  front_top_y_m: {derived['front_top_y_m']:.6f}",
        f"  front_triangle_bottom_m: [{triangle_bottom}]",
        f"  front_triangle_rear_m: [{triangle_rear}]",
        f"  front_triangle_top_m: [{triangle_top}]",
        f"  rear_bottom_y_m: {derived['rear_bottom_y_m']:.6f}",
        f"  first_step_top_height_m: {derived['first_step_top_height_m']:.6f}",
        f"  highest_step_top_height_m: {derived['highest_step_top_height_m']:.6f}",
        f"  highest_step_center_height_m: {derived['highest_step_center_height_m']:.6f}",
        f"  front_slope_length_m: {derived['front_slope_length_m']:.6f}",
        f"  total_depth_m: {derived['total_depth_m']:.6f}",
        f"  step_top_heights_m: [{step_top_heights}]",
        f"  step_front_y_positions_m: [{step_front_y_positions}]",
        f"  step_center_y_positions_m: [{step_center_y_positions}]",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stl(path, parts):
    lines = ["solid wood_step_ladder"]
    for part in parts:
        vertices, faces = box_mesh(part["center"], part["size"], part["rot"])
        for face in faces:
            a, b, c = [vertices[idx] for idx in face]
            n = triangle_normal(a, b, c)
            lines.append(f"  facet normal {n[0]:.8e} {n[1]:.8e} {n[2]:.8e}")
            lines.append("    outer loop")
            for p in (a, b, c):
                lines.append(f"      vertex {p[0]:.8e} {p[1]:.8e} {p[2]:.8e}")
            lines.append("    endloop")
            lines.append("  endfacet")
    lines.append("endsolid wood_step_ladder")
    path.write_text("\n".join(lines), encoding="ascii")


def write_obj(path, mtl_path, parts):
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl wood_step",
                "Kd 0.86 0.55 0.25",
                "newmtl wood_side",
                "Kd 0.78 0.45 0.20",
                "newmtl wood_rear",
                "Kd 0.70 0.40 0.18",
                "newmtl metal",
                "Kd 0.55 0.55 0.55",
                "",
            ]
        ),
        encoding="ascii",
    )
    lines = [f"mtllib {mtl_path.name}"]
    vertex_offset = 1
    for part in parts:
        vertices, faces = box_mesh(part["center"], part["size"], part["rot"])
        lines.append(f"o {part['name']}")
        lines.append(f"usemtl {part['material']}")
        for p in vertices:
            lines.append(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}")
        for face in faces:
            i, j, k = [idx + vertex_offset for idx in face]
            lines.append(f"f {i} {j} {k}")
        vertex_offset += len(vertices)
    path.write_text("\n".join(lines), encoding="ascii")


def write_usda(path, parts):
    points = []
    face_counts = []
    face_indices = []
    for part in parts:
        vertices, faces = box_mesh(part["center"], part["size"], part["rot"])
        offset = len(points)
        points.extend(vertices)
        for face in faces:
            face_counts.append(3)
            face_indices.extend([idx + offset for idx in face])

    point_lines = [f"        ({p[0]:.8f}, {p[1]:.8f}, {p[2]:.8f})" for p in points]
    path.write_text(
        "\n".join(
            [
                "#usda 1.0",
                "(",
                '    defaultPrim = "World"',
                "    metersPerUnit = 1",
                '    upAxis = "Z"',
                ")",
                "",
                'def Xform "World"',
                "{",
                '    def Xform "Wood_Step_Ladder"',
                "    {",
                '        def Mesh "mesh"',
                "        {",
                "            point3f[] points = [",
                ",\n".join(point_lines),
                "            ]",
                f"            int[] faceVertexCounts = {face_counts}",
                f"            int[] faceVertexIndices = {face_indices}",
                '            uniform token subdivisionScheme = "none"',
                "        }",
                "    }",
                "}",
                "",
            ]
        ),
        encoding="ascii",
    )


def write_sim_urdf(path, parts):
    material_rgba = {
        "wood_step": "0.86 0.55 0.25 1.0",
        "wood_side": "0.78 0.45 0.20 1.0",
        "wood_rear": "0.70 0.40 0.18 1.0",
        "metal": "0.55 0.55 0.55 1.0",
    }
    lines = [
        '<?xml version="1.0"?>',
        '<robot name="wood_step_ladder_sim">',
        "  <!-- Simulation-friendly fixed obstacle: primitive box visuals and collisions. -->",
        "  <!-- Load with fix_base_link=True and disable_gravity=True in Isaac Gym. -->",
    ]
    for material, rgba in material_rgba.items():
        lines.extend(
            [
                f'  <material name="{material}">',
                f'    <color rgba="{rgba}"/>',
                "  </material>",
            ]
        )
    lines.extend(
        [
            '  <link name="base">',
            '    <inertial>',
            '      <origin xyz="0 -0.55 0.90" rpy="0 0 0"/>',
            '      <mass value="1000.0"/>',
            '      <inertia ixx="250.0" ixy="0.0" ixz="0.0" iyy="250.0" iyz="0.0" izz="250.0"/>',
            '    </inertial>',
        ]
    )
    for part in parts:
        name = escape(safe_name(part["name"]))
        x, y, z = part["center"]
        sx, sy, sz = part["size"]
        roll, pitch, yaw = matrix_to_rpy(part["rot"])
        material = escape(part["material"])
        lines.extend(
            [
                f'    <visual name="{name}_visual">',
                f'      <origin xyz="{x:.8f} {y:.8f} {z:.8f}" rpy="{roll:.8f} {pitch:.8f} {yaw:.8f}"/>',
                "      <geometry>",
                f'        <box size="{sx:.8f} {sy:.8f} {sz:.8f}"/>',
                "      </geometry>",
                f'      <material name="{material}"/>',
                "    </visual>",
                f'    <collision name="{name}_collision">',
                f'      <origin xyz="{x:.8f} {y:.8f} {z:.8f}" rpy="{roll:.8f} {pitch:.8f} {yaw:.8f}"/>',
                "      <geometry>",
                f'        <box size="{sx:.8f} {sy:.8f} {sz:.8f}"/>',
                "      </geometry>",
                "    </collision>",
            ]
        )
    lines.extend(
        [
            "  </link>",
            '  <gazebo>',
            '    <static>true</static>',
            '  </gazebo>',
            "</robot>",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="ascii")


def write_mesh_urdf(path):
    lines = [
        '<?xml version="1.0"?>',
        '<robot name="wood_step_ladder">',
        '  <!-- Mesh wrapper for quick preview/CAD import. Use wood_step_ladder_sim.urdf for Isaac Gym contacts. -->',
        '  <link name="base">',
        '    <visual>',
        '      <origin xyz="0 0 0" rpy="0 0 0"/>',
        '      <geometry>',
        '        <mesh filename="wood_step_ladder.stl" scale="1 1 1"/>',
        '      </geometry>',
        '      <material name="wood">',
        '        <color rgba="0.78 0.48 0.22 1.0"/>',
        '      </material>',
        '    </visual>',
        '    <collision>',
        '      <origin xyz="0 0 0" rpy="0 0 0"/>',
        '      <geometry>',
        '        <mesh filename="wood_step_ladder.stl" scale="1 1 1"/>',
        '      </geometry>',
        '    </collision>',
        '    <inertial>',
        '      <origin xyz="0 -0.55 0.9" rpy="0 0 0"/>',
        '      <mass value="100.0"/>',
        '      <inertia ixx="10.0" ixy="0.0" ixz="0.0" iyy="10.0" iyz="0.0" izz="10.0"/>',
        '    </inertial>',
        '  </link>',
        '</robot>',
        '',
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def write_physics_usda(path, parts, helper_parts):
    colors = {
        "wood_step": (0.86, 0.55, 0.25),
        "wood_side": (0.78, 0.45, 0.20),
        "wood_rear": (0.70, 0.40, 0.18),
        "metal": (0.55, 0.55, 0.55),
        "helper_run": (0.0, 0.9, 0.1),
        "helper_height": (0.1, 0.35, 1.0),
        "helper_slope": (1.0, 0.05, 0.05),
    }
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def Xform "Wood_Step_Ladder"',
        "    {",
        "        custom bool ladder:isStaticObstacle = 1",
        '        custom string ladder:loadHint = "Static primitive colliders. Keep the ladder immovable during training."',
        "",
    ]
    for part in parts:
        name = safe_name(part["name"])
        x, y, z = part["center"]
        sx, sy, sz = [v * 0.5 for v in part["size"]]
        qw, qx, qy, qz = matrix_to_quat_wxyz(part["rot"])
        color = colors.get(part["material"], (0.8, 0.8, 0.8))
        lines.extend(
            [
                f'        def Cube "{name}" (',
                '            prepend apiSchemas = ["PhysicsCollisionAPI"]',
                "        )",
                "        {",
                "            double size = 2",
                "            bool physics:collisionEnabled = 1",
                f"            color3f[] primvars:displayColor = [({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})]",
                f"            double3 xformOp:translate = ({x:.8f}, {y:.8f}, {z:.8f})",
                f"            quatf xformOp:orient = ({qw:.8f}, {qx:.8f}, {qy:.8f}, {qz:.8f})",
                f"            double3 xformOp:scale = ({sx:.8f}, {sy:.8f}, {sz:.8f})",
                '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]',
                "        }",
                "",
            ]
        )
    lines.extend(
        [
            "        # Visual-only measurement guides for the front right-triangle.",
            "        # These helpers have no PhysicsCollisionAPI and do not affect contacts.",
            "",
        ]
    )
    for part in helper_parts:
        name = safe_name(part["name"])
        x, y, z = part["center"]
        sx, sy, sz = [v * 0.5 for v in part["size"]]
        qw, qx, qy, qz = matrix_to_quat_wxyz(part["rot"])
        color = colors[part["material"]]
        lines.extend(
            [
                f'        def Cube "{name}"',
                "        {",
                "            double size = 2",
                f"            color3f[] primvars:displayColor = [({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})]",
                f"            double3 xformOp:translate = ({x:.8f}, {y:.8f}, {z:.8f})",
                f"            quatf xformOp:orient = ({qw:.8f}, {qx:.8f}, {qy:.8f}, {qz:.8f})",
                f"            double3 xformOp:scale = ({sx:.8f}, {sy:.8f}, {sz:.8f})",
                '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]',
                "        }",
                "",
            ]
        )
    lines.extend(["    }", "}", ""])
    path.write_text("\n".join(lines), encoding="ascii")


def write_scad(path, p):
    text = f"""// Wood step ladder parametric OpenSCAD model
// Units: meters

total_h = {p['total_height_m']};
front_run = {p['front_run_m']};
base_h = {p['base_platform_height_m']};
base_w = {p['base_platform_width_m']};
base_l = {p['base_platform_length_m']};
step_count = {p['step_count']};
step_w = {p['step_width_m']};
step_d = {p['step_depth_m']};
step_t = {p['step_thickness_m']};
step_top_spacing = {p['step_top_spacing_m']};
step_y_spacing = {p['step_y_spacing_m']};
rear_spread = {p['rear_spread_m']};
side_t = {p['side_board_thickness_m']};
side_d = {p['side_board_depth_m']};
top_d = {p['top_platform_depth_m']};
top_t = {p['top_platform_thickness_m']};
rear_beam_h = {p['rear_cross_beam_height_m']};
top_z = base_h + total_h;

module box(size) {{ cube(size, center=true); }}

function y_from_z(z) = -(z - base_h) * front_run / total_h;
function step_top_z(i) = base_h + step_t + i * step_top_spacing;
function step_front_y(i) = -i * step_y_spacing;

translate([0, -base_l / 2, base_h / 2])
    box([base_w, base_l, base_h]);

for (i = [0:step_count - 1]) {{
    step_z = step_top_z(i);
    center_z = step_z - step_t / 2;
    translate([0, step_front_y(i) - step_d / 2, center_z])
        box([step_w, step_d, step_t]);
}}

front_phi = atan2(total_h, front_run);
rail_len = sqrt(pow(total_h, 2) + pow(front_run, 2)) + 0.4;
rail_mid_y = y_from_z(top_z) / 2;
rail_mid_z = (base_h + top_z) / 2;

for (x = [-step_w/2 - side_t/2, step_w/2 + side_t/2]) {{
    translate([x, rail_mid_y, rail_mid_z])
        rotate([90-front_phi, 0, 0])
            box([side_t, side_d, rail_len]);
}}

top_y = y_from_z(top_z);
translate([0, top_y - top_d * 0.15, top_z - top_t / 2])
    box([step_w + 0.12, top_d, top_t]);

for (x = [-step_w/2 - side_t/2, step_w/2 + side_t/2]) {{
    translate([x, top_y, top_z + 0.11])
        box([0.09, 0.12, 0.22]);
}}

rear_y = y_from_z(base_h) - rear_spread;
rear_mid_y = (rear_y + top_y) / 2;
rear_mid_z = (0.02 + top_z) / 2;
rear_angle = atan2(top_z - 0.02, top_y - rear_y);

for (x = [-step_w/2 - 0.05, step_w/2 + 0.05]) {{
    translate([x, rear_mid_y, rear_mid_z])
        rotate([90-rear_angle, 0, 0])
            box([0.08, 0.08, (top_z - 0.02) / sin(rear_angle) + 0.16]);
}}

translate([0, rear_y + (top_y - rear_y) * (rear_beam_h / total_h), rear_beam_h])
    box([step_w + 0.12, 0.07, 0.08]);

// Visual-only measurement triangle: green run, blue height, red hypotenuse.
translate([step_w / 2 + side_t + 0.08, -front_run / 2, base_h])
    color("green") box([0.018, front_run, 0.018]);
translate([step_w / 2 + side_t + 0.08, -front_run, base_h + total_h / 2])
    color("blue") box([0.018, 0.018, total_h]);
translate([step_w / 2 + side_t + 0.08, -front_run / 2, base_h + total_h / 2])
    rotate([90-front_phi, 0, 0])
        color("red") box([0.018, 0.018, sqrt(pow(total_h, 2) + pow(front_run, 2))]);
"""
    path.write_text(text, encoding="ascii")


def write_notes(path, params):
    d = params["derived"]
    text = f"""# 木制人字步梯模型说明

这是参照用户提供图片生成的木制人字步梯简化模型，用于 LadderMan 第一阶段动捕、仿真和接触目标设计。

## 文件

- `wood_step_ladder.yaml`: 参数源文件。
- `wood_step_ladder.stl`: 通用三维预览文件。
- `wood_step_ladder.obj` / `wood_step_ladder.mtl`: 通用三维模型，带简单材质。
- `wood_step_ladder.usda`: IsaacSim/IsaacLab 可导入的 USD ASCII Mesh。
- `wood_step_ladder_physics.usda`: IsaacSim/IsaacLab 仿真专用 USD，使用多个 primitive box collider。
- `wood_step_ladder_sim.urdf`: Isaac Gym 仿真专用 URDF，使用多个 primitive box collider。
- `wood_step_ladder.scad`: OpenSCAD 参数化模型，方便检查和修改。
- `generate_wood_step_ladder_model.py`: 生成脚本。

## 主要参数

- 底座平台高度: {params['base_platform_height_m']} m
- 底座平台宽度: {params['base_platform_width_m']} m
- 底座平台长度: {params['base_platform_length_m']} m
- 前侧斜面直角三角形高度: {params['total_height_m']} m
- 前侧斜面直角三角形水平长: {params['front_run_m']} m
- 前侧斜面长度: {d['front_slope_length_m']:.3f} m
- 前侧踏板数: {params['step_count']}
- 踏板宽度: {params['step_width_m']} m
- 踏板深度: {params['step_depth_m']} m
- 踏板厚度: {params['step_thickness_m']} m
- 相邻踏板顶部高度差: {params['step_top_spacing_m']} m
- 相邻踏板前缘 y 方向距离: {params['step_y_spacing_m']} m
- 前侧倾角: {params['front_incline_deg']} deg
- 后侧支撑腿展开: {params['rear_spread_m']} m
- 第一阶踏板顶部高度: {d['first_step_top_height_m']:.3f} m
- 最高踏板顶部高度: {d['highest_step_top_height_m']:.3f} m
- 顶部平台 y 坐标约: {d['front_top_y_m']:.3f} m
- 后脚 y 坐标约: {d['rear_bottom_y_m']:.3f} m
- 整体深度约: {d['total_depth_m']:.3f} m
- 斜面测量三角形: 底部前点 `(0, 0, {params['base_platform_height_m']})`，后点 `(0, -{params['front_run_m']}, {params['base_platform_height_m']})`，顶点 `(0, -{params['front_run_m']}, {params['base_platform_height_m'] + params['total_height_m']})`
- `wood_step_ladder_physics.usda` 里有三条无碰撞彩色辅助线：绿色为 1.04m 水平边，蓝色为 1.6m 竖直边，红色为斜边。

## 坐标系

原点在前侧斜面底部的地面投影点。

- x 轴: 梯子左右方向。
- y 轴: 梯子前后方向；机器人从正 y 方向接近，后支撑腿朝负 y 方向展开。
- z 轴: 世界竖直向上。

第 i 级踏板中心近似为：

```text
top_z_i = base_platform_height + step_thickness + i * step_top_spacing
center_z_i = top_z_i - step_thickness / 2
front_y_i = -i * step_y_spacing
center_y_i = front_y_i - step_depth / 2
x_i = 0
```

其中 i 从 0 到 6。第一阶踏板前缘与底座平台前缘对齐在 `y=0`；相邻踏板前缘沿 y 方向相差 `0.135 m`。训练时可把踏板中心或踏板前缘作为 foot/hand contact target。

## 建模取舍

模型用规则 cuboid 组合，优先保证仿真碰撞和尺寸清晰。没有做木纹、螺栓、真实链条和倒角。后续如果用于真实加工，需要朋友根据材料强度、连接件、倒角和安全裕度重新出工程图。

## 仿真使用建议

不要把 `wood_step_ladder.stl` 当作训练碰撞体使用；单个非凸 mesh 在足端/手端接触时更容易抖动或穿透。

Isaac Gym 推荐加载 `wood_step_ladder_sim.urdf`，并固定底座：

```python
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
ladder_asset = gym.load_asset(sim, asset_root, "wood_step_ladder_sim.urdf", asset_options)
```

IsaacLab / IsaacSim 推荐加载 `wood_step_ladder_physics.usda`，该文件由多个 `Cube` primitive 组成，并给每个部件添加 `PhysicsCollisionAPI`。它作为静态碰撞障碍物使用，不添加刚体 API，因此不会被机器人撞倒。
"""
    path.write_text(text, encoding="utf-8")


def main():
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = ladder_parts(PARAMS)
    helper_parts = slope_triangle_visual_helpers(PARAMS)
    write_yaml(out_dir / "wood_step_ladder.yaml", PARAMS)
    write_stl(out_dir / "wood_step_ladder.stl", parts)
    write_obj(out_dir / "wood_step_ladder.obj", out_dir / "wood_step_ladder.mtl", parts)
    write_usda(out_dir / "wood_step_ladder.usda", parts)
    write_mesh_urdf(out_dir / "wood_step_ladder.urdf")
    write_sim_urdf(out_dir / "wood_step_ladder_sim.urdf", parts)
    write_physics_usda(out_dir / "wood_step_ladder_physics.usda", parts, helper_parts)
    write_scad(out_dir / "wood_step_ladder.scad", PARAMS)
    write_notes(out_dir / "wood_step_ladder_notes.md", PARAMS)
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
