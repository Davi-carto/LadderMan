"""Shared contact-label schema for the first LadderMan expert task."""

LIMB_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")

DEFAULT_LIMB_BODY_NAMES = {
    "left_hand": "left_wrist_yaw_link",
    "right_hand": "right_wrist_yaw_link",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
}

CR1_LIMB_BODY_NAMES = {
    "left_hand": "left_sphere_hand_link",
    "right_hand": "right_sphere_hand_link",
    "left_foot": "left_sole_contact_point",
    "right_foot": "right_sole_contact_point",
}

# Hand offsets target the sphere centers: +4.5 cm is the sphere radius above
# the step top. Both hands use small trajectory-derived fore/aft shifts.
# The left sole trajectory remains about 5 cm inboard of the symmetric target.
CR1_CONTACT_TARGET_OFFSETS = {
    "left_hand": (-0.015, 0.25, 0.045),
    "right_hand": (-0.01, -0.25, 0.045),
    "left_foot": (0.0, 0.13, 0.0),
    "right_foot": (0.0, -0.18, 0.0),
}

CR1_CONTACT_DISTANCE_THRESHOLDS = {
    "left_hand": 0.052,
    "right_hand": 0.05,
    "left_foot": 0.145,
    "right_foot": 0.145,
}

# Training uses the same sphere-center hand targets, but keeps foot targets
# symmetric. The left-foot offset above is annotation-only.
CR1_COMMAND_CONTACT_TARGET_OFFSETS = {
    "left_hand": CR1_CONTACT_TARGET_OFFSETS["left_hand"],
    "right_hand": CR1_CONTACT_TARGET_OFFSETS["right_hand"],
    "left_foot": (0.0, 0.18, 0.0),
    "right_foot": (0.0, -0.18, 0.0),
}

# Both ankle_x meshes have their sole bottom center at approximately (0.045, 0, -0.075) m.
CR1_VIRTUAL_BODY_SPECS = {
    "left_sole_contact_point": ("left_ankle_x_link", (0.045, 0.0, -0.075)),
    "right_sole_contact_point": ("right_ankle_x_link", (0.045, 0.0, -0.075)),
}

CR1_INITIAL_SOLE_TARGET_Z = -0.03

# This is the expected Unitree G1 body order for the local URDF used by the Ladder
# extension. Pass explicit limb indexes to the preprocessing script if a motion
# exporter writes body tensors in a different order.
DEFAULT_G1_BODY_NAMES = (
    "pelvis",
    "pelvis_contour_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "logo_link",
    "head_link",
    "imu_in_torso",
    "imu_in_pelvis",
    "mid360_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
    "LL_FOOT",
    "LR_FOOT",
)
