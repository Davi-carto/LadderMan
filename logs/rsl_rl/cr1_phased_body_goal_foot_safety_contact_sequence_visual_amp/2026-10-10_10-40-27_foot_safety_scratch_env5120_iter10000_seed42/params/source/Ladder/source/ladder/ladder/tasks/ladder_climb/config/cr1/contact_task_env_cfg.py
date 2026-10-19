"""Oracle limb-placement PPO + AMP without online reference tracking."""

from copy import deepcopy

from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp
from ladder.tasks.ladder_climb.mdp.contact_task import (
    ContactTaskCommandCfg, contact_task_observation, contact_task_critic,
    contact_task_reward, contact_task_done,
)
from .wood_step_end_to_end_visual_ppo_env_cfg import (
    CR1EndToEndVisualOracleContactGoalAMPPPOEnvCfg, CR1AMPStyleObservationCfg,
    CR1EndToEndVisualInputsCfg, CR1StopCommandTransformerVisionDepthCfg,
)


@configclass
class ContactTaskObsCfg(ObsGroup):
    state = ObsTerm(func=contact_task_observation)
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ContactTaskCriticCfg(ContactTaskObsCfg):
    state = ObsTerm(func=contact_task_critic)


@configclass
class ContactTaskObservationsCfg:
    transformer_vision_depth = CR1StopCommandTransformerVisionDepthCfg()
    transformer_vision_inputs = CR1EndToEndVisualInputsCfg()
    contact_task = ContactTaskObsCfg()
    critic = ContactTaskCriticCfg()
    amp_style = CR1AMPStyleObservationCfg()


def _reward(name, weight):
    return RewTerm(func=contact_task_reward, weight=weight, params={"term": name})


@configclass
class ContactTaskRewardsCfg:
    progress = _reward("progress", 2.0)  # 2 reward units / meter of net approach
    reach = _reward("reach", 0.5)
    contact = _reward("contact", 1.0)
    support_match = _reward("support_match", 0.5)
    support_loss = _reward("support_loss", -1.0)
    slip = _reward("slip", -0.2)
    wrong_contact = _reward("wrong_contact", -1.0)
    danger = _reward("danger", -2.0)
    completion = _reward("completion", 5.0)  # One event, not per second
    failure = _reward("failure", -2.0)


@configclass
class ContactTaskTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_success = DoneTerm(func=contact_task_done, params={"success": True})
    task_failure = DoneTerm(func=contact_task_done, params={"success": False})


@configclass
class CR1ContactTaskAMPEnvCfg(CR1EndToEndVisualOracleContactGoalAMPPPOEnvCfg):
    """Random-frame reset followed by successive support goals through motion end."""

    def __post_init__(self):
        super().__post_init__()
        inherited = self.commands.motion
        motion = ContactTaskCommandCfg()
        for name, value in vars(inherited).items():
            if name != "class_type":
                setattr(motion, name, deepcopy(value))
        self.commands.motion = motion
        motion.advance_reference = False
        motion.motion_semantic_progress_enabled = False
        motion.stop_command_training_enabled = False
        motion.stop_hold_reset_probability = 0.0
        motion.stop_seek_episode_probability = 0.0
        # Nominal geometry and small stable resets isolate the task hypothesis.
        motion.ladder_inclination_delta_range = (0.0, 0.0)
        motion.ladder_spacing_delta_range = (0.0, 0.0)
        motion.pose_range = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        motion.pose_range["z"] = (0.0, 0.005)
        motion.fixed_start_pose_range = dict(motion.pose_range)
        motion.velocity_range = {k: (0.0, 0.0) for k in motion.pose_range}
        motion.joint_position_range = (-0.005, 0.005)
        previous = self.rewards
        self.rewards = ContactTaskRewardsCfg()
        # Keep only phase-independent control regularizers; no reference terms.
        for name in ("action_rate_l2", "joint_limit", "joint_torque_l2", "joint_acc_l2_upper",
                     "joint_acc_l2_lower", "action_second_difference_l2_lower"):
            if getattr(previous, name, None) is not None:
                setattr(self.rewards, name, deepcopy(getattr(previous, name)))
        self.observations = ContactTaskObservationsCfg()
        self.terminations = ContactTaskTerminationsCfg()
        # Per-goal deadlines total at most 162.82 s for this motion. Intermediate
        # completion switches goals in place; only the final goal ends success.
        motion.single_task_episode = False
        self.episode_length_s = 170.0
        self.events.root_velocity_push = None


@configclass
class CR1ContactSequenceAMPEnvCfg(CR1ContactTaskAMPEnvCfg):
    """Explicit sequence task name; ContactTask now uses the same behavior."""


@configclass
class PhasedContactTaskRewardsCfg:
    progress = _reward("progress", 4.0)
    return_progress = _reward("return_progress", 1.0)  # Released limb's safe return XYZ
    phase_contact_error = _reward("phase_contact_error", -0.5)
    release_event = _reward("release_event", 0.2)  # Total per-goal budget, shared by active limbs
    contact_event = _reward("contact_event", 0.5)
    support_loss = _reward("support_loss", -1.0)
    slip = _reward("slip", -0.2)
    wrong_contact = _reward("wrong_contact", -1.0)
    danger = _reward("danger", -2.0)
    completion = _reward("completion", 5.0)
    failure = _reward("failure", -2.0)


@configclass
class CR1PhasedContactSequenceAMPEnvCfg(CR1ContactSequenceAMPEnvCfg):
    """Contact-phase reward experiment; preserve the v5 task as a baseline."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.contact_task_phases_enabled = True
        previous = self.rewards
        self.rewards = PhasedContactTaskRewardsCfg()
        for name in ("action_rate_l2", "joint_limit", "joint_torque_l2", "joint_acc_l2_upper",
                     "joint_acc_l2_lower", "action_second_difference_l2_lower"):
            if getattr(previous, name, None) is not None:
                setattr(self.rewards, name, deepcopy(getattr(previous, name)))


@configclass
class CR1PhasedBodyGoalContactSequenceAMPEnvCfg(CR1PhasedContactSequenceAMPEnvCfg):
    """Isolated body-anchor goal shaping; the phased baseline stays unchanged."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.contact_task_body_goal_enabled = True
        self.rewards.body_progress = _reward("body_progress", 0.5)  # Reward units / meter


@configclass
class CR1FootSafetyContactSequenceAMPEnvCfg(CR1PhasedBodyGoalContactSequenceAMPEnvCfg):
    """Foot-only surface and swing-pose shaping, with the same actor contract."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.contact_task_foot_safety_enabled = True
        # Bounded rates per offending foot, multiplied by dt by RewardManager.
        self.rewards.foot_front_contact = _reward("foot_front_contact", -3.0)
        self.rewards.foot_underside_contact = _reward("foot_underside_contact", -3.0)
        self.rewards.foot_swing_pose = _reward("foot_swing_pose", -1.0)
