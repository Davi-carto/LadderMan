"""Visual DAgger variants that track the retained blind SmoothOutput task."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp

from ..wood_step_transformer_vision_stop_command_validated_task_env_cfg import (
    ZDW0824_VISUAL_STOP_COMMAND_MOTION,
)
from ..wood_step_zdw0824_no_command_full_state_env_cfg import (
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg,
)
from .pure_env_cfg import (
    CR1PureVisualDaggerInputsCfg,
    CR1PureVisualDaggerObservationsCfg,
)
from .visual_env_cfg import CR1DaggerVisualObservationsCfg


@configclass
class CR1SyncedVisualDaggerEnvCfg(
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg
):
    """Keep the current blind MDP, adding only deployable visual observations."""

    observations: CR1DaggerVisualObservationsCfg = CR1DaggerVisualObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Stage 37 preserves every Stage-36 pose/contact/reset array and adds
        # the up/down command annotations consumed by the student observations.
        self.commands.motion.motion_file = str(ZDW0824_VISUAL_STOP_COMMAND_MOTION)


@configclass
class CR1SyncedPureVisualDaggerEnvCfg(CR1SyncedVisualDaggerEnvCfg):
    """Use the same synchronized MDP without reward or auxiliary observations."""

    observations: CR1PureVisualDaggerObservationsCfg = CR1PureVisualDaggerObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        for name, value in vars(self.rewards).items():
            if isinstance(value, RewardTermCfg):
                setattr(self.rewards, name, None)


@configclass
class CR1SyncedVisualMseDaggerEnvCfg(CR1SyncedVisualDaggerEnvCfg):
    """Pure visual observations while retaining the synchronized MDP metrics."""

    observations: CR1PureVisualDaggerObservationsCfg = CR1PureVisualDaggerObservationsCfg()


@configclass
class CR1RecurrentVisualMseDaggerInputsCfg(CR1PureVisualDaggerInputsCfg):
    """25 command-aware proprio frames and one current depth exposure."""

    depth_updated = ObsTerm(func=mdp.cr1_transformer_vision_depth_updated)

    def __post_init__(self):
        super().__post_init__()
        self.proprio_history.params = dict(self.proprio_history.params)
        self.proprio_history.params["num_proprio_frames"] = 25
        self.proprio_history.params["num_depth_frames"] = 1


@configclass
class CR1RecurrentVisualMseDaggerObservationsCfg(
    CR1PureVisualDaggerObservationsCfg
):
    transformer_vision_inputs: CR1RecurrentVisualMseDaggerInputsCfg = (
        CR1RecurrentVisualMseDaggerInputsCfg()
    )


@configclass
class CR1SyncedRecurrentVisualMseDaggerEnvCfg(CR1SyncedVisualDaggerEnvCfg):
    """Synchronized blind MDP with recurrent deployable visual inputs."""

    observations: CR1RecurrentVisualMseDaggerObservationsCfg = (
        CR1RecurrentVisualMseDaggerObservationsCfg()
    )


__all__ = [
    "CR1SyncedVisualDaggerEnvCfg",
    "CR1SyncedPureVisualDaggerEnvCfg",
    "CR1SyncedVisualMseDaggerEnvCfg",
    "CR1SyncedRecurrentVisualMseDaggerEnvCfg",
]
