from isaaclab.utils import configclass

from .wood_step_geometry_dr_env_cfg import CR1WoodStepGeometryDREnvCfg


@configclass
class CR1WoodStepGeometryDRRelaxedResetEnvCfg(CR1WoodStepGeometryDREnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Use the nominal-ladder reset mask instead of conservative randomized-geometry masks.
        self.commands.motion.use_geometry_conditioned_reset_masks = False
