from isaaclab.utils import configclass

from .wood_step_geometry_dr_relaxed_reset_env_cfg import (
    CR1WoodStepGeometryDRRelaxedResetEnvCfg,
)


@configclass
class CR1WoodStepGeometryDRRelaxedResetNominal30EnvCfg(
    CR1WoodStepGeometryDRRelaxedResetEnvCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.nominal_ladder_geometry_probability = 0.30
