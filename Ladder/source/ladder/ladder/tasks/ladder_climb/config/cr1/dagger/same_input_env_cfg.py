"""Exact-observation positive control for CR1 DAgger distillation."""

from isaaclab.utils import configclass

from ..wood_step_zdw0824_no_command_full_state_env_cfg import (
    CR1ZDW0824FullStateObservationCfg,
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg,
)


@configclass
class CR1SameInputDaggerObservationsCfg:
    """A single 780-D tensor shared by the student and teacher models."""

    policy: CR1ZDW0824FullStateObservationCfg = (
        CR1ZDW0824FullStateObservationCfg()
    )


@configclass
class CR1SameInputDaggerEnvCfg(
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg
):
    """Latest Stage-36 task with exactly one full-state observation group."""

    observations: CR1SameInputDaggerObservationsCfg = (
        CR1SameInputDaggerObservationsCfg()
    )


__all__ = ["CR1SameInputDaggerEnvCfg", "CR1SameInputDaggerObservationsCfg"]
