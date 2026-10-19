"""Visual DAgger tasks for CR1."""

from .agent_cfg import (
    CR1PureVisualDaggerRunnerCfg,
    CR1SyncedPureVisualDaggerRunnerCfg,
    CR1SyncedVisualDaggerPPORunnerCfg,
    CR1VisualDaggerPPORunnerCfg,
)
from .pure_env_cfg import CR1PureVisualDaggerEnvCfg
from .privileged_env_cfg import CR1SyncedPrivilegedRecurrentVisualMseDaggerEnvCfg
from .privileged_recurrent_visual_mse_algorithm import (
    CR1PrivilegedRecurrentVisualMseDagger,
)
from .privileged_recurrent_visual_mse_agent_cfg import (
    CR1PrivilegedRecurrentVisualMseDaggerRunnerCfg,
)
from .recurrent_visual_mse_agent_cfg import (
    CR1RecurrentVisualMixedMseDaggerRunnerCfg,
    CR1RecurrentVisualMseDaggerRunnerCfg,
    CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg,
    CR1RecurrentVisualScheduledMseDaggerRunnerCfg,
)
from .recurrent_visual_ppo_agent_cfg import CR1RecurrentVisualPpoDaggerRunnerCfg
from .recurrent_visual_ln_mse_agent_cfg import CR1RecurrentVisualLnMseDaggerRunnerCfg
from .same_input_agent_cfg import CR1SameInputDaggerRunnerCfg
from .same_input_env_cfg import CR1SameInputDaggerEnvCfg
from .visual_mse_agent_cfg import CR1VisualMseDaggerRunnerCfg
from .visual_env_cfg import CR1DaggerVisualEnvCfg
from .synced_env_cfg import (
    CR1SyncedPureVisualDaggerEnvCfg,
    CR1SyncedRecurrentVisualMseDaggerEnvCfg,
    CR1SyncedVisualDaggerEnvCfg,
    CR1SyncedVisualMseDaggerEnvCfg,
)

__all__ = [
    "CR1DaggerVisualEnvCfg",
    "CR1VisualDaggerPPORunnerCfg",
    "CR1PureVisualDaggerEnvCfg",
    "CR1PureVisualDaggerRunnerCfg",
    "CR1SyncedVisualDaggerEnvCfg",
    "CR1SyncedPureVisualDaggerEnvCfg",
    "CR1SameInputDaggerEnvCfg",
    "CR1SameInputDaggerRunnerCfg",
    "CR1SyncedVisualDaggerPPORunnerCfg",
    "CR1SyncedPureVisualDaggerRunnerCfg",
    "CR1SyncedVisualMseDaggerEnvCfg",
    "CR1VisualMseDaggerRunnerCfg",
    "CR1SyncedRecurrentVisualMseDaggerEnvCfg",
    "CR1RecurrentVisualMseDaggerRunnerCfg",
    "CR1RecurrentVisualMixedMseDaggerRunnerCfg",
    "CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg",
    "CR1RecurrentVisualScheduledMseDaggerRunnerCfg",
    "CR1RecurrentVisualPpoDaggerRunnerCfg",
    "CR1RecurrentVisualLnMseDaggerRunnerCfg",
    "CR1SyncedPrivilegedRecurrentVisualMseDaggerEnvCfg",
    "CR1PrivilegedRecurrentVisualMseDagger",
    "CR1PrivilegedRecurrentVisualMseDaggerRunnerCfg",
]
