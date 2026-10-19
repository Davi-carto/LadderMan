import gymnasium as gym

from . import agents, ladder_A_env_cfg


gym.register(
    id="Ladder-Climb-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ladder_A_env_cfg.G1LadderAEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LadderPPORunnerCfg",
    },
)
