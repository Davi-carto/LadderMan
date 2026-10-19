"""Run a bounded Gate-D smoke test for CR1's recurrent Warp visual PPO path."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

parser = argparse.ArgumentParser(description="Validate CR1's recurrent SSR PPO mechanics.")
parser.add_argument(
    "--task",
    type=str,
    default="Ladder-Climb-CR1-Geometry-DR-WarpSSRVisual-v0",
    help="The opt-in CR1 Warp SSR visual task.",
)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.steps <= 0:
    parser.error("--steps must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ladder.tasks  # noqa: F401
from ladder.algorithms import LadderSSRPPO
from ladder.runners import LadderSSRVecEnvWrapper
from ladder.tasks.ladder_climb.config.cr1.agents.rsl_rl_ppo_cfg import CR1SSRVisualPPORunnerCfg


def _assert_finite(observations) -> None:
    for name, value in observations.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"SSR PPO observation {name!r} contains non-finite values")


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        vector_env = LadderSSRVecEnvWrapper(env)
        runner_cfg = CR1SSRVisualPPORunnerCfg()
        runner_cfg.device = args_cli.device
        runner_cfg.num_steps_per_env = args_cli.steps
        runner_cfg.algorithm.num_learning_epochs = 1
        runner_cfg.algorithm.num_mini_batches = 1
        runner_cfg.algorithm.enable_auxiliary_update = False
        algorithm = LadderSSRPPO.construct_algorithm(
            vector_env.get_observations(), vector_env, runner_cfg.to_dict(), args_cli.device
        )
        algorithm.train_mode()
        observations = vector_env.get_observations()
        _assert_finite(observations)
        print(
            "[SSR PPO] "
            f"envs={args_cli.num_envs} steps={args_cli.steps} "
            f"proprio={tuple(observations['ssr_proprio_history'].shape)} "
            f"depth={tuple(observations['ssr_depth_history'].shape)} "
            f"maps={tuple(observations['ssr_limb_heightmaps'].shape)} "
            f"critic={tuple(observations['critic'].shape)}",
            flush=True,
        )
        for step in range(args_cli.steps):
            actions = algorithm.act(observations)
            observations, rewards, dones, extras = vector_env.step(actions)
            _assert_finite(observations)
            algorithm.process_env_step(observations, rewards, dones, extras)
            print(
                f"[SSR PPO] rollout={step + 1}/{args_cli.steps} "
                f"reward_mean={float(rewards.mean()):.6f} done_count={int(dones.sum())}",
                flush=True,
            )

        algorithm.compute_returns(observations)
        ratio_error = float(algorithm.rollout_log_prob_max_error())
        if ratio_error > 1.0e-5:
            raise RuntimeError(f"Stored actor input changed the PPO log-probability by {ratio_error:.3e}")
        losses = algorithm.update()
        if not all(torch.isfinite(torch.tensor(value)) for value in losses.values()):
            raise RuntimeError(f"Non-finite PPO losses: {losses}")
        print(f"[SSR PPO] fixed_latent_log_prob_error={ratio_error:.3e} losses={losses}", flush=True)

        algorithm.eval_mode()
        algorithm.reset_inference()
        reference_actions = algorithm.act_inference(observations)
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save(algorithm.save(), checkpoint.name)
            restored = LadderSSRPPO.construct_algorithm(
                observations, vector_env, runner_cfg.to_dict(), args_cli.device
            )
            restored.load(torch.load(checkpoint.name, weights_only=False), load_cfg=None, strict=True)
        restored.eval_mode()
        restored.reset_inference()
        torch.testing.assert_close(reference_actions, restored.act_inference(observations))
        print(
            f"[SSR PPO] checkpoint_reload=pass deterministic_actions={tuple(reference_actions.shape)} "
            "auxiliary_update=disabled",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
