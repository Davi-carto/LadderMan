"""Train TransformerVision while the established CR1 actor remains frozen."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

DEFAULT_ACTOR_CHECKPOINT = (
    REPO_ROOT
    / "logs"
    / "rsl_rl"
    / "cr1_wood_step_reference_force_limited_strict_front_side_deploy_obs"
    / "2026-07-31_14-01-06_deploy_obs_from_scratch_to7000"
    / "model_6999.pt"
)
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "cr1_transformer_vision_pretrain"


parser = argparse.ArgumentParser(
    description="Visual-only TransformerVision pretraining under CR1's frozen nonvisual actor."
)
parser.add_argument(
    "--task",
    type=str,
    default="Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-WarpTransformerVision-v0",
)
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ACTOR_CHECKPOINT, help="Frozen actor checkpoint.")
parser.add_argument("--resume", type=Path, default=None, help="Optional TransformerVision checkpoint to resume.")
parser.add_argument(
    "--depth_update_hz", type=float, default=None,
    help="Override Warp depth refresh Hz (0: policy rate). Resume defaults to saved camera timing.",
)
parser.add_argument(
    "--visual_architecture",
    choices=("deterministic", "task_goal"),
    default="deterministic",
    help="Visual decoder/supervision contract to train.",
)
parser.add_argument("--num_envs", type=int, default=32, help="Parallel frozen-policy environments.")
parser.add_argument("--iterations", type=int, default=1, help="One fresh rollout and one visual update per iteration.")
parser.add_argument("--rollout_steps", type=int, default=24, help="Stored control steps per environment.")
parser.add_argument("--learning_rate", type=float, default=1.0e-3)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument(
    "--mini_batch_envs",
    type=int,
    default=256,
    help="Environments per visual optimizer mini-batch; every batch keeps the full rollout sequence.",
)
parser.add_argument("--save_interval", type=int, default=1, help="Iterations between visual checkpoints.")
parser.add_argument("--output_dir", type=Path, default=None, help="Run directory; defaults to a timestamped RSL-RL log directory.")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.iterations <= 0:
    parser.error("--iterations must be positive")
if args_cli.rollout_steps <= 0:
    parser.error("--rollout_steps must be positive")
if args_cli.learning_rate <= 0.0:
    parser.error("--learning_rate must be positive")
if args_cli.max_grad_norm <= 0.0:
    parser.error("--max_grad_norm must be positive")
if args_cli.mini_batch_envs <= 0:
    parser.error("--mini_batch_envs must be positive")
if args_cli.save_interval <= 0:
    parser.error("--save_interval must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import parse_env_cfg

import ladder.tasks  # noqa: F401
from ladder.rsl_rl.modules import TaskGoalTransformerVisionEncoder, TransformerVisionEncoder
from ladder.rsl_rl.runners import FrozenPolicyVisionRunner, TransformerVisionVecEnvWrapper
from ladder.utils.depth_camera_config import configure_depth_camera_timing


def _loss_values(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    values = {name: float(value.detach()) for name, value in losses.items()}
    if not all(torch.isfinite(torch.tensor(value)) for value in values.values()):
        raise RuntimeError(f"Non-finite TransformerVision losses: {values}")
    return values


def _run_directory() -> Path:
    if args_cli.output_dir is not None:
        return args_cli.output_dir.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "task_goal" if args_cli.visual_architecture == "task_goal" else "deterministic"
    return DEFAULT_LOG_ROOT / suffix / timestamp


def main() -> None:
    torch.manual_seed(args_cli.seed)
    actor_checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not actor_checkpoint.is_file():
        raise FileNotFoundError(f"Frozen actor checkpoint was not found: {actor_checkpoint}")
    resume_checkpoint = None
    if args_cli.resume is not None:
        resume_checkpoint = args_cli.resume.expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"TransformerVision resume checkpoint was not found: {resume_checkpoint}")
    output_dir = _run_directory()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    configure_depth_camera_timing(
        env_cfg, checkpoint=resume_checkpoint, update_hz=args_cli.depth_update_hz,
    )
    dump_yaml(str(output_dir / "params" / "env.yaml"), env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        vector_env = TransformerVisionVecEnvWrapper(env)
        encoder = (
            TaskGoalTransformerVisionEncoder()
            if args_cli.visual_architecture == "task_goal"
            else TransformerVisionEncoder()
        )
        runner = FrozenPolicyVisionRunner(
            vector_env,
            actor_checkpoint,
            rollout_steps=args_cli.rollout_steps,
            learning_rate=args_cli.learning_rate,
            max_grad_norm=args_cli.max_grad_norm,
            mini_batch_envs=args_cli.mini_batch_envs,
            encoder=encoder,
        )
        print(
            "[TRANSFORMER VISION TRAIN] GRU hidden state carries across rollout chunks and resets only for done environments.",
            flush=True,
        )
        start_iteration = 0
        if resume_checkpoint is not None:
            start_iteration = runner.load_visual_checkpoint(resume_checkpoint, load_optimizer=True)
        print(
            "[TRANSFORMER VISION TRAIN] "
            f"actor={actor_checkpoint.name} envs={args_cli.num_envs} rollout_steps={args_cli.rollout_steps} "
            f"iterations={args_cli.iterations} start_iteration={start_iteration} "
            f"learning_rate={args_cli.learning_rate:g} max_grad_norm={args_cli.max_grad_norm:g} "
            f"mini_batch_envs={args_cli.mini_batch_envs} data_contract={runner.encoder.data_contract}",
            flush=True,
        )
        history: list[dict[str, object]] = []
        for offset in range(1, args_cli.iterations + 1):
            iteration = start_iteration + offset
            step = runner.collect_and_update()
            losses = _loss_values(step.losses)
            metrics = {
                "iteration": iteration,
                "losses_before_update": losses,
                "gradient_norm_before_clip": step.vision_gradient_norm,
                "visual_parameter_change_max_abs": step.vision_parameter_change_max_abs,
                "frozen_actor_parameter_change_max_abs": step.actor_parameter_change_max_abs,
                "mean_reward": float(step.rewards.mean()),
                "done_count": int(step.dones.sum()),
                "future_valid_transitions": int(runner.storage.future_valid.sum()),
                "future_total_transitions": int(runner.storage.future_valid.numel()),
            }
            history.append(metrics)
            contract_metrics = (
                f"limb_goal={losses['limb_goal']:.6f} "
                f"contact_bce={losses['contact_bce']:.6f} "
                f"contact_accuracy={losses['contact_accuracy']:.3f} "
                f"motion_joint={losses['motion_joint']:.6f} "
                f"motion_orientation={losses['motion_root_orientation']:.6f}"
                if "limb_goal" in losses
                else f"future={losses['future']:.6f}"
            )
            print(
                "[TRANSFORMER VISION TRAIN] "
                f"iteration={iteration} total={losses['total']:.6f} velocity={losses['velocity']:.6f} "
                f"velocity_zero={losses['velocity_zero_baseline']:.6f} "
                f"velocity_skill={losses['velocity_skill_score']:.3f} "
                f"hand={losses['hand']:.6f} foot={losses['foot']:.6f} {contract_metrics} "
                f"rung={losses['heightmap_rung']:.6f} non_rung={losses['heightmap_non_rung']:.6f} "
                f"dynamic_abs_max={losses['dynamic_abs_max']:.6f} "
                f"gradient_norm={step.vision_gradient_norm:.6f} "
                f"visual_delta={step.vision_parameter_change_max_abs:.3e} "
                f"actor_delta={step.actor_parameter_change_max_abs:.1e}",
                flush=True,
            )
            if iteration % args_cli.save_interval == 0 or offset == args_cli.iterations:
                checkpoint_path = output_dir / f"model_{iteration}.pt"
                runner.save_visual_checkpoint(checkpoint_path, iteration=iteration)
                print(f"[TRANSFORMER VISION TRAIN] saved={checkpoint_path}", flush=True)

        summary = {
            "purpose": "TransformerVision auxiliary-only pretraining. The frozen actor and PPO are excluded.",
            "actor_checkpoint": str(actor_checkpoint),
            "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "rollout_steps": args_cli.rollout_steps,
            "learning_rate": args_cli.learning_rate,
            "max_grad_norm": args_cli.max_grad_norm,
            "mini_batch_envs": args_cli.mini_batch_envs,
            "visual_architecture": args_cli.visual_architecture,
            "data_contract": runner.encoder.data_contract,
            "history": history,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
        print(f"[TRANSFORMER VISION TRAIN] summary={output_dir / 'summary.json'}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
