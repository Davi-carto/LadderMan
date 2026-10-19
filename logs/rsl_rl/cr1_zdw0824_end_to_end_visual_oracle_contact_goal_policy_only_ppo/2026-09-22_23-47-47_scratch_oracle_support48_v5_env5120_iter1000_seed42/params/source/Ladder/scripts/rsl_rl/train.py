"""Train LadderMan policies with RSL-RL.

This entry point is intentionally local to Ladder so task registration and
reference-motion loading do not depend on whole_body_tracking scripts.
"""

import argparse
import hashlib
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


REPO_ROOT = Path(__file__).resolve().parents[3]
LADDER_SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(LADDER_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(LADDER_SOURCE_DIR))


parser = argparse.ArgumentParser(description="Train a LadderMan RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings in steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Ladder-Climb-CR1-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to LadderMan climbing motion.npz.")
parser.add_argument(
    "--depth_update_hz", type=float, default=None,
    help="Override Warp depth refresh Hz (0: policy rate). Resume defaults to saved camera timing.",
)
parser.add_argument(
    "--vision_checkpoint",
    type=str,
    default=None,
    help="TransformerVision pretraining checkpoint used to initialize a fresh visual PPO run.",
)
parser.add_argument(
    "--teacher_checkpoint",
    type=str,
    default=None,
    help="Frozen full-state teacher checkpoint for Hybrid DAgger + PPO.",
)
parser.add_argument(
    "--disable_randomization",
    action="store_true",
    help="Disable startup physics randomization and reset-state perturbations for baseline training.",
)
parser.add_argument(
    "--base_reset_noise_only",
    action="store_true",
    help="Disable physics/joint randomization but retain the configured base pose and velocity reset noise.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.disable_randomization and args_cli.base_reset_noise_only:
    parser.error("--disable_randomization and --base_reset_noise_only are mutually exclusive")

if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.task.endswith("-Visual-v0"):
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import ladder.tasks  # noqa: F401
from ladder.utils.depth_camera_config import configure_depth_camera_timing
from rsl_rl_compat import (
    convert_legacy_distillation_runner_cfg,
    convert_legacy_runner_cfg,
)


try:
    from isaaclab.utils.io import dump_pickle
except ImportError:

    def dump_pickle(filename, data):
        """Compatibility fallback for IsaacLab versions without dump_pickle."""

        if not filename.endswith(".pkl"):
            filename += ".pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as file:
            pickle.dump(data, file)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
    # Command/contact markers are useful for replay, but add no training signal
    # and create point-instancer warnings in headless multi-env training.
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    if args_cli.disable_randomization:
        for event_name in (
            "robot_material",
            "robot_mass",
            "joint_position_offset",
            "torso_com",
            "motor_strength",
            "motor_strength_interval",
            "camera_extrinsics",
            "root_velocity_push",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
        env_cfg.commands.motion.pose_range = {
            name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        env_cfg.commands.motion.fixed_start_pose_range = {}
        env_cfg.commands.motion.velocity_range = {
            name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
        env_cfg.commands.motion.ladder_inclination_delta_range = (0.0, 0.0)
        env_cfg.commands.motion.ladder_spacing_delta_range = (0.0, 0.0)
        print("[INFO] Domain and reset-state randomization disabled for this run.")
    elif args_cli.base_reset_noise_only:
        for event_name in (
            "robot_material",
            "robot_mass",
            "joint_position_offset",
            "torso_com",
            "motor_strength",
            "motor_strength_interval",
            "camera_extrinsics",
            "root_velocity_push",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
        env_cfg.commands.motion.ladder_inclination_delta_range = (0.0, 0.0)
        env_cfg.commands.motion.ladder_spacing_delta_range = (0.0, 0.0)
        print("[INFO] Physics/joint randomization disabled; base reset noise retained.")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if agent_cfg.resume:
        explicit_checkpoint = None
        if args_cli.checkpoint is not None:
            checkpoint_path = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
            if os.path.isfile(checkpoint_path):
                explicit_checkpoint = checkpoint_path
        resume_path = (
            explicit_checkpoint
            if explicit_checkpoint is not None
            else get_checkpoint_path(
                log_root_path,
                agent_cfg.load_run,
                agent_cfg.load_checkpoint,
            )
        )

    configure_depth_camera_timing(
        env_cfg, checkpoint=resume_path if agent_cfg.resume else None,
        update_hz=args_cli.depth_update_hz,
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    runner_cfg = agent_cfg.to_dict()
    runner_class = OnPolicyRunner
    if runner_cfg.get("use_end_to_end_visual_ppo", False):
        from ladder.rsl_rl.runners.end_to_end_visual_wrapper import EndToEndVisualPPOVecEnvWrapper

        if args_cli.teacher_checkpoint is not None or args_cli.vision_checkpoint is not None:
            raise ValueError(
                "End-to-end reconstruction PPO trains from scratch (or --resume); "
                "do not pass teacher/vision pretraining checkpoints."
            )
        # The rollout and sequence replay must use the same deterministic FP32
        # encoder path; mixed precision can be considered after this baseline.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        env = EndToEndVisualPPOVecEnvWrapper(
            env, clip_actions=agent_cfg.clip_actions,
            clip_observations=getattr(agent_cfg, "clip_observations", None),
        )
        if runner_cfg["policy"].get("use_oracle_contact_goal", False):
            print(
                "[INFO] ORACLE DIAGNOSTIC: actor receives GRU256 plus next-stable goal12, "
                "actual limb position12, actual contact4 and desired contact4; NOT deployable."
            )
            policy_gradient_path = "GRU256 directly to actions"
        else:
            policy_gradient_path = "latent heads to actions"
        print(
            "[INFO] End-to-end PPO + reconstruction: 100% current-policy rollout; "
            "25x85 command-aware proprio, 2x60x80 depth at 30 Hz, GRU at policy rate. "
            f"PPO backpropagates through CNN/Transformer/{policy_gradient_path}. "
            "One optimizer jointly trains policy, critic and velocity/height-map supervision; "
            "no teacher, motion decoder or next-state loss.",
            flush=True,
        )
    elif runner_cfg.get("use_same_input_dagger_runner", False):
        if args_cli.teacher_checkpoint is None and agent_cfg.resume:
            resume_payload = torch.load(
                resume_path, map_location="cpu", weights_only=False
            )
            args_cli.teacher_checkpoint = resume_payload.get("teacher_checkpoint")
        if args_cli.teacher_checkpoint is None:
            raise ValueError(
                "Same-input DAgger requires --teacher_checkpoint for a fresh run."
            )
        teacher_checkpoint = os.path.abspath(
            os.path.expanduser(args_cli.teacher_checkpoint)
        )
        if not os.path.isfile(teacher_checkpoint):
            raise FileNotFoundError(
                "Same-input DAgger teacher checkpoint was not found: "
                f"{teacher_checkpoint}"
            )
        runner_cfg["algorithm"]["teacher_checkpoint"] = teacher_checkpoint
        agent_cfg.algorithm.teacher_checkpoint = teacher_checkpoint
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cfg = convert_legacy_distillation_runner_cfg(runner_cfg)
        runner_class = DistillationRunner
        print(
            "[INFO] Same-input DAgger control: student and teacher consume the "
            "same 780-D policy tensor; student-only stochastic rollout, online "
            "teacher labels, and action-mean MSE updates.",
            flush=True,
        )
    elif runner_cfg.get("use_pure_dagger_visual_runner", False):
        from ladder.tasks.ladder_climb.config.cr1.dagger.pure_wrapper import (
            CR1PureVisualDaggerVecEnvWrapper,
        )

        visual_mse_dagger = runner_cfg.get("use_visual_mse_dagger_runner", False)
        recurrent_visual_mse_dagger = runner_cfg.get(
            "use_recurrent_visual_mse_dagger_runner", False
        )
        recurrent_visual_ppo_dagger = runner_cfg.get(
            "use_recurrent_visual_ppo_dagger_runner", False
        )
        privileged_recurrent_visual_mse_dagger = runner_cfg.get(
            "use_privileged_recurrent_visual_mse_dagger_runner", False
        )
        if args_cli.teacher_checkpoint is None and agent_cfg.resume and visual_mse_dagger:
            resume_payload = torch.load(
                resume_path, map_location="cpu", weights_only=False
            )
            args_cli.teacher_checkpoint = resume_payload.get("teacher_checkpoint")
        if args_cli.teacher_checkpoint is None:
            args_cli.teacher_checkpoint = runner_cfg["algorithm"].get(
                "teacher_checkpoint"
            ) or None
        if args_cli.teacher_checkpoint is None:
            raise ValueError("A pure visual DAgger run requires --teacher_checkpoint")
        teacher_checkpoint = os.path.abspath(os.path.expanduser(args_cli.teacher_checkpoint))
        if not os.path.isfile(teacher_checkpoint):
            raise FileNotFoundError(
                f"Pure visual DAgger teacher checkpoint was not found: {teacher_checkpoint}"
            )
        runner_cfg["algorithm"]["teacher_checkpoint"] = teacher_checkpoint
        agent_cfg.algorithm.teacher_checkpoint = teacher_checkpoint
        env = CR1PureVisualDaggerVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if recurrent_visual_ppo_dagger:
            print(
                "[INFO] Recurrent Visual PPO-DAgger: fixed "
                f"{agent_cfg.algorithm.teacher_rollout_fraction:.0%} teacher / "
                f"{1.0 - agent_cfg.algorithm.teacher_rollout_fraction:.0%} student "
                "environments from iteration 0. All states train action-mean MSE; "
                "only student trajectories train PPO, GAE and the privileged critic. "
                f"PPO actor coefficient ramps from {agent_cfg.algorithm.ppo_initial_coeff:g} "
                f"to 1 over {agent_cfg.algorithm.ppo_warmup_iterations} updates "
                "(surrogate and entropy); critic coefficient stays "
                f"{agent_cfg.algorithm.value_loss_coef:g} from update 0. "
                f"MSE coefficient {agent_cfg.algorithm.dagger_initial_coeff:g} is held "
                f"through update {agent_cfg.algorithm.dagger_anneal_start}, then decays "
                f"to {agent_cfg.algorithm.dagger_final_coeff:g} by update "
                f"{agent_cfg.algorithm.dagger_anneal_start + agent_cfg.algorithm.dagger_anneal_iterations}. "
                "LayerNorm before the fresh-frame GRU; PPO updates the entire student.",
                flush=True,
            )
        elif runner_cfg.get("use_recurrent_visual_ln_mse_dagger_runner", False):
            print(
                "[INFO] LayerNorm-only MSE ablation: fixed 25% teacher / 75% student; "
                "action-mean MSE only, fixed std=0.1, fixed learning rate=1e-3, "
                "1 epoch. Two branch LayerNorms are the only network change; "
                "no PPO, critic, entropy or auxiliary loss.",
                flush=True,
            )
        elif recurrent_visual_mse_dagger:
            if privileged_recurrent_visual_mse_dagger:
                print(
                    "[INFO] Privileged recurrent visual MSE-DAgger: current "
                    "25x85 proprio + 30-Hz depth GRU visual feature, plus "
                    "28-D oracle actor input (next-stable target12 + actual "
                    "ladder contact4 + limb velocity12). Action-mean MSE only; "
                    "separate LayerNorm(128) on proprio/depth features before GRU; "
                    "teacher remains frozen. It uses "
                    f"{agent_cfg.algorithm.bc_warmup_iterations} all-teacher "
                    "updates, then linearly decreases the teacher fraction to "
                    f"{agent_cfg.algorithm.teacher_rollout_schedule_final_fraction:.0%} "
                    f"at update {agent_cfg.algorithm.teacher_rollout_schedule_end}. "
                    "This is an information-"
                    "sufficiency ablation and is NOT deployable.",
                    flush=True,
                )
            elif runner_cfg.get(
                "use_recurrent_visual_scheduled_mse_dagger_runner", False
            ):
                print(
                    "[INFO] Recurrent Visual Scheduled MSE-DAgger control: "
                    "100% deterministic teacher rollout before iteration 500; "
                    "teacher fraction linearly decreases from 100% at iteration "
                    f"{agent_cfg.algorithm.teacher_rollout_schedule_start} to "
                    f"{agent_cfg.algorithm.teacher_rollout_schedule_final_fraction:.0%} "
                    f"at iteration {agent_cfg.algorithm.teacher_rollout_schedule_end}, "
                    "then remains at the final fraction. All states use online "
                    "teacher action-mean labels and sequence MSE updates; no PPO, "
                    "critic, feature, or auxiliary reconstruction loss.",
                    flush=True,
                )
            elif runner_cfg.get(
                "use_recurrent_visual_mixed_mse_dagger_runner", False
            ):
                print(
                    "[INFO] Recurrent Visual Mixed MSE-DAgger control: "
                    f"{agent_cfg.algorithm.bc_warmup_iterations} iterations of "
                    "100% deterministic teacher rollout, then fixed "
                    f"{agent_cfg.algorithm.teacher_rollout_fraction:.0%} teacher / "
                    f"{1.0 - agent_cfg.algorithm.teacher_rollout_fraction:.0%} "
                    "stochastic student environments; all states use online teacher "
                    "action-mean labels and sequence MSE updates.",
                    flush=True,
                )
            else:
                print(
                    "[INFO] Recurrent Visual MSE-DAgger control: 100% stochastic "
                    "student rollout, 25 command-aware proprio frames, 30-Hz fresh-frame "
                    "depth GRU, continuous sequence updates, and action-mean MSE only.",
                    flush=True,
                )
        elif visual_mse_dagger:
            print(
                "[INFO] Visual MSE-DAgger control: 100% stochastic student rollout, "
                "online frozen-teacher action-mean labels, and action-mean MSE only. "
                "No PPO, KL, feature, velocity, or reconstruction loss.",
                flush=True,
            )
        elif runner_cfg.get("use_teacher_only_visual_bc", False):
            print(
                "[INFO] Teacher-only visual BC: every environment executes teacher means forever; "
                f"last {agent_cfg.algorithm.validation_num_envs} envs are held out of optimization. "
                "Only 2*KL + 1*feature MSE updates. Episode performance belongs to TEACHER.",
                flush=True,
            )
        else:
            print(
                "[INFO] Pure Visual DAgger uses BC warmup then fixed teacher/student mixing, "
                "online teacher labels, and only 2*KL + 1*feature imitation updates.",
                flush=True,
            )
    elif runner_cfg.get("use_dagger_visual_runner", False):
        from ladder.tasks.ladder_climb.config.cr1.dagger.visual_wrapper import (
            CR1DaggerVisualVecEnvWrapper,
        )

        if args_cli.teacher_checkpoint is None:
            raise ValueError("A visual DAgger run requires --teacher_checkpoint")
        teacher_checkpoint = os.path.abspath(
            os.path.expanduser(args_cli.teacher_checkpoint)
        )
        if not os.path.isfile(teacher_checkpoint):
            raise FileNotFoundError(
                f"Visual DAgger teacher checkpoint was not found: {teacher_checkpoint}"
            )
        runner_cfg["algorithm"]["teacher_checkpoint"] = teacher_checkpoint
        agent_cfg.algorithm.teacher_checkpoint = teacher_checkpoint
        env = CR1DaggerVisualVecEnvWrapper(
            env, clip_actions=agent_cfg.clip_actions
        )
        print(
            "[INFO] Visual DAgger student uses 5x85 proprio with +/-1 command, "
            "two Warp depth frames, and a 35-D velocity/height-map latent.",
            flush=True,
        )
    elif runner_cfg.get("use_transformer_vision_runner", False):
        from ladder.rsl_rl.runners import TransformerVisionPPOVecEnvWrapper
        from ladder.rsl_rl.modules.transformer_vision import (
            LEGACY_TASK_GOAL_TRANSFORMER_VISION_DATA_CONTRACT,
            VARIATIONAL_TRANSFORMER_VISION_DATA_CONTRACT,
        )

        if agent_cfg.resume:
            checkpoint_payload = torch.load(
                resume_path, map_location="cpu", weights_only=False
            )
            checkpoint_contract = checkpoint_payload.get("data_contract")
            if checkpoint_contract == VARIATIONAL_TRANSFORMER_VISION_DATA_CONTRACT:
                runner_cfg["algorithm"]["visual_architecture"] = "variational_legacy"
                agent_cfg.algorithm.visual_architecture = "variational_legacy"
            elif checkpoint_contract == LEGACY_TASK_GOAL_TRANSFORMER_VISION_DATA_CONTRACT:
                runner_cfg["algorithm"]["visual_architecture"] = "task_goal_legacy"
                agent_cfg.algorithm.visual_architecture = "task_goal_legacy"

        visual_initialization = runner_cfg["algorithm"].get(
            "visual_initialization", "checkpoint"
        )
        if visual_initialization not in ("checkpoint", "scratch"):
            raise ValueError(
                "TransformerVision visual_initialization must be 'checkpoint' or "
                f"'scratch', got {visual_initialization!r}"
            )
        if visual_initialization == "scratch" and args_cli.vision_checkpoint is not None:
            raise ValueError(
                "This TransformerVision task is configured for scratch training; "
                "do not pass --vision_checkpoint."
            )
        if args_cli.vision_checkpoint is not None:
            visual_checkpoint = os.path.abspath(os.path.expanduser(args_cli.vision_checkpoint))
            if not os.path.isfile(visual_checkpoint):
                raise FileNotFoundError(
                    f"TransformerVision pretraining checkpoint was not found: {visual_checkpoint}"
                )
            runner_cfg["algorithm"]["visual_checkpoint"] = visual_checkpoint
            agent_cfg.algorithm.visual_checkpoint = visual_checkpoint
        elif not agent_cfg.resume and visual_initialization == "checkpoint":
            raise ValueError(
                "A fresh TransformerVision PPO run requires --vision_checkpoint. "
                "Use --resume with a complete PPO checkpoint to continue an existing run."
            )
        elif not agent_cfg.resume:
            print(
                "[INFO] TransformerVision actor, critic, and visual encoder will train from scratch.",
                flush=True,
            )
        env = TransformerVisionPPOVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
            clip_observations=getattr(agent_cfg, "clip_observations", None),
        )
        visual_latent_dim = (
            67
            if runner_cfg["algorithm"].get("visual_architecture") == "task_goal"
            else 51
        )
        print(
            "[INFO] TransformerVision PPO uses current proprio 85 + rollout-fixed "
            f"latent {visual_latent_dim}; the privileged critic remains separate.",
            flush=True,
        )
    elif runner_cfg.get("use_ladder_ssr_runner", False):
        from ladder.runners import LadderSSRVecEnvWrapper

        env = LadderSSRVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        # rsl-rl 5 moved from one ActorCritic policy config to separate actor and
        # critic model configs. Older versions do not expose rsl_rl.models.MLPModel.
        try:
            from rsl_rl.models import MLPModel  # noqa: F401
        except ImportError:
            pass
        else:
            runner_cfg = convert_legacy_runner_cfg(runner_cfg)
    runner = runner_class(
        env, runner_cfg, log_dir=log_dir, device=agent_cfg.device
    )
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        if runner_cfg.get("use_teacher_only_visual_bc", False):
            # Saved `iter` is the last completed (zero-based) iteration; BC's
            # update_iteration is the next one. Keep diagnostics/model IDs aligned.
            runner.current_learning_iteration = runner.alg.update_iteration
        elif runner_cfg.get("use_same_input_dagger_runner", False):
            # RSL-RL saves the just-completed zero-based `iter`, while this
            # algorithm persists the number of completed optimizer updates.
            # Resume at the latter so model_500 continues from 501 rather than
            # collecting and logging iteration 500 a second time.
            runner.current_learning_iteration = runner.alg.num_updates
        elif runner_cfg.get("use_visual_mse_dagger_runner", False):
            runner.current_learning_iteration = runner.alg.update_iteration
        elif runner_cfg.get("use_end_to_end_visual_ppo", False):
            runner.current_learning_iteration = runner.alg.num_updates

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    if runner_cfg.get("use_end_to_end_visual_ppo", False):
        runner.alg.configure_diagnostics(log_dir)
        motion_path = Path(args_cli.motion_file).expanduser().resolve()
        source_root = LADDER_SOURCE_DIR / "ladder"
        source_paths = [
            Path(__file__).resolve(),
            source_root / "rsl_rl/algorithms/end_to_end_visual_ppo.py",
            source_root / "rsl_rl/modules/end_to_end_visual_actor.py",
            source_root / "rsl_rl/modules/transformer_vision.py",
            source_root / "rsl_rl/modules/transformer_vision_actor.py",
            source_root / "rsl_rl/storage/transformer_vision_ppo_storage.py",
            source_root / "rsl_rl/runners/end_to_end_visual_wrapper.py",
            source_root / "rsl_rl/runners/transformer_vision_ppo_wrapper.py",
            source_root / "tasks/ladder_climb/config/cr1/wood_step_end_to_end_visual_ppo_env_cfg.py",
            source_root / "tasks/ladder_climb/config/cr1/wood_step_zdw0824_no_command_full_state_env_cfg.py",
            source_root / "tasks/ladder_climb/config/cr1/agents/end_to_end_visual_ppo_cfg.py",
            *sorted((source_root / "tasks/ladder_climb/mdp").glob("*.py")),
        ]
        source_hashes = {}
        for source_path in source_paths:
            relative_path = source_path.relative_to(REPO_ROOT)
            snapshot_path = Path(log_dir) / "params/source" / relative_path
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, snapshot_path)
            source_hashes[str(relative_path)] = _sha256_file(source_path)
        dump_yaml(os.path.join(log_dir, "params", "end_to_end_run_manifest.yaml"), {
            "task": args_cli.task,
            "data_contract": runner.alg.data_contract,
            "motion_file": str(motion_path),
            "motion_file_sha256": _sha256_file(motion_path),
            "resume_checkpoint": resume_path if agent_cfg.resume else None,
            "resume_checkpoint_sha256": _sha256_file(Path(resume_path)) if agent_cfg.resume else None,
            "blind_environment_class": "CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg",
            "num_envs": env.num_envs,
            "seed": agent_cfg.seed,
            "sim_dt_s": env_cfg.sim.dt,
            "policy_dt_s": env_cfg.sim.dt * env_cfg.decimation,
            "depth_update_period_s": env_cfg.observations.transformer_vision_depth.image.params["update_period"],
            "proprio_history_frames": 25,
            "gru_update": "every policy step; sample-and-held depth",
            "actor_contains_reference": runner.alg.policy.use_oracle_contact_goal,
            "actor_input_dim": runner.alg.policy.actor_input_dim,
            "oracle_contact_goal": ("privileged diagnostic: goal_b12 + actual/current/next support12 each"
                                    if runner.alg.policy.use_oracle_contact_goal else None),
            "teacher_used": False,
            "empirical_observation_normalization": False,
            "precision": "float32; TF32 disabled",
            "source_sha256": source_hashes,
        })
    elif runner_cfg.get("use_teacher_only_visual_bc", False):
        runner.alg.configure_diagnostics(
            log_dir, runner_cfg, args_cli.motion_file, teacher_checkpoint, runner.logger,
            resume_path=resume_path if agent_cfg.resume else None,
        )
    elif runner_cfg.get("use_same_input_dagger_runner", False):
        motion_path = Path(args_cli.motion_file).expanduser().resolve()
        motion_digest = _sha256_file(motion_path)
        action_cfg = env_cfg.actions.joint_pos
        motion_cfg = env_cfg.commands.motion
        runner.alg.configure_diagnostics(
            log_dir,
            {
                "task": args_cli.task,
                "motion_file": str(motion_path),
                "motion_file_sha256": motion_digest,
                "num_envs": env.num_envs,
                "seed": agent_cfg.seed,
                "sim_dt_s": env_cfg.sim.dt,
                "decimation": env_cfg.decimation,
                "policy_dt_s": env_cfg.sim.dt * env_cfg.decimation,
                "min_control_delay_physics_steps": action_cfg.min_control_delay,
                "max_control_delay_physics_steps": action_cfg.max_control_delay,
                "fixed_start_probability": motion_cfg.fixed_start_probability,
                "fixed_start_pose_randomization_probability": (
                    motion_cfg.fixed_start_pose_randomization_probability
                ),
                "secondary_fixed_start_frame": (
                    motion_cfg.secondary_fixed_start_frame
                ),
                "secondary_fixed_start_probability": (
                    motion_cfg.secondary_fixed_start_probability
                ),
                "uniform_start_probability": (
                    motion_cfg.uniform_start_probability
                ),
                "use_reset_valid_mask": motion_cfg.use_reset_valid_mask,
                "clip_actions": agent_cfg.clip_actions,
                "student_rollout_noise_std": agent_cfg.policy.init_noise_std,
                "num_steps_per_env": agent_cfg.num_steps_per_env,
                "num_learning_epochs": agent_cfg.algorithm.num_learning_epochs,
                "gradient_length": agent_cfg.algorithm.gradient_length,
                "learning_rate": agent_cfg.algorithm.learning_rate,
                "max_grad_norm": agent_cfg.algorithm.max_grad_norm,
            },
        )
    elif runner_cfg.get("use_visual_mse_dagger_runner", False):
        motion_path = Path(args_cli.motion_file).expanduser().resolve()
        action_cfg = env_cfg.actions.joint_pos
        motion_cfg = env_cfg.commands.motion
        depth_params = (
            env_cfg.observations.transformer_vision_depth.image.params
        )
        history_params = (
            env_cfg.observations.transformer_vision_inputs.proprio_history.params
        )
        runner.alg.configure_diagnostics(
            log_dir,
            {
                "task": args_cli.task,
                "motion_file": str(motion_path),
                "motion_file_sha256": _sha256_file(motion_path),
                "num_envs": env.num_envs,
                "seed": agent_cfg.seed,
                "sim_dt_s": env_cfg.sim.dt,
                "decimation": env_cfg.decimation,
                "policy_dt_s": env_cfg.sim.dt * env_cfg.decimation,
                "depth_update_period_s": depth_params["update_period"],
                "depth_update_hz": (
                    0.0
                    if depth_params["update_period"] == 0.0
                    else 1.0 / depth_params["update_period"]
                ),
                "depth_center_m": history_params["depth_mean"],
                "depth_scale": history_params["depth_scale"],
                "proprio_input_noise_enabled": history_params["add_proprio_noise"],
                "student_observation_normalization": False,
                "teacher_actor_observation_normalization": False,
                **(
                    {
                        "student_feature_normalization": "separate_layer_norm_proprio_depth_before_gru",
                        "ablation": "layer_norm_only_vs_no_warmup_mixed_mse",
                        "student_rollout_noise_learnable": False,
                        "algorithm_config": runner_cfg["algorithm"],
                    }
                    if runner_cfg.get("use_recurrent_visual_ln_mse_dagger_runner", False)
                    else {}
                ),
                **(
                    {
                        "student_feature_normalization": (
                            "separate_layer_norm_proprio_depth_before_gru"
                        ),
                        "critic_observation_key": "teacher",
                        "critic_observation_dim": (
                            agent_cfg.algorithm.expected_teacher_observation_dim
                        ),
                        "ppo_rollout_source": "student_environments_only",
                        "dagger_rollout_source": "all_environments",
                        "student_rollout_noise_learnable": True,
                        "algorithm_config": runner_cfg["algorithm"],
                    }
                    if runner_cfg.get("use_recurrent_visual_ppo_dagger_runner", False)
                    else {}
                ),
                **(
                    {
                        "privileged_actor_input": {
                            "dimension": 28,
                            "target_position": "next stable four-limb support target in robot anchor frame, 12-D meters",
                            "actual_ladder_contact": "rung-filtered force threshold, 4-D binary",
                            "limb_linear_velocity": "four end-effectors in robot anchor frame, 12-D scaled by 0.1",
                            "deployable": False,
                        },
                        "student_feature_normalization": (
                            "separate_layer_norm_proprio_depth_before_gru"
                        ),
                        "student_observation_normalization": "fixed field-wise normalization; privileged target meters and limb velocity*0.1, binary contacts",
                    }
                    if runner_cfg.get("use_privileged_recurrent_visual_mse_dagger_runner", False)
                    else {}
                ),
                "min_control_delay_physics_steps": action_cfg.min_control_delay,
                "max_control_delay_physics_steps": action_cfg.max_control_delay,
                "fixed_start_probability": motion_cfg.fixed_start_probability,
                "fixed_start_pose_randomization_probability": (
                    motion_cfg.fixed_start_pose_randomization_probability
                ),
                "secondary_fixed_start_frame": motion_cfg.secondary_fixed_start_frame,
                "secondary_fixed_start_probability": (
                    motion_cfg.secondary_fixed_start_probability
                ),
                "uniform_start_probability": motion_cfg.uniform_start_probability,
                "use_reset_valid_mask": motion_cfg.use_reset_valid_mask,
                "clip_actions": agent_cfg.clip_actions,
                "num_steps_per_env": agent_cfg.num_steps_per_env,
                "num_learning_epochs": agent_cfg.algorithm.num_learning_epochs,
                "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
                "samples_per_optimizer_mini_batch": (
                    env.num_envs
                    * agent_cfg.num_steps_per_env
                    // agent_cfg.algorithm.num_mini_batches
                ),
                "learning_rate": agent_cfg.algorithm.learning_rate,
                "max_grad_norm": agent_cfg.algorithm.max_grad_norm,
                "save_interval": agent_cfg.save_interval,
            },
        )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
