"""RSL-RL configuration for the exact-observation DAgger control."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class CR1SameInputDaggerAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name: str = (
        "ladder.tasks.ladder_climb.config.cr1.dagger.same_input_algorithm:"
        "CR1SameInputDagger"
    )
    teacher_checkpoint: str = ""
    expected_observation_dim: int = 780
    diagnostic_snapshot_interval: int = 500
    diagnostic_sample_envs: int = 16


@configclass
class CR1SameInputDaggerRunnerCfg(RslRlDistillationRunnerCfg):
    """Canonical action-MSE DAgger with the accepted teacher architecture."""

    use_same_input_dagger_runner: bool = True
    seed: int = 42
    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 500
    experiment_name: str = "cr1_zdw0824_same_input_dagger_control"
    empirical_normalization: bool = False
    clip_actions: float | None = None

    # IsaacLab's public schema calls the student observation set "policy".
    # rsl_rl_compat translates it to rsl-rl 5's "student" set while preserving
    # this deliberate mapping of both networks to the one physical tensor.
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "teacher": ["policy"],
    }
    policy: RslRlDistillationStudentTeacherCfg = (
        RslRlDistillationStudentTeacherCfg(
            init_noise_std=0.1,
            noise_std_type="scalar",
            student_obs_normalization=False,
            teacher_obs_normalization=False,
            student_hidden_dims=[512, 256, 256],
            teacher_hidden_dims=[512, 256, 256],
            activation="elu",
        )
    )
    algorithm: CR1SameInputDaggerAlgorithmCfg = (
        CR1SameInputDaggerAlgorithmCfg(
            num_learning_epochs=1,
            learning_rate=1.0e-3,
            gradient_length=1,
            max_grad_norm=1.0,
            optimizer="adam",
            loss_type="mse",
        )
    )


__all__ = [
    "CR1SameInputDaggerAlgorithmCfg",
    "CR1SameInputDaggerRunnerCfg",
]
