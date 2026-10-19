"""Bounded config -> reset/step -> native resume validation; no long training."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Ladder/scripts/rsl_rl"))
import train
import pytest
import torch

native_learn = train.OnPolicyRunner.learn


def validated_learn(self, num_learning_iterations, init_at_random_ep_len=False):
    assert self.env.num_envs == 8 and num_learning_iterations == 2
    payload = torch.load(train.args_cli.checkpoint, map_location="cpu", weights_only=False)
    assert self.current_learning_iteration == self.alg.num_updates == payload["num_updates"]
    assert self.alg.amp_training_updates == payload["amp_training_updates"]
    saved = self.alg.save()
    for key in ("policy_state_dict", "critic_state_dict", "amp_discriminator_state_dict",
                "amp_normalizer_state_dict"):
        for name, value in payload[key].items():
            torch.testing.assert_close(saved[key][name].cpu(), value, rtol=0, atol=0)
    assert saved["optimizer_state_dict"]["state"]
    assert saved["amp_discriminator_optimizer_state_dict"]["state"]
    base = self.env.unwrapped
    manager = base.reward_manager
    power_index = manager.active_terms.index("joint_mechanical_power")
    compute = manager.compute
    power_samples = []

    def checked_compute(dt):
        reward = compute(dt)
        data = base.scene["robot"].data
        power = (data.applied_torque * data.joint_vel).abs().sum(-1)
        assert power.shape == reward.shape == (8,) and torch.isfinite(reward).all()
        torch.testing.assert_close(manager._step_reward[:, power_index], -5e-5 * power)
        torch.testing.assert_close(reward, manager._step_reward.sum(-1) * dt, atol=2e-6, rtol=1e-5)
        power_samples.append(power.detach().cpu())
        return reward

    manager.compute = checked_compute
    self.alg.eval_mode()
    policy = self.alg.get_policy()
    obs = self.env.reset()
    policy.reset()
    with torch.inference_mode():
        for _ in range(24):
            assert all(torch.isfinite(value).all() for value in obs.values())
            action = policy(obs)
            assert action.shape == (8, 25) and torch.isfinite(action).all()
            obs, reward, dones, _ = self.env.step(action)
            assert dones.shape == (8,)
            policy.reset(dones)
    print("REG_ENERGY_RESET_STEP_PASS", {k: list(v.shape) for k, v in obs.items()}, flush=True)
    # IsaacLab delay buffers created during rollout are inference tensors;
    # reset them under the same context used by the native training rollout.
    with torch.inference_mode():
        self.env.reset()
        policy.reset()
    native_learn(self, num_learning_iterations, init_at_random_ep_len)
    assert self.alg.num_updates == payload["num_updates"] + 2
    assert self.alg.amp_training_updates > payload["amp_training_updates"]
    power = torch.cat(power_samples)
    result = {
        "resume_checkpoint": str(train.args_cli.checkpoint),
        "start_num_updates": payload["num_updates"], "end_num_updates": self.alg.num_updates,
        "last_checkpoint_iteration": self.current_learning_iteration,
        "num_envs": 8, "reset_step_smoke_steps": 24, "training_updates": 2,
        "reward_compute_checks": len(power_samples), "mean_power_w": float(power.mean()),
        "mean_power_penalty_rate": float(-5e-5 * power.mean()),
        "log_dir": self.logger.log_dir,
    }
    (Path(__file__).parent / "validation.json").write_text(json.dumps(result, indent=2))
    print("REG_ENERGY_RESUME_PASS", result, flush=True)


try:
    result = pytest.main([
        str(ROOT / "Ladder/tests/test_end_to_end_visual_env_config.py"), "-q", "-k",
        "reg2x_energy_variant or double_reference_variant or pause_decay_variant",
    ])
    print("REG_ENERGY_CONFIG_TEST_EXIT", int(result), flush=True)
    if result:
        raise RuntimeError("Config tests failed; will not start environment smoke")
    train.OnPolicyRunner.learn = validated_learn
    train.main()
finally:
    train.simulation_app.close()
