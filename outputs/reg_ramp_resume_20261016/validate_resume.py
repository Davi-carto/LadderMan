"""Real-runtime tests, eight-env step sanity, and two native resume updates."""
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
    alg = self.alg
    payload = torch.load(train.args_cli.checkpoint, map_location="cpu", weights_only=False)
    start = payload["num_updates"]
    assert self.current_learning_iteration == alg.num_updates == alg._regulation_start_update == start
    schedule = alg.regulation_reward_schedule
    manager = self.env.unwrapped.reward_manager
    curve = []
    for elapsed in (0, 1250, 2500, 4500):
        alg.num_updates = start + elapsed
        alg._apply_regulation_schedule()
        factor = 1.0 + 1.5 * min(elapsed / 2500, 1.)
        for name, weight in schedule["base_weights"].items():
            assert manager.get_term_cfg(name).weight == weight * factor
        assert manager.get_term_cfg("joint_mechanical_power").weight == -5e-5
        assert manager.get_term_cfg("reference_joint_lower").weight == .25
        assert manager.get_term_cfg("joint_limit").weight == -10.
        curve.append([elapsed, factor])
    alg.num_updates = start
    alg._apply_regulation_schedule()
    compute = manager.compute
    checks = []

    def checked_compute(dt):
        rewards = compute(dt)
        multiplier = 1.0 + 1.5 * min((alg.num_updates - start) / 2500, 1.)
        for name, weight in schedule["base_weights"].items():
            assert manager.get_term_cfg(name).weight == weight * multiplier
        data = self.env.unwrapped.scene["robot"].data
        power = (data.applied_torque * data.joint_vel).abs().sum(-1)
        idx = manager.active_terms.index("joint_mechanical_power")
        torch.testing.assert_close(manager._step_reward[:, idx], -5e-5 * power)
        torch.testing.assert_close(rewards, manager._step_reward.sum(-1) * dt, atol=2e-6, rtol=1e-5)
        assert rewards.shape == (8,) and torch.isfinite(rewards).all()
        checks.append(multiplier)
        return rewards

    manager.compute = checked_compute
    alg.eval_mode()
    policy = alg.get_policy()
    with torch.inference_mode():
        obs = self.env.reset()
        policy.reset()
        for _ in range(24):
            assert all(torch.isfinite(value).all() for value in obs.values())
            actions = policy(obs)
            assert actions.shape == (8, 25) and torch.isfinite(actions).all()
            obs, _, dones, _ = self.env.step(actions)
            assert dones.shape == (8,)
            policy.reset(dones)
        self.env.reset()
        policy.reset()
    print("REG_RAMP_RESET_STEP_PASS", flush=True)
    native_learn(self, num_learning_iterations, init_at_random_ep_len)
    assert checks == [1.] * 48 + [1.0006] * 24
    saved = alg.save()
    assert saved["regulation_reward_schedule"]["start_update"] == start
    assert alg.num_updates == start + 2
    # The explicit inference smoke above creates actor-only inference buffers;
    # resetting those on same-process reload needs the same tensor context.
    with torch.inference_mode():
        alg.load(saved)
    assert alg._regulation_start_update == start
    assert alg._regulation_metrics["regulation_multiplier"] == 1.0012
    result = {"curve": curve, "start_update": start, "end_update": alg.num_updates,
              "next_rollout_multiplier_after_reload": alg._regulation_metrics["regulation_multiplier"],
              "reward_compute_checks": len(checks), "log_dir": self.logger.log_dir}
    (Path(__file__).parent / "validation.json").write_text(json.dumps(result, indent=2))
    print("REG_RAMP_RESUME_PASS", result, flush=True)


try:
    result = pytest.main([
        str(ROOT / "Ladder/tests/test_contact_task_amp.py"),
        str(ROOT / "Ladder/tests/test_ladder_rewards.py"),
        *[str(ROOT / "Ladder/tests/test_end_to_end_visual_env_config.py") + "::" + name
          for name in (
              "test_pause_decay_variant_preserves_all_other_environment_and_runner_settings",
              "test_double_reference_variant_only_scales_six_weights_and_keeps_resume_contract",
              "test_reg2x_energy_variant_only_changes_control_costs_and_keeps_resume_contract",
              "test_reg_ramp_energy_variant_preserves_baseline_weights_and_only_enables_schedule")],
        "-q",
    ])
    print("REG_RAMP_TEST_EXIT", int(result), flush=True)
    if result:
        raise RuntimeError("Tests failed; environment validation not started")
    train.OnPolicyRunner.learn = validated_learn
    train.main()
finally:
    train.simulation_app.close()
