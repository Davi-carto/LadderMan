"""Run focused regressions with the actual IsaacLab/Isaac Sim imports loaded."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Ladder/scripts/rsl_rl"))
import play_transformer_vision_ppo as play
import pytest

try:
    result = pytest.main([
        str(ROOT / "Ladder/tests/test_contact_task_amp.py"),
        str(ROOT / "Ladder/tests/test_support_contact_phases.py"),
        str(ROOT / "Ladder/tests/test_reference_guidance.py"),
        str(ROOT / "Ladder/tests/test_contact_task_foot_safety.py"),
        *[str(ROOT / "Ladder/tests/test_end_to_end_visual_env_config.py") + "::" + name
          for name in (
              "test_body_goal_experiment_only_adds_goal_and_shaping_to_phased_task",
              "test_foot_safety_only_adds_three_reward_rates_and_keeps_actor_contract",
              "test_reference_guidance_preserves_policy_amp_contacts_and_safety",
              "test_pause_decay_variant_preserves_all_other_environment_and_runner_settings",
              "test_double_reference_variant_only_scales_six_weights_and_keeps_resume_contract")],
        "-q",
    ])
    print("PHASE_TEST_EXIT", int(result), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    play.simulation_app.close()
raise SystemExit(result)
