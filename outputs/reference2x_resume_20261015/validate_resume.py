"""Run isolated-config checks, then the supplied bounded native train command."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Ladder/scripts/rsl_rl"))
import train
import pytest

try:
    result = pytest.main([
        str(ROOT / "Ladder/tests/test_end_to_end_visual_env_config.py"),
        "-q", "-k", "double_reference_variant or pause_decay_variant",
    ])
    if result:
        raise RuntimeError(f"Configuration regression checks failed: {result}")
    train.main()
finally:
    train.simulation_app.close()
