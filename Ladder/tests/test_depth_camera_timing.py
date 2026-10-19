"""Exercise the real observation terms without starting the Isaac Sim app."""

import ast
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def terms(monkeypatch):
    # Only isolate simulator imports; execute the complete production classes.
    path = Path(__file__).resolve().parents[1] / (
        "source/ladder/ladder/tasks/ladder_climb/mdp/observations.py"
    )
    tree = ast.parse(path.read_text())
    names = {
        "warp_visual_depth",
        "cr1_transformer_vision_history",
        "cr1_transformer_vision_depth_updated",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.device = env.device
            self.num_envs = env.num_envs

    class CameraCfg:
        def __init__(self, **kwargs):
            self.output_height = kwargs["height"]
            self.output_width = kwargs["width"]

    class Renderer:
        def __init__(self, cfg, **kwargs):
            self.cfg = cfg
            self.calls = []

        def render(self, **kwargs):
            self.calls.append(kwargs)
            positions = kwargs["camera_positions_w"]
            depth = positions[:, 0, None, None].expand(
                -1, self.cfg.output_height, self.cfg.output_width
            ).clone()
            return SimpleNamespace(depth=depth, hit_class=torch.zeros_like(depth))

    ground = ModuleType("ladder.tasks.ladder_climb.config.cr1.wood_step_env_cfg")
    ground.CR1_GROUND_Z = 0.0
    monkeypatch.setitem(__import__("sys").modules, ground.__name__, ground)
    namespace = {
        "math": math, "torch": torch, "ManagerTermBase": ManagerTermBase,
        "ManagerBasedEnv": object, "WarpCameraCfg": CameraCfg,
        "CR1WarpBatchCamera": Renderer, "CR1_URDF": "unused",
        "WOOD_STEP_RUNG_BOX_NAMES": (), "RUNG": 1, "ROBOT": 2,
        "_cr1_wood_step_box_specs": lambda name: (),
        "_visual_depth_augmentation_cfg": lambda params: SimpleNamespace(min_depth=0.1, max_depth=2.0),
        "sample_uniform": lambda low, high, shape, device: torch.zeros(shape, device=device),
        "quat_from_euler_xyz": lambda *args: torch.tensor([[1., 0., 0., 0.]]).expand(args[0].numel(), -1),
        "combine_frame_transforms": lambda p, q, lp, lq: (p + lp, q),
        "convert_camera_frame_orientation_convention": lambda q, **kwargs: q,
        "process_warp_visual_depth": lambda depth, hits, **kwargs: (depth[:, None], None),
        "cr1_transformer_vision_proprio_frame": lambda env, **kwargs: torch.full(
            (env.num_envs, 85), float(env.common_step_counter)
        ),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def make_env():
    robot = SimpleNamespace(
        body_names=["d435_head_link"],
        data=SimpleNamespace(
            body_pos_w=torch.zeros(2, 1, 3),
            body_quat_w=torch.tensor([[[1., 0., 0., 0.]]]).expand(2, -1, -1),
        ),
    )
    ladder = SimpleNamespace(
        object_names=[],
        data=SimpleNamespace(object_pos_w=torch.zeros(2, 0, 3), object_quat_w=torch.zeros(2, 0, 4)),
    )
    return SimpleNamespace(
        num_envs=2, device="cpu", step_dt=0.02, common_step_counter=0,
        scene=SimpleNamespace(articulations={"robot": robot}, rigid_object_collections={"ladder": ladder}),
    )


def make_camera(terms, env, period):
    return terms.warp_visual_depth(
        SimpleNamespace(params={"width": 3, "height": 2, "update_period": period}), env
    )


@pytest.mark.parametrize("period,expected_count", [(1 / 30, 30), (0.1, 10), (0.0, 50)])
def test_camera_clock_keeps_fractional_phase_and_deduplicates_queries(terms, period, expected_count):
    env = make_env()
    camera = make_camera(terms, env, period)
    camera(env)
    capture_steps = []
    for step in range(1, 51):
        env.common_step_counter = step
        env.scene.articulations["robot"].data.body_pos_w[:, 0, 0] = step
        previous_calls = len(camera._renderer.calls)
        image = camera(env).clone()
        if len(camera._renderer.calls) > previous_calls:
            capture_steps.append(step)
        torch.testing.assert_close(camera(env), image)
    assert len(camera._renderer.calls) == expected_count + 1  # initial exposure
    assert len(capture_steps) == expected_count
    if period == 1 / 30:
        assert capture_steps[:6] == [2, 4, 5, 7, 9, 10]
    if period == 0.1:
        assert capture_steps == list(range(5, 51, 5))


def test_only_reset_env_refreshes_between_ticks_and_clock_is_not_rephased(terms):
    env = make_env()
    camera = make_camera(terms, env, 1 / 30)
    camera(env)
    env.common_step_counter = 1
    env.scene.articulations["robot"].data.body_pos_w[:, 0, 0] = torch.tensor([7., 9.])
    camera.reset([0])
    image = camera(env)
    assert image[:, 0, 0, 0].tolist() == [7., 0.]
    assert camera._renderer.calls[-1]["env_ids"].tolist() == [0]
    assert env._cr1_visual_depth_capture_step.tolist() == [1, 0]
    assert env._cr1_visual_depth_frame_id.tolist() == [1, 0]
    camera(env)
    assert len(camera._renderer.calls) == 2
    env.common_step_counter = 2
    assert camera(env)[:, 0, 0, 0].tolist() == [7., 9.]
    assert camera._renderer.calls[-1]["env_ids"] is None


def test_same_step_reset_forces_new_frame_without_refreshing_other_env(terms):
    env = make_env()
    camera = make_camera(terms, env, 1 / 30)
    camera(env)
    env.scene.articulations["robot"].data.body_pos_w[:, 0, 0] = 5
    camera.reset([1])
    assert camera(env)[:, 0, 0, 0].tolist() == [0., 5.]
    assert env._cr1_visual_depth_frame_id.tolist() == [0, 1]


@pytest.mark.parametrize("period", [-1., float("nan"), float("inf"), 0.01])
def test_invalid_or_faster_than_policy_camera_rate_is_rejected(terms, period):
    with pytest.raises(ValueError, match="Warp camera"):
        make_camera(terms, make_env(), period)


def test_history_uses_distinct_camera_frames_and_resets_without_episode_leak(terms):
    env = make_env()
    camera = make_camera(terms, env, 1 / 30)
    params = dict(num_proprio_frames=9, num_depth_frames=2, depth_height=2, depth_width=3)
    history = terms.cr1_transformer_vision_history(SimpleNamespace(params=params), env)
    camera(env)
    history(env, **params)
    assert terms.cr1_transformer_vision_depth_updated(env).tolist() == [[1.0], [1.0]]
    for step in (1, 2, 3):
        env.common_step_counter = step
        env.scene.articulations["robot"].data.body_pos_w[:, 0, 0] = step
        camera(env)
        history(env, **params)
        expected_update = 1.0 if step == 2 else 0.0
        assert terms.cr1_transformer_vision_depth_updated(env).tolist() == [
            [expected_update],
            [expected_update],
        ]
    assert history._proprio_history[0, -3:, 0].tolist() == [1., 2., 3.]
    # At step 3 no new camera frame: do not push [2, 2] into the history.
    assert history._depth_history[:, :, 0, 0].tolist() == [[0., 2.], [0., 2.]]
    history(env, **params)
    assert history._proprio_history[0, -3:, 0].tolist() == [1., 2., 3.]
    env.scene.articulations["robot"].data.body_pos_w[0, 0, 0] = 8
    camera.reset([0])
    history.reset([0])
    camera(env)
    history(env, **params)
    assert terms.cr1_transformer_vision_depth_updated(env).tolist() == [[1.0], [0.0]]
    assert history._depth_history[:, :, 0, 0].tolist() == [[8., 8.], [0., 2.]]
    assert history._proprio_history[0, :, 0].tolist() == [3.] * 9
    env.common_step_counter = 4
    env.scene.articulations["robot"].data.body_pos_w[:, 0, 0] = 4
    camera(env)
    history(env, **params)
    assert history._depth_history[:, :, 0, 0].tolist() == [[8., 4.], [2., 4.]]


def test_time_sampled_history_uses_explicit_10hz_proprio_and_30hz_depth_offsets(terms):
    """The end-to-end contract samples history by time, not by call count."""

    env = make_env()
    camera = make_camera(terms, env, 1 / 30)
    params = dict(
        num_proprio_frames=10,
        num_depth_frames=3,
        depth_height=2,
        depth_width=3,
        proprio_interval_s=0.1,
        depth_history_offsets_s=(0.0, 0.2, 0.4),
        history_buffer_capacity=64,
    )
    history = terms.cr1_transformer_vision_history(SimpleNamespace(params=params), env)

    camera(env)
    history(env, **params)
    for step in range(1, 51):
        env.common_step_counter = step
        env.scene.articulations["robot"].data.body_pos_w[:, 0, 0] = step
        camera(env)
        history(env, **params)

    # Ten samples at 0.1 s spacing cover 0.9 s, so at t=1.0 the
    # oldest-to-newest proprio frames are steps 5, 10, ..., 50.
    assert history._proprio_history[0, :, 0].tolist() == list(range(5, 51, 5))
    # Camera timestamps are fractional-clock exposures.  At t=1.0 the
    # nearest available exposures to 1.0, 0.8, and 0.6 s are 50, 40, and 30
    # policy steps respectively; output is chronological (oldest first).
    assert history._depth_history[0, :, 0, 0].tolist() == [30.0, 40.0, 50.0]
    # The command is sampled from its own ring at exactly the same 0.1 s
    # targets as proprioception, rather than copied from the latest call.
    assert history._proprio_history[0, :, -1].tolist() == list(range(5, 51, 5))

    # A partial reset clears all three rings for only the selected environment.
    env.scene.articulations["robot"].data.body_pos_w[0, 0, 0] = 100.0
    camera.reset([0])
    history.reset([0])
    camera(env)
    history(env, **params)
    assert history._proprio_history[0, :, 0].tolist() == [50.0] * 10
    assert history._proprio_history[0, :, -1].tolist() == [50.0] * 10
    assert history._depth_history[0, :, 0, 0].tolist() == [100.0] * 3
    assert history._proprio_history[1, :, 0].tolist() == list(range(5, 51, 5))


@pytest.mark.parametrize("same_step", [False, True])
@pytest.mark.parametrize("reset_all", [False, True])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_command_ring_reset_clears_metadata_after_wraparound(
    terms, monkeypatch, same_step, reset_all, direction,
):
    env = make_env()
    camera = make_camera(terms, env, 1 / 30)
    command = torch.full((env.num_envs,), direction)

    def proprio_frame(env, **kwargs):
        frame = torch.full((env.num_envs, 85), float(env.common_step_counter))
        frame[:, -1] = command
        return frame

    monkeypatch.setitem(
        terms.cr1_transformer_vision_history.__call__.__globals__,
        "cr1_transformer_vision_proprio_frame", proprio_frame,
    )
    params = dict(
        num_proprio_frames=10, num_depth_frames=3, depth_height=2, depth_width=3,
        proprio_interval_s=0.1, depth_history_offsets_s=(0.0, 0.2, 0.4),
        history_buffer_capacity=64,
    )
    history = terms.cr1_transformer_vision_history(SimpleNamespace(params=params), env)
    for step in range(81):
        env.common_step_counter = step
        camera(env)
        history(env, **params)
    assert history._command_count.tolist() == [64, 64]
    untouched = {
        name: getattr(history, name)[1].clone() for name in (
            "_command_buffer", "_command_timestamps", "_command_write_index",
            "_command_count", "_proprio_write_index", "_proprio_history",
        )
    }
    selection = slice(None) if reset_all else [0]
    command[selection] = -direction
    if not same_step:
        env.common_step_counter += 1
    camera.reset(None if reset_all else [0])
    history.reset(None if reset_all else [0])
    assert (history._command_timestamps[selection] == -1).all()
    assert (history._command_write_index[selection] == 0).all()
    assert (history._command_count[selection] == 0).all()
    camera(env)
    history(env, **params)
    assert (history._proprio_history[selection, :, -1] == -direction).all()
    assert (env._cr1_transformer_vision_history_cache["command_history"][selection] == -direction).all()
    assert (history._command_count[selection] == 1).all()
    # Same-step partial reset must not append a duplicate/noisy sample to
    # another environment or displace its oldest retained history sample.
    if same_step and not reset_all:
        for name, value in untouched.items():
            torch.testing.assert_close(getattr(history, name)[1], value, rtol=0, atol=0)
    first_step = env.common_step_counter
    for step in range(first_step + 1, first_step + 47):
        env.common_step_counter = step
        camera(env)
        history(env, **params)
        assert (history._proprio_history[selection, :, -1] == -direction).all()


def test_depth_augmentation_is_only_sampled_for_new_exposures(terms, monkeypatch):
    env = make_env()
    camera = make_camera(terms, env, 1 / 30)
    calls = []

    def augment(depth, hits, **kwargs):
        calls.append(depth.shape[0])
        return torch.rand_like(depth[:, None]), None

    monkeypatch.setitem(camera._capture.__func__.__globals__, "process_warp_visual_depth", augment)
    image = camera(env).clone()
    env.common_step_counter = 1
    torch.testing.assert_close(camera(env), image)
    assert calls == [2]
    camera.reset([0])
    refreshed = camera(env)
    torch.testing.assert_close(refreshed[1], image[1])
    assert calls == [2, 1]
