import numpy as np
import pytest
import torch
from torch import nn

from patchwam.data import SampleProcessor
from patchwam.evaluation.policy import OnlinePolicy


class RecordingPolicy(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.width, self.batches, self.draws = width, [], []

    def sample_actions(self, batch, *, horizon, steps, guidance_scale, generator):
        assert "future_tokens" not in batch and "action" not in batch and "subtask" not in batch
        assert batch["video"].shape[2] == 1
        assert steps == 10 and guidance_scale == 1
        self.batches.append({key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()})
        self.draws.append(torch.rand((), generator=generator).item())
        values = 0.5 + torch.arange(horizon).float() * 0.1
        return {"action": values[None, :, None].expand(1, horizon, self.width).clone()}


def feature_processor(action_width, state_width, mode="z-score", relative=False):
    shape = {
        "images": [{"key": name, "shape": [3, 8, 8]} for name in ("head", "left", "right")],
        "action": [{"key": "default", "shape": action_width}],
        "state": [{"key": "default", "shape": state_width}],
    }
    processor = SampleProcessor(shape, norm_default_mode=mode, relative_joint_keys=["default"] if relative else None)
    statistics = {}
    for field, width, mean, spread in (("action", action_width, 10, 2), ("state", state_width, 20, 4)):
        center = np.arange(width) + mean
        statistics[field] = {"default": {"global_mean": center.tolist(), "global_std": [spread] * width,
                                         "global_min": (center - spread * 0.5).tolist(), "global_max": (center + spread * 1.5).tolist()}}
    processor.set_normalizer_from_stats(statistics)
    return processor


def observation(state_width, timestamp=0, *, value=100, state_shift=0, instruction="place object"):
    return {
        "images": {name: np.full((8, 8, 3), value, dtype=np.uint8) for name in ("head", "left", "right")},
        "state": {"default": np.arange(state_width, dtype=np.float32) + 24 + state_shift},
        "instruction": instruction, "timestamp": timestamp,
    }


def adapter(model, processor, **options):
    return OnlinePolicy(model, processor, video_size=(8, 24), concat_multi_camera="horizontal",
                        head_camera_key="head", vl_image_size=8, **options)


@pytest.mark.parametrize("action_width,state_width,mode,relative", [
    (14, 14, "z-score", True), (12, 16, "z-score", False), (7, 8, "min/max", False),
])
def test_benchmark_widths_normalization_and_chunk_replanning(action_width, state_width, mode, relative):
    processor = feature_processor(action_width, state_width, mode, relative)
    model = RecordingPolicy(action_width)
    policy = adapter(model, processor, replan_horizon=3)
    policy.reset("episode")
    first_state = np.arange(state_width, dtype=np.float32) + 24
    for step in range(3):
        action = policy.act(observation(state_width, step / 20, state_shift=step * 0.25))
        expected = np.arange(action_width) + 10 + (0.5 + step * 0.1 + (0.5 if mode == "min/max" else 0)) * 2
        if relative:
            expected = expected + first_state
        np.testing.assert_allclose(action, expected, atol=2e-6)
        assert action.shape == (action_width,) and action.dtype == np.float32
    assert len(model.batches) == 1
    torch.testing.assert_close(model.batches[0]["proprio"], torch.full((1, state_width), 0.5 if mode == "min/max" else 1.0))
    assert "place object" in model.batches[0]["prompt"][0]
    replanned = policy.act(observation(state_width, 3 / 20, state_shift=0.75))
    expected = np.arange(action_width) + (12 if mode == "min/max" else 11)
    if relative:
        expected = expected + first_state + 0.75
    np.testing.assert_allclose(replanned, expected, atol=2e-6)
    assert len(model.batches) == 2 and not model.training


def test_every_step_records_history_and_reset_clears_queue_clock_and_rng():
    model = RecordingPolicy(14)
    policy = adapter(model, feature_processor(14, 14), replan_horizon=3,
                     history_slots=2, history_tolerance_s=0.01, separate_camera_views=True)
    policy.reset("first")
    for timestamp, value in ((0, 0), (0.1, 50), (0.2, 100), (1.1, 150)):
        policy.act(observation(14, timestamp, value=value))
    batch = model.batches[-1]
    assert batch["camera_video"].shape == (1, 3, 3, 1, 8, 8)
    assert batch["history_valid"].tolist() == [[True, False]]
    assert batch["vl_history_valid"].tolist() == [[True, False]]
    torch.testing.assert_close(batch["history_video"][0, 0], torch.full((3, 8, 8), 50 / 255 * 2 - 1))
    assert batch["history_video"][0, 1].count_nonzero() == 0
    torch.testing.assert_close(batch["vl_history"][0, 0], batch["history_video"][0, 0])
    first_draw = model.draws[0]
    policy.reset("second")
    action = policy.act(observation(14, 0, value=127))
    assert model.draws[-1] == first_draw
    assert not model.batches[-1]["history_valid"].any()
    np.testing.assert_allclose(action, np.arange(14) + 11)
    with pytest.raises(ValueError, match="once"):
        policy.act(observation(14, 0, value=127))
    with pytest.raises(ValueError, match="increase"):
        policy.observe(observation(14, -0.1))


def test_observe_then_act_deduplicates_and_instruction_change_replans():
    model = RecordingPolicy(14)
    policy = adapter(model, feature_processor(14, 14), replan_horizon=16, history_slots=1)
    with pytest.raises(RuntimeError, match="Reset"):
        policy.act(observation(14))
    policy.reset("episode")
    current = observation(14)
    policy.observe(current)
    policy.act(current)
    assert len(policy.history.frames) == 1
    changed = observation(14, 0.05, instruction="pick object")
    policy.act(changed)
    assert len(model.batches) == 2
    assert "pick object" in model.batches[-1]["prompt"][0]
    with pytest.raises(ValueError, match="same images and state"):
        policy.observe(observation(14, 0.05, state_shift=1, instruction="pick object"))


def test_action_field_order_and_relative_anchor():
    shape = {
        "images": [{"key": "head", "shape": [3, 8, 8]}],
        "action": [{"key": "left", "shape": 2}, {"key": "right", "shape": 1}],
        "state": [{"key": "left", "shape": 2}, {"key": "right", "shape": 1}],
    }
    processor = SampleProcessor(shape, norm_default_mode="z-score", relative_joint_keys=["left"])
    stats = {field: {"left": {"global_mean": [0, 0], "global_std": [1, 1]},
                     "right": {"global_mean": [10], "global_std": [2]}} for field in ("action", "state")}
    processor.set_normalizer_from_stats(stats)
    policy = OnlinePolicy(RecordingPolicy(3), processor, video_size=(8, 8), concat_multi_camera="horizontal", vl_image_size=8)
    policy.reset("episode")
    data = {"images": {"head": np.zeros((3, 8, 8), dtype=np.float32)},
            "state": {"left": np.array([1, 2]), "right": np.array([12])},
            "instruction": "place object", "timestamp": 0}
    np.testing.assert_allclose(policy.act(data), [1.5, 2.5, 11])


def test_invalid_observations_and_nonfinite_model_actions_fail():
    model = RecordingPolicy(14)
    policy = adapter(model, feature_processor(14, 14))
    policy.reset("episode")
    bad = observation(14)
    bad["images"]["head"] = np.full((3, 8, 8), 2, dtype=np.float32)
    with pytest.raises(ValueError, match=r"in \[0,1\]"):
        policy.observe(bad)
    model.sample_actions = lambda *args, **kwargs: {"action": torch.full((1, 16, 14), float("nan"))}
    with pytest.raises(ValueError, match="finite"):
        policy.act(observation(14))


def test_raw_width_conversion_uses_processor_and_validates_decoded_width():
    class ConvertedProcessor(SampleProcessor):
        def transform_fields(self, fields):
            reduced = {field: {key: value[..., :3] for key, value in values.items()} for field, values in fields.items()}
            return super().transform_fields(reduced)

        def decode_actions(self, actions, proprio, embodiment=None):
            decoded = super().decode_actions(actions, proprio, embodiment)
            decoded["default"] = torch.cat((decoded["default"], torch.full_like(actions[..., :1], 9)), -1)
            return decoded

    reference = feature_processor(3, 3)
    shape = reference.shape_meta
    for field in ("action", "state"):
        shape[field][0]["raw_shape"] = 4
    processor = ConvertedProcessor(shape, norm_default_mode="z-score")
    processor.set_normalizer_from_stats(reference.codec.statistics)
    model = RecordingPolicy(3)
    policy = adapter(model, processor)
    policy.reset("converted")
    action = policy.act(observation(4))
    np.testing.assert_allclose(action, [11, 12, 13, 9])
    torch.testing.assert_close(model.batches[0]["proprio"], torch.ones(1, 3))
    processor.decode_actions = lambda *args, **kwargs: {"default": torch.zeros(1, 16, 3)}
    policy.reset("bad inverse")
    with pytest.raises(ValueError, match="raw feature width"):
        policy.act(observation(4))


def test_per_embodiment_statistics_and_padded_state_match_training_processor():
    processor = feature_processor(14, 14, relative=True)
    statistics = processor.codec.statistics
    processor.proprio_output_dim = 80
    processor.action_output_dim = 80
    processor.set_normalizer_from_stats({"type": "per_embodiment", "embodiments": {"robot": statistics}})
    model = RecordingPolicy(80)
    policy = adapter(model, processor, replan_horizon=1)
    policy.reset("robot")
    data = observation(14)
    data["embodiment"] = "robot"
    action = policy.act(data)
    np.testing.assert_allclose(action, np.arange(14) * 2 + 35)
    expected = processor.preprocess({
        "state": {"default": torch.from_numpy(data["state"]["default"])[None]},
        "action": {"default": torch.zeros(1, 14)},
        "images": {key: torch.from_numpy(image)[None] for key, image in data["images"].items()},
        "task": data["instruction"], "embodiment": "robot", "action_is_pad": torch.zeros(1, dtype=torch.bool),
        "state_is_pad": torch.zeros(1, dtype=torch.bool), "image_is_pad": torch.zeros(1, dtype=torch.bool),
    })
    torch.testing.assert_close(model.batches[0]["proprio"], expected["proprio"])
    assert model.batches[0]["proprio"].shape == (1, 80)
    assert not processor.training
    with pytest.raises(ValueError, match="Embodiment"):
        policy.observe(observation(14, 0.05))


def test_global_statistics_accept_robotwin_style_embodiment():
    processor = feature_processor(14, 14, relative=True)
    model = RecordingPolicy(14)
    policy = adapter(model, processor)
    policy.reset("robotwin")
    data = observation(14)
    data["embodiment"] = "aloha-agilex"
    np.testing.assert_allclose(policy.act(data), np.arange(14) * 2 + 35)
    torch.testing.assert_close(model.batches[0]["proprio"], torch.ones(1, 14))
