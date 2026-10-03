from hashlib import sha256

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_data_pipeline import make_root, open_dataset

from patchwam.data import FeatureScaler


def test_fitted_statistics_preserve_episode_stepwise_quantile_envelope(tmp_path):
    root, _ = make_root(tmp_path)
    dataset = open_dataset(root, None, num_frames=5)
    statistics = dataset.statistics["action"]["default"]
    values = torch.arange(20, dtype=torch.float32)
    windows = values[(torch.arange(20)[:, None] + torch.arange(4)).clamp_max(19)]
    expected_mean = windows.mean()
    expected_variance = (windows.var(0) + (windows.mean(0) - expected_mean).square()).mean()
    torch.testing.assert_close(statistics["global_mean"][0], expected_mean)
    torch.testing.assert_close(statistics["global_std"][0].square(), expected_variance)
    assert statistics["global_q01"][0].item() == pytest.approx(0.19)
    assert statistics["global_q99"][0].item() == pytest.approx(19)
    # Window replication must not change quantiles or variance.
    assert statistics["global_q01"][0] != torch.quantile(windows.flatten(), 0.01)


def test_qwen3_cache_uses_native_model_inputs(tmp_path):
    root, stats = make_root(tmp_path)
    prompt = open_dataset(root, stats)[0]["prompt"]
    cache = tmp_path / "text-cache"
    cache.mkdir()
    context = torch.randn(8, 32)
    mask = torch.tensor([True] * 6 + [False] * 2)
    path = cache / f"{sha256(prompt.encode()).hexdigest()}.qwen3_flux2_len8.pt"
    torch.save({"text_hidden_states": context, "text_attention_mask": mask}, path)
    dataset = open_dataset(root, stats, qwen_text_cache_dir=cache,
                           qwen_text_cache_format="qwen3_flux2", qwen_context_len=8)
    sample = dataset[0]
    torch.testing.assert_close(sample["text_tokens"], context)
    torch.testing.assert_close(sample["text_valid"], mask)
    assert sample["text_cache_format"] == "qwen3_flux2"
    assert "text_hidden_states" not in sample
    torch.save({"text_hidden_states": context + 1, "text_attention_mask": mask}, path)
    changed = open_dataset(root, stats, qwen_text_cache_dir=cache,
                           qwen_text_cache_format="qwen3_flux2", qwen_context_len=8)
    assert changed.fingerprint != dataset.fingerprint


def test_video_file_replacement_changes_dataset_fingerprint(tmp_path):
    root, stats = make_root(tmp_path, video=True)
    original = open_dataset(root, stats)
    path = root / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    path.write_bytes(path.read_bytes() + b"\0")
    assert open_dataset(root, stats).fingerprint != original.fingerprint


def test_replaced_parquet_is_not_served_from_a_stale_table_cache(tmp_path):
    root, stats = make_root(tmp_path)
    original = open_dataset(root, stats)
    assert original[0]["action"][0, 0].item() == -1
    path = root / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(path)
    actions = [[value + 10 for value in row] for row in table["action"].to_pylist()]
    table = table.set_column(table.column_names.index("action"), "action", pa.array(actions))
    pq.write_table(table, path)
    changed = open_dataset(root, stats)
    assert changed.fingerprint != original.fingerprint
    assert changed[0]["action"][0, 0].item() == pytest.approx(0)


def test_named_embodiment_statistics_preserve_flattened_global_recipe(tmp_path):
    root, _ = make_root(tmp_path)
    path = root / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(path).append_column("embodiment", pa.array([7] * 20))
    pq.write_table(table, path)
    dataset = open_dataset(root, None, num_frames=5)
    statistics = dataset.statistics["embodiments"]["7"]["action"]["default"]
    windows = torch.arange(20, dtype=torch.float32)[(torch.arange(20)[:, None] + torch.arange(4)).clamp_max(19)]
    torch.testing.assert_close(statistics["global_q01"][0], torch.quantile(windows.flatten(), 0.01))
    torch.testing.assert_close(statistics["global_std"][0], windows.flatten().std())
    assert dataset[0]["embodiment"] == "7"
    assert dataset[0]["action"].isfinite().all()


def test_offset_video_timestamps_do_not_silently_choose_nearby_frames(tmp_path):
    root, stats = make_root(tmp_path, video=True)
    path = root / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(path)
    times = [value + 0.001 for value in table["timestamp"].to_pylist()]
    table = table.set_column(table.column_names.index("timestamp"), "timestamp", pa.array(times))
    pq.write_table(table, path)
    with pytest.raises(RuntimeError, match="Could not decode requested timestamps"):
        open_dataset(root, stats)[0]
    assert open_dataset(root, stats, lerobot_tolerance_s=0.002)[0]["video"].isfinite().all()


def test_invalid_episode_timestamp_cadence_is_rejected(tmp_path):
    root, stats = make_root(tmp_path)
    path = root / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(path)
    times = table["timestamp"].to_pylist()
    times[1] += 0.01
    table = table.set_column(table.column_names.index("timestamp"), "timestamp", pa.array(times))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="timestamps do not match dataset FPS"):
        open_dataset(root, stats)


@pytest.mark.parametrize("statistics,mode", [
    ({"min": [0], "max": [float("nan")]}, "min/max"),
    ({"min": [2], "max": [1]}, "min/max"),
    ({"mean": [0], "std": [-1]}, "z-score"),
    ({"min": [0], "max": [1]}, "nan/2"),
])
def test_invalid_scaling_statistics_fail_before_preprocessing(statistics, mode):
    with pytest.raises(ValueError):
        FeatureScaler(statistics, mode)
