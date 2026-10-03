import json
from io import BytesIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from patchwam.data import (
    ClipAugment,
    EpisodeDataset,
    FeatureScaler,
    RobotFeatureCodec,
    SampleProcessor,
    compose_cameras,
)

SHAPE_META = {
    "images": [{"key": "front", "raw_shape": [3, 8, 8], "shape": [3, 8, 8]}],
    "action": [{"key": "default", "raw_shape": 2, "shape": 2}],
    "state": [{"key": "default", "raw_shape": 3, "shape": 3}],
}


def make_root(tmp_path, video=False):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {"fps": 20, "total_episodes": 1, "chunks_size": 1000, "codebase_version": "v2.1",
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 20}) + "\n")
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "place the object"}) + "\n")
    table = {"action": [[float(index), float(index + 1)] for index in range(20)],
             "observation.state": [[float(index), float(index + 1), float(index + 2)] for index in range(20)],
             "timestamp": [index / 20 for index in range(20)], "task_index": [0] * 20}
    if video:
        import av
        path = root / "videos/chunk-000/observation.images.front/episode_000000.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=20)
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            for index in range(20):
                rgb = np.full((16, 16, 3), index * 10, dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    else:
        frames = []
        for index in range(20):
            buffer = BytesIO()
            Image.fromarray(np.full((8, 8, 3), index * 10, dtype=np.uint8)).save(buffer, format="PNG")
            frames.append({"bytes": buffer.getvalue(), "path": None})
        table["observation.images.front"] = frames
    pq.write_table(pa.table(table), root / "data/chunk-000/episode_000000.parquet")
    statistics = {field: {"default": {"global_min": [0] * width, "global_max": [20] * width,
                                      "global_q01": [0] * width, "global_q99": [20] * width}}
                  for field, width in (("action", 2), ("state", 3))}
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(statistics))
    return root, stats_path


def open_dataset(root, stats, **kwargs):
    return EpisodeDataset([root], SHAPE_META, pretrained_norm_stats=stats,
                          video_size=(8, 8), require_text_cache=False, **kwargs)


def test_parquet_images_to_loader_preserves_endpoints_horizon_and_masks(tmp_path):
    root, stats = make_root(tmp_path)
    dataset = open_dataset(root, stats)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["video"].shape == (2, 3, 2, 8, 8)
    assert batch["action"].shape == (2, 16, 2)
    assert batch["proprio"].shape == (2, 16, 3)
    assert batch["action_is_pad"].shape == (2, 16)
    assert batch["proprio_is_pad"].shape == (2, 17)
    assert batch["action_dim_is_pad"].shape == (2, 2)
    torch.testing.assert_close(batch["action"][0, :, 0], torch.arange(16) / 10 - 1)
    torch.testing.assert_close(batch["video"][0, :, 1], torch.full((3, 8, 8), 160 / 255 * 2 - 1))
    assert not batch["action_is_pad"].any()
    final = dataset[19]
    assert final["action_is_pad"].tolist() == [False] + [True] * 15
    assert final["image_is_pad"].tolist() == [False, True]


def test_nonidle_ranges_form_filtered_timeline_without_crossing_episodes(tmp_path):
    root, stats = make_root(tmp_path)
    path = tmp_path / "nonidle.json"
    path.write_text(json.dumps({"episodes": {"0": [[0, 2], [6, 10]]}}))
    dataset = open_dataset(root, stats, num_frames=5, nonidle_filter_path=path)
    assert len(dataset) == 6
    sample = dataset[0]
    torch.testing.assert_close(sample["action"][:, 0], torch.tensor([0, 1, 6, 7]) / 10 - 1)
    torch.testing.assert_close(sample["video"][:, 1], torch.full((3, 8, 8), 80 / 255 * 2 - 1))
    assert not sample["action_is_pad"].any()


def test_real_mp4_decode_obeys_endpoint_timestamps(tmp_path):
    root, stats = make_root(tmp_path, video=True)
    sample = open_dataset(root, stats)[0]
    assert sample["video"].shape == (3, 2, 8, 8)
    assert abs(sample["video"][:, 0].mean().item() + 1) < 0.03
    assert abs(sample["video"][:, 1].mean().item() - (160 / 255 * 2 - 1)) < 0.03


def test_delta_time_padding_keeps_absolute_gripper_and_dimension_mask(tmp_path):
    root, stats = make_root(tmp_path)
    processor = SampleProcessor(SHAPE_META, action_output_dim=4, proprio_output_dim=5,
                                delta_action_dim_mask={"default": [True, False]})
    sample = open_dataset(root, stats, processor=processor)[19]
    assert sample["action_dim_is_pad"].tolist() == [False, False, True, True]
    assert sample["proprio_dim_is_pad"].tolist() == [False, False, False, True, True]
    assert (sample["action"][1:, 0] == -1).all()
    assert (sample["action"][1:, 1] == 1).all()
    assert (sample["action"][:, 2:] == 0).all()
    assert (sample["proprio"][:, 3:] == 0).all()
    decoded = processor.decode_actions(sample["action"], sample["proprio"])["default"]
    assert decoded.shape == (16, 2)
    torch.testing.assert_close(decoded[1:, 0], torch.zeros(15))
    torch.testing.assert_close(decoded[1:, 1], torch.full((15,), 20.0))


@pytest.mark.parametrize("mode", ["min/max", "q01/q99", "z-score", "-2/2"])
def test_affine_roundtrip_including_constant_features(mode):
    stats = {"min": [-2, 3], "max": [2, 3], "q01": [-2, 3], "q99": [2, 3], "mean": [0, 3], "std": [1, 0.5]}
    scaler = FeatureScaler(stats, mode)
    value = torch.tensor([[1., 3.], [-1., 3.]])
    torch.testing.assert_close(scaler.decode(scaler.encode(value)), value)
    if mode in {"min/max", "q01/q99"}:
        assert (scaler.encode(value)[:, 1] == 0).all()


def test_robotwin_camera_order_and_compact_dimensions():
    cameras = [torch.full((2, 3, 8, 8), value) for value in (0., 0.5, 1.)]
    video = compose_cameras(cameras, "robotwin", (288, 256))
    assert video.shape == (3, 2, 288, 256)
    assert video[:, :, :192].mean().item() == -1
    assert video[:, :, 192:, :128].mean().item() == 0
    assert video[:, :, 192:, 128:].mean().item() == 1


def test_clip_augmentation_temporal_consistency_and_probability():
    clip = torch.linspace(0.1, 0.9, 3 * 12 * 12).reshape(1, 3, 12, 12).repeat(2, 1, 1, 1)
    assert ClipAugment(p=0)([clip])[0] is clip
    torch.manual_seed(17)
    augmented = ClipAugment(p=1, augment_types=["both"], color_jitter={"brightness": 0.2, "hue": 0.01},
                             gamma={"range": [0.9, 1.1]}, exposure={"ev_range": [-0.1, 0.1]},
                             gaussian_noise={"std": 0.01}, random_resized_crop={"scale": [0.9, 1], "ratio": "preserve"},
                             rotate={"degrees": 5, "fill": "mean"})([clip])[0]
    torch.testing.assert_close(augmented[0], augmented[1])
    assert 0 <= augmented.min() <= augmented.max() <= 1
    assert not torch.equal(clip, augmented)


def test_fitting_stats_and_corrupt_sample_error(tmp_path):
    root, _ = make_root(tmp_path)
    dataset = open_dataset(root, None)
    assert dataset.statistics["action"]["default"]["global_mean"].isfinite().all()
    assert dataset[0]["action"].isfinite().all()
    (root / "meta/tasks.jsonl").write_text("")
    invalid = open_dataset(root, None)
    with pytest.raises(KeyError, match="Missing task instruction"):
        invalid[0]


def test_fingerprint_tracks_selection_normalization_and_metadata(tmp_path):
    root, stats = make_root(tmp_path)
    original = open_dataset(root, stats)
    assert len(original.fingerprint) == 64
    assert open_dataset(root, stats).fingerprint == original.fingerprint
    assert open_dataset(root, stats, sample_index_stride=2).fingerprint != original.fingerprint
    payload = json.loads(stats.read_text())
    payload["action"]["default"]["global_q99"][0] = 30
    stats.write_text(json.dumps(payload))
    assert open_dataset(root, stats).fingerprint != original.fingerprint
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "changed instruction"}) + "\n")
    assert open_dataset(root, stats).fingerprint != original.fingerprint


def test_stepwise_and_per_embodiment_statistics_are_selected_explicitly():
    def group(low, high):
        return {"action": {"default": {"stepwise_min": [[low, low], [low + 2, low + 2]],
                                          "stepwise_max": [[high, high], [high + 2, high + 2]]}},
                "state": {"default": {"global_min": [low] * 3, "global_max": [high] * 3}}}
    stats = {"type": "per_embodiment", "embodiments": {"left": group(0, 2), "right": group(10, 12)}}
    codec = RobotFeatureCodec(SHAPE_META, stats, mode="min/max", stepwise=True)
    fields = {"action": {"default": torch.tensor([[10., 10.], [12., 12.]])},
              "state": {"default": torch.tensor([[11., 11., 11.]])}}
    encoded = codec.encode(fields, "right")
    torch.testing.assert_close(encoded["action"]["default"], -torch.ones(2, 2))
    torch.testing.assert_close(encoded["state"]["default"], torch.zeros(1, 3))
    torch.testing.assert_close(codec.decode(encoded, "right")["action"]["default"], fields["action"]["default"])
    with pytest.raises(KeyError, match="Missing feature statistics"):
        codec.encode(fields)


def test_relative_joint_actions_use_first_state_and_roundtrip():
    shape = {**SHAPE_META, "state": [{"key": "default", "raw_shape": 2, "shape": 2}]}
    processor = SampleProcessor(shape, relative_joint_keys=["default"], norm_default_mode="min/max")
    fields = {"action": {"default": torch.tensor([[5., 8.], [6., 9.]])},
              "state": {"default": torch.tensor([[2., 3.], [20., 30.]])}}
    transformed = processor.transform_fields(fields)
    torch.testing.assert_close(transformed["action"]["default"], torch.tensor([[3., 5.], [4., 6.]]))
    stats = {field: {"default": {"global_min": [0, 0], "global_max": [10, 10]}} for field in ("action", "state")}
    processor.set_normalizer_from_stats(stats)
    encoded = processor.codec.encode(transformed)
    decoded = processor.decode_actions(encoded["action"]["default"], encoded["state"]["default"])
    torch.testing.assert_close(decoded["default"], fields["action"]["default"])


def test_appearance_recipe_is_applied_by_dataset_and_fingerprinted(tmp_path):
    root, stats = make_root(tmp_path)
    options = {"p": 1, "style": {"enabled": True, "p": 1}, "fourier": {"enabled": True, "p": 1}, "background": {"enabled": False}}
    dataset = open_dataset(root, stats, video_augmentation=options)
    torch.manual_seed(42)
    result = dataset[0]
    assert result["video"].shape == (3, 2, 8, 8)
    assert result["video"].isfinite().all()
    assert dataset.fingerprint != open_dataset(root, stats).fingerprint
