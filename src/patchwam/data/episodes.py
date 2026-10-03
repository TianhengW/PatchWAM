"""Local LeRobot v2 episode windows.

The reader is independently implemented against LeRobot's documented on-disk
format. ImageWAM's data-window and filtering behavior informed its contract;
see PROVENANCE.json for the reviewed source files and their licenses.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Yuyang "Alice.L"
Source contract attribution: LICENSE-MIT and PROVENANCE.json.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .appearance import AppearanceRandomizer
from .augmentation import ClipAugment
from .cameras import compose_cameras, prepare_rgb, resize_clip
from .history import PastFrameSelector
from .processing import SampleProcessor
from .scaling import read_statistics


@dataclass
class _Episode:
    root: Path
    index: int
    path: Path
    info: dict
    rows: np.ndarray
    tasks: dict
    columns: tuple[str, ...]


def _json_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _column_name(meta, field):
    if meta.get("lerobot_key"):
        return meta["lerobot_key"]
    prefix = {"action": "action", "state": "observation.state", "images": "observation.images"}[field]
    return prefix if meta["key"] == "default" else f"{prefix}.{meta['key']}"


def _feature_tensor(table, column, indices):
    if column not in table.column_names:
        raise KeyError(f"Missing parquet column {column!r}")
    values = table[column].take(indices.tolist()).to_pylist()
    result = torch.from_numpy(np.asarray(values, dtype=np.float32))
    return result.unsqueeze(-1) if result.ndim == 1 else result


class EpisodeDataset(Dataset):
    """Endpoint images and action windows with separate padding masks.

    At 17 frames and ratio 1, images are t/t+16 and actions are t..t+15.
    True marks invalid dimensions or steps. Corrupt samples raise.
    """

    def __init__(self, dataset_dirs, shape_meta, num_frames=17, video_size=(224, 448),
                 processor=None, pretrained_norm_stats=None, val_set_proportion=0,
                 is_training_set=True, val_split_level="episode", seed=42,
                 global_sample_stride=1, sample_index_stride=1,
                 action_video_freq_ratio=1, endpoint_frames_only=True,
                 concat_multi_camera="horizontal", robotwin_camera_layout="compact_288x256",
                 nonidle_filter_path=None, episode_index_filter=None,
                 require_text_cache=False, text_embedding_cache_dir=None, context_len=128,
                 qwen_text_cache_dir=None, qwen_context_len=128, qwen_text_cache_format="qwen2_5_vl",
                 override_instruction=None, prompt_template="A video recorded from a robot's point of view executing the following instruction: {task}",
                 lerobot_backend="v2", lerobot_tolerance_s=None, video_augmentation=None,
                 separate_camera_views=False, include_vl_inputs=False, history_slots=0,
                 history_interval_s=1.0, history_jitter_s=0.4, history_whole_dropout=0.2,
                 history_slot_dropout=0.2, history_truncate=True, vl_history_jitter_frames=1,
                 vl_image_size=448, head_camera_key=None, subtask_column=None,
                 require_subtask_labels=False):
        if lerobot_backend != "v2":
            raise ValueError("EpisodeDataset currently supports local LeRobot v2; v3 merged video files require a separate backend")
        if num_frames < 2 or global_sample_stride < 1 or sample_index_stride < 1:
            raise ValueError("Frames must be >=2 and sample strides must be positive")
        if action_video_freq_ratio < 1 or (num_frames - 1) % action_video_freq_ratio:
            raise ValueError("Action horizon must divide evenly into the selected video transitions")
        if not 0 <= val_set_proportion < 1 or val_split_level not in {"episode", "root"}:
            raise ValueError("Invalid train/validation split")
        if not dataset_dirs:
            raise ValueError("At least one local dataset root is required")
        self.shape_meta = shape_meta
        self.training = bool(is_training_set)
        self.separate_camera_views = bool(separate_camera_views)
        self.include_vl_inputs = bool(include_vl_inputs or separate_camera_views or history_slots)
        self.vl_image_size = int(vl_image_size)
        if self.vl_image_size < 1:
            raise ValueError("VL image size must be positive")
        self.head_camera_key = head_camera_key or shape_meta["images"][0]["key"]
        if self.head_camera_key not in {item["key"] for item in shape_meta["images"]}:
            raise ValueError("The history head camera is missing from shape metadata")
        self.history_selector = PastFrameSelector(
            slots=history_slots, interval_s=history_interval_s, jitter_s=history_jitter_s,
            whole_dropout=history_whole_dropout, slot_dropout=history_slot_dropout,
            truncate_recent=history_truncate, vl_jitter_frames=vl_history_jitter_frames,
        )
        self.subtask_column, self.require_subtask_labels = subtask_column, bool(require_subtask_labels)
        if self.require_subtask_labels and not self.subtask_column:
            raise ValueError("Required subtask supervision needs an explicit parquet column")
        self.num_frames = int(num_frames)
        self.horizon = self.num_frames - 1
        self.frame_stride = int(global_sample_stride)
        self.sample_stride = int(sample_index_stride)
        self.image_steps = [0, self.horizon] if endpoint_frames_only else list(range(0, self.num_frames, action_video_freq_ratio))
        self.video_size, self.layout, self.robotwin_layout = tuple(video_size), concat_multi_camera, robotwin_camera_layout
        self.require_text_cache, self.text_cache = require_text_cache, text_embedding_cache_dir
        self.context_len, self.qwen_cache = int(context_len), qwen_text_cache_dir
        self.qwen_context_len, self.qwen_format = int(qwen_context_len), qwen_text_cache_format
        self.override_instruction, self.prompt_template = override_instruction, prompt_template
        self.video_tolerance = 1e-4 if lerobot_tolerance_s is None else float(lerobot_tolerance_s)
        if not np.isfinite(self.video_tolerance) or self.video_tolerance <= 0:
            raise ValueError("Video timestamp tolerance must be finite and positive")
        if isinstance(video_augmentation, dict):
            appearance_keys = {"photometric", "style", "fourier", "background"}
            augmentation_type = AppearanceRandomizer if appearance_keys & set(video_augmentation) else ClipAugment
            video_augmentation = augmentation_type(**video_augmentation)
        self.video_augmentation = video_augmentation if is_training_set else None
        roots = [Path(item).expanduser().resolve() for item in dataset_dirs]
        if val_set_proportion and val_split_level == "root":
            selected = np.arange(len(roots))
            np.random.default_rng(seed).shuffle(selected)
            boundary = int(len(selected) * (1 - val_set_proportion))
            selected = selected[:boundary] if is_training_set else selected[boundary:]
            roots = [roots[item] for item in sorted(selected)]
        self.episodes = []
        fps = set()
        import pyarrow.parquet as pq

        for root in roots:
            info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
            if str(info.get("codebase_version", "v2")).startswith("v3"):
                raise ValueError(f"LeRobot v3 root cannot be opened as v2: {root}")
            root_fps = float(info["fps"])
            if not np.isfinite(root_fps) or root_fps <= 0:
                raise ValueError(f"Dataset FPS must be finite and positive: {root}")
            fps.add(root_fps)
            chunk_size = int(info.get("chunks_size", 1000))
            episode_meta = _json_lines(root / "meta/episodes.jsonl")
            indices = sorted(int(item["episode_index"]) for item in episode_meta)
            if not indices:
                indices = list(range(int(info["total_episodes"])))
            cfg = episode_index_filter or {}
            if cfg.get("mode", "none") in {"periodic_prefix", "periodic_first", "first_k_per_period"}:
                period, first, offset = int(cfg["period"]), int(cfg["keep_first"]), int(cfg.get("offset", 0))
                if period <= 0 or not 0 <= first <= period:
                    raise ValueError("Invalid periodic episode filter")
                indices = [index for index in indices if (index - offset) % period < first]
            elif cfg.get("mode", "none") not in {"none", "all", ""}:
                raise ValueError(f"Unknown episode filter: {cfg.get('mode')}")
            if val_set_proportion and val_split_level == "episode":
                np.random.default_rng(seed).shuffle(indices)
                boundary = int(len(indices) * (1 - val_set_proportion))
                indices = indices[:boundary] if is_training_set else indices[boundary:]
            tasks = {int(item["task_index"]): item["task"] for item in _json_lines(root / "meta/tasks.jsonl")}
            filter_path = nonidle_filter_path
            if isinstance(filter_path, dict):
                if str(root) not in filter_path:
                    raise KeyError(f"No nonidle filter configured for root {root}")
                filter_path = filter_path[str(root)]
            ranges = {}
            if filter_path is not None:
                payload = json.loads(Path(filter_path).expanduser().read_text(encoding="utf-8"))
                ranges = payload.get("episodes", payload)
            for index in indices:
                template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
                path = root / template.format(episode_chunk=index // chunk_size, episode_index=index)
                parquet = pq.ParquetFile(path)
                if self.require_subtask_labels and self.subtask_column not in parquet.schema_arrow.names:
                    raise KeyError(f"Missing required subtask labels {self.subtask_column!r}: {path}")
                length = parquet.metadata.num_rows
                if "timestamp" in parquet.schema_arrow.names:
                    times = np.asarray(parquet.read(columns=["timestamp"])["timestamp"].to_pylist(), dtype=np.float64)
                    if not np.isfinite(times).all() or (len(times) > 1 and np.any(np.abs(np.diff(times) - 1 / root_fps) > self.video_tolerance)):
                        raise ValueError(f"Episode timestamps do not match dataset FPS: {path}")
                if str(index) in ranges:
                    chosen = set()
                    for start, end in ranges[str(index)]:
                        chosen.update(range(max(0, int(start)), min(length, int(end))))
                    rows = np.array(sorted(chosen), dtype=np.int64)
                else:
                    rows = np.arange(length, dtype=np.int64)
                if len(rows):
                    self.episodes.append(_Episode(root, index, path, info, rows, tasks, tuple(parquet.schema_arrow.names)))
        if len(fps) != 1:
            raise ValueError(f"Dataset roots must share FPS, received {sorted(fps)}")
        if not self.episodes:
            raise ValueError("The selected split and filters contain no episode frames")
        self.ends = np.cumsum([len(item.rows) for item in self.episodes]).tolist()
        self.processor = processor or SampleProcessor(shape_meta)
        self.processor.train() if is_training_set else self.processor.eval()
        if pretrained_norm_stats:
            statistics = read_statistics(pretrained_norm_stats)
        elif is_training_set:
            statistics = self.compute_statistics()
        else:
            raise ValueError("Validation requires training normalization statistics")
        self.statistics = statistics
        self.processor.set_normalizer_from_stats(statistics)
        self.fingerprint = self._fingerprint()

    def _fingerprint(self):
        """Hash metadata, statistics, selection, and media/cache file metadata.

        Media/cache files use path, size, and mtime; full payloads and paths
        embedded in image columns are excluded.
        """
        def canonical(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, Mapping):
                return {str(key): canonical(item) for key, item in value.items()}
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [canonical(item) for item in value]
            return value

        roots = {}
        for episode in self.episodes:
            if str(episode.root) not in roots:
                roots[str(episode.root)] = {name: (episode.root / "meta" / name).read_text(encoding="utf-8") if (episode.root / "meta" / name).exists() else None for name in ("info.json", "episodes.jsonl", "tasks.jsonl")}
        processor_config = {name: getattr(self.processor, name) for name in ("action_output_dim", "proprio_output_dim", "norm_default_mode", "norm_exception_mode", "stepwise", "relative_joint_keys", "delta_action_dim_mask", "training")}

        def file_metadata(path):
            path = Path(path).expanduser().resolve()
            stat = path.stat()
            return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

        videos = [file_metadata(self._video_path(episode, column)) for episode in self.episodes for meta in self.shape_meta["images"] if (column := _column_name(meta, "images")) not in episode.columns]
        cache_directories = []
        for cache in (self.text_cache if self.require_text_cache else None, self.qwen_cache):
            if cache is not None:
                directory = Path(cache).expanduser().resolve()
                if not directory.is_dir():
                    raise ValueError(f"Text cache must be an existing directory: {directory}")
                cache_directories.append({"directory": str(directory), "files": [file_metadata(path) for path in sorted(directory.rglob("*.pt"))]})
        payload = {
            "scope": "metadata_statistics_selection_and_media_cache_stat_v2", "roots": roots,
            "episodes": [{"root": str(item.root), "index": item.index, "rows": item.rows.tolist(), "parquet_size": item.path.stat().st_size, "parquet_mtime_ns": item.path.stat().st_mtime_ns} for item in self.episodes],
            "videos": videos, "cache_directories": cache_directories,
            "shape_meta": self.shape_meta, "statistics": self.statistics, "processor": processor_config,
            "frames": self.num_frames, "frame_stride": self.frame_stride, "sample_stride": self.sample_stride, "image_steps": self.image_steps,
            "video_size": self.video_size, "layout": self.layout, "robotwin_layout": self.robotwin_layout,
            "prompt_template": self.prompt_template, "override_instruction": self.override_instruction,
            "text_cache": self.text_cache, "require_text_cache": self.require_text_cache, "context_len": self.context_len,
            "qwen_cache": self.qwen_cache, "qwen_context_len": self.qwen_context_len, "qwen_format": self.qwen_format,
            "video_tolerance": self.video_tolerance, "augmentation": vars(self.video_augmentation) if self.video_augmentation else None,
            "history": vars(self.history_selector), "separate_camera_views": self.separate_camera_views,
            "include_vl_inputs": self.include_vl_inputs, "vl_image_size": self.vl_image_size,
            "head_camera_key": self.head_camera_key, "subtask_column": self.subtask_column,
            "require_subtask_labels": self.require_subtask_labels,
        }
        return sha256(json.dumps(canonical(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def __len__(self):
        return (self.ends[-1] + self.sample_stride - 1) // self.sample_stride

    @staticmethod
    def _table(path):
        path = Path(path)
        stat = path.stat()
        return EpisodeDataset._read_table(str(path), stat.st_size, stat.st_mtime_ns)

    @staticmethod
    @lru_cache(maxsize=4)
    def _read_table(path, size, mtime_ns):
        """Size and modification time invalidate cached reads after replacement."""
        import pyarrow.parquet as pq
        return pq.read_table(path)

    @staticmethod
    def _embedded_frame(value, root):
        if isinstance(value, dict):
            source = BytesIO(value["bytes"]) if value.get("bytes") is not None else root / value["path"]
            with Image.open(source) as image:
                return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)
        tensor = torch.as_tensor(np.asarray(value))
        return tensor.permute(2, 0, 1) if tensor.shape[-1] == 3 else tensor

    def _video_frames(self, episode, column, rows, table):
        if column in table.column_names:
            values = table[column].take(rows.tolist()).to_pylist()
            return torch.stack([self._embedded_frame(value, episode.root) for value in values])
        path = self._video_path(episode, column)
        if "timestamp" in table.column_names:
            times = np.asarray(table["timestamp"].take(rows.tolist()).to_pylist(), dtype=np.float64)
        else:
            times = rows / float(episode.info["fps"])
        import av

        targets = sorted(set(times.tolist()))
        frames = {}
        tolerance = self.video_tolerance
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(max(0, int((targets[0] - tolerance) * av.time_base)), backward=True)
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                current = float(frame.time)
                for target in targets:
                    distance = abs(current - target)
                    if distance <= tolerance and (target not in frames or distance < frames[target][0]):
                        frames[target] = (distance, torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
                if current > targets[-1] + tolerance:
                    break
        if any(target not in frames for target in targets):
            raise RuntimeError(f"Could not decode requested timestamps from {path}")
        return torch.stack([frames[float(target)][1] for target in times])

    @staticmethod
    def _video_path(episode, column):
        template = episode.info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
        return episode.root / template.format(episode_chunk=episode.index // int(episode.info.get("chunks_size", 1000)), episode_index=episode.index, video_key=column)

    def _fields(self, episode, table, positions):
        result = {}
        for field, steps in (("action", self.horizon), ("state", self.num_frames)):
            raw = episode.rows[np.minimum(positions[:steps], len(episode.rows) - 1)]
            result[field] = {}
            for meta in self.shape_meta[field]:
                value = _feature_tensor(table, _column_name(meta, field), raw)
                if value.shape[-1] != meta.get("raw_shape", meta["shape"]):
                    raise ValueError(f"Raw {field} width differs from shape metadata for {meta['key']}")
                result[field][meta["key"]] = value
        return result

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        rank = index * self.sample_stride
        episode_position = bisect_right(self.ends, rank)
        episode = self.episodes[episode_position]
        start = rank - (self.ends[episode_position - 1] if episode_position else 0)
        positions = start + np.arange(self.num_frames) * self.frame_stride
        padded = torch.from_numpy(positions >= len(episode.rows))
        table = self._table(str(episode.path))
        fields = self._fields(episode, table, positions)
        image_rows = episode.rows[np.minimum(positions[self.image_steps], len(episode.rows) - 1)]
        task_index = int(table["task_index"][int(episode.rows[start])].as_py()) if "task_index" in table.column_names else 0
        task = episode.tasks.get(task_index)
        if task is None and "task" in table.column_names:
            task = table["task"][int(episode.rows[start])].as_py()
        if task is None and self.override_instruction is None:
            raise KeyError(f"Missing task instruction in {episode.root}")
        sample = {**fields, "task": self.override_instruction or task,
                  "images": {meta["key"]: self._video_frames(episode, _column_name(meta, "images"), image_rows, table) for meta in self.shape_meta["images"]},
                  "action_is_pad": padded[:-1], "state_is_pad": padded,
                  "image_is_pad": padded[self.image_steps]}
        for key in ("action_dim_is_pad", "state_dim_is_pad", "embodiment"):
            if key in table.column_names:
                sample[key] = table[key][int(episode.rows[start])].as_py()
        if "embodiment" in sample:
            sample["embodiment"] = str(sample["embodiment"])
        result = self.processor.preprocess(sample)
        cameras = result.pop("images")
        history_clips = []
        if self.include_vl_inputs:
            head = next(item for item in self.shape_meta["images"] if item["key"] == self.head_camera_key)
            clean_current = prepare_rgb(sample["images"][self.head_camera_key][:1])
            result["vl_current"] = resize_clip(clean_current, (self.vl_image_size,) * 2)[0] * 2 - 1
            current_row = int(episode.rows[start])
            fps = float(episode.info["fps"])
            for vision_language in (False, True):
                rows, valid = self.history_selector.select(
                    current_row, fps, training=self.training, vision_language=vision_language,
                )
                if len(rows):
                    frames = prepare_rgb(self._video_frames(episode, _column_name(head, "images"), rows, table))
                else:
                    frames = clean_current.new_empty((0, *clean_current.shape[1:]))
                if vision_language:
                    if len(rows):
                        frames = resize_clip(frames, (self.vl_image_size,) * 2)
                    else:
                        frames = clean_current.new_empty((0, 3, self.vl_image_size, self.vl_image_size))
                    result["vl_history"] = (frames * 2 - 1).masked_fill(~valid[:, None, None, None], 0)
                    result["vl_history_valid"] = valid
                else:
                    if len(rows):
                        frames = resize_clip(frames, head["shape"][-2:])
                    result["history_valid"] = valid
                    history_clips = [frame.unsqueeze(0) for frame in frames]
        if self.video_augmentation is not None:
            augmented = self.video_augmentation(cameras + history_clips)
            cameras, history_clips = augmented[:len(cameras)], augmented[len(cameras):]
        if self.separate_camera_views:
            if len({tuple(camera.shape) for camera in cameras}) != 1:
                raise ValueError("Separate camera views require equal temporal and image shapes")
            result["camera_video"] = torch.stack([(camera * 2 - 1).permute(1, 0, 2, 3) for camera in cameras])
        if self.include_vl_inputs:
            if history_clips:
                history = torch.cat(history_clips, 0) * 2 - 1
            else:
                history = cameras[0].new_empty((0, 3, *cameras[0].shape[-2:]))
            result["history_video"] = history.masked_fill(~result["history_valid"][:, None, None, None], 0)
        if self.subtask_column is not None:
            label = table[self.subtask_column][int(episode.rows[start])].as_py() if self.subtask_column in table.column_names else None
            if self.require_subtask_labels and (not isinstance(label, str) or not label.strip()):
                raise ValueError(f"Invalid subtask label at row {episode.rows[start]}: {episode.path}")
            if label is not None:
                if not isinstance(label, str):
                    raise ValueError("Subtask labels must be sentence strings")
                result["subtask"] = label
        result["video"] = compose_cameras(cameras, self.layout, self.video_size, self.robotwin_layout)
        result["proprio"] = result["proprio"][:-1]
        result["proprio_is_pad"] = result["proprio_is_pad"][:-1]
        prompt = self.prompt_template.format(task=result["instruction"])
        result["prompt"] = result["instruction"] = prompt
        if self.require_text_cache:
            path = Path(self.text_cache).expanduser() / f"{sha256(prompt.encode()).hexdigest()}.t5_len{self.context_len}.wan22ti2v5b.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            context, mask = payload["context"].clone(), payload["mask"].bool()
            if context.ndim != 2 or context.shape[0] != self.context_len or mask.shape != (self.context_len,):
                raise ValueError(f"Unexpected text cache shape: {path}")
            context[~mask] = 0
            result.update(context=context, context_mask=torch.ones_like(mask), text_cache_format="t5")
        if self.qwen_cache:
            if self.qwen_format not in {"qwen2_5_vl", "qwen3_flux2"}:
                raise ValueError(f"Unsupported text cache format {self.qwen_format}")
            path = Path(self.qwen_cache).expanduser() / f"{sha256(prompt.encode()).hexdigest()}.{self.qwen_format}_len{self.qwen_context_len}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            text, mask = payload["text_hidden_states"], payload["text_attention_mask"].bool()
            if text.ndim != 2 or text.shape[0] != self.qwen_context_len or mask.shape != (self.qwen_context_len,):
                raise ValueError(f"Unexpected text cache shape: {path}")
            if self.qwen_format == "qwen3_flux2":
                result.update(text_tokens=text, text_valid=mask)
            else:
                result.update(text_hidden_states=text, text_attention_mask=mask)
            result["text_cache_format"] = self.qwen_format
        if "embodiment" in sample:
            result["embodiment"] = sample["embodiment"]
        return result

    def compute_statistics(self):
        """Preserve equal-episode weighting and the source episode-quantile envelope."""
        grouped = {}
        for episode in self.episodes:
            table = self._table(str(episode.path))
            embodiment = str(table["embodiment"][int(episode.rows[0])].as_py()) if "embodiment" in table.column_names else "default"
            group = grouped.setdefault(embodiment, {field: {meta["key"]: [] for meta in self.shape_meta[field]} for field in ("action", "state")})
            fields = {field: {meta["key"]: _feature_tensor(table, _column_name(meta, field), episode.rows) for meta in self.shape_meta[field]} for field in ("action", "state")}
            actions = {}
            for key, values in fields["action"].items():
                offsets = np.arange(len(values))[:, None] + np.arange(self.horizon)[None, :] * self.frame_stride
                actions[key] = values[np.minimum(offsets, len(values) - 1)]
            fields = self.processor.transform_fields({"action": actions, "state": {key: value.unsqueeze(1) for key, value in fields["state"].items()}})
            for field in group:
                for key, value in fields[field].items():
                    mask_name = "action_dim_is_pad" if field == "action" else "state_dim_is_pad"
                    if mask_name in table.column_names and len(self.shape_meta[field]) == 1:
                        mask = torch.as_tensor(table[mask_name][int(episode.rows[0])].as_py(), dtype=torch.bool)
                        value = value.masked_fill(mask, 0)
                    stats = {}
                    stats["stepwise_mean"] = value.mean(0)
                    stats["stepwise_var"] = value.var(0, correction=1) if value.shape[0] > 1 else torch.zeros_like(value.mean(0))
                    stats["stepwise_min"] = value.amin(0)
                    stats["stepwise_max"] = value.amax(0)
                    stats["stepwise_q01"] = torch.quantile(value, 0.01, dim=0)
                    stats["stepwise_q99"] = torch.quantile(value, 0.99, dim=0)
                    flattened = value.flatten(0, 1)
                    stats["global_mean"] = flattened.mean(0)
                    stats["global_var"] = flattened.var(0, correction=1) if flattened.shape[0] > 1 else torch.zeros_like(flattened.mean(0))
                    stats["global_q01"] = torch.quantile(flattened, 0.01, dim=0)
                    stats["global_q99"] = torch.quantile(flattened, 0.99, dim=0)
                    group[field][key].append(stats)
        merged = {}
        per_embodiment = set(grouped) != {"default"}
        for embodiment, group in grouped.items():
            result = {"action": {}, "state": {}}
            for field, features in group.items():
                for key, episodes in features.items():
                    stats = {}
                    means = torch.stack([item["stepwise_mean"] for item in episodes])
                    variances = torch.stack([item["stepwise_var"] for item in episodes])
                    mean = means.mean(0)
                    global_mean = means.mean((0, 1))
                    stats["stepwise_mean"] = mean
                    stats["stepwise_std"] = (variances + (means - mean) ** 2).mean(0).sqrt()
                    stats["global_mean"] = global_mean
                    stats["global_std"] = (variances + (means - global_mean) ** 2).mean((0, 1)).sqrt()
                    for name, reduction in (("min", "amin"), ("max", "amax"), ("q01", "amin"), ("q99", "amax")):
                        stepwise = getattr(torch.stack([item["stepwise_" + name] for item in episodes]), reduction)(0)
                        stats["stepwise_" + name] = stepwise
                        stats["global_" + name] = getattr(stepwise, reduction)(0)
                    if per_embodiment:
                        global_means = torch.stack([item["global_mean"] for item in episodes])
                        global_variances = torch.stack([item["global_var"] for item in episodes])
                        global_mean = global_means.mean(0)
                        stats["global_mean"] = global_mean
                        stats["global_std"] = (global_variances + (global_means - global_mean) ** 2).mean(0).sqrt()
                        for name, reduction in (("q01", "amin"), ("q99", "amax")):
                            stats["global_" + name] = getattr(torch.stack([item["global_" + name] for item in episodes]), reduction)(0)
                    result[field][key] = stats
            merged[embodiment] = result
        metadata = {"num_episodes": len(self.episodes), "num_transition": self.ends[-1]}
        return {**metadata, **merged["default"]} if set(merged) == {"default"} else {**metadata, "type": "per_embodiment", "embodiments": merged}
