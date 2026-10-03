"""Causal data history and trainable image-grounded instruction contracts."""

import numpy as np
import pytest
import torch
from torch import nn

from patchwam.data.history import CausalObservationBuffer, PastFrameSelector
from patchwam.models import FluxAssetPolicy, make_tiny_policy
from patchwam.models.vision_language import (
    LowRankResidualLinear,
    VisualInstructionEncoder,
    pool_visual_grid,
)


def test_episode_clock_history_and_independent_dropout():
    selector = PastFrameSelector(slots=3, whole_dropout=1, jitter_s=0, vl_jitter_frames=0)
    rows, valid = selector.select(25, 20)
    assert rows.tolist() == [5, 0, 0]
    assert valid.tolist() == [True, False, False]
    _, dropped = selector.select(60, 20, training=True)
    _, vl_valid = selector.select(60, 20, training=True, vision_language=True)
    assert not dropped.any() and vl_valid.all()
    np.random.seed(123)
    jittered, valid = PastFrameSelector(slots=20).select(450, 20, training=True)
    assert (jittered[valid.numpy()] < 450).all()
    assert (np.abs(jittered - (450 - np.arange(1, 21) * 20)) <= 8).all()


def test_online_history_resets_and_rejects_clock_rewind():
    buffer = CausalObservationBuffer(slots=3, tolerance_s=0)
    for timestamp in range(4):
        buffer.append(torch.full((3, 4, 4), float(timestamp)), timestamp, episode_id="first")
    result = buffer.history(3)
    assert result["history_valid"].all()
    assert result["history_video"][:, 0, 0, 0].tolist() == [2, 1, 0]
    buffer.append(torch.ones(3, 4, 4), 0, episode_id="second")
    assert not buffer.history(0)["history_valid"].any()
    with pytest.raises(ValueError, match="increase"):
        buffer.append(torch.ones(3, 4, 4), 0, episode_id="second")
    buffer.reset()
    with pytest.raises(ValueError, match="Append"):
        buffer.history(0)


def test_low_rank_branch_preserves_base_and_only_trains_adapters():
    layer = nn.Linear(5, 7)
    value = torch.randn(2, 5)
    original = layer(value).detach()
    adapter = LowRankResidualLinear(layer, rank=2, alpha=4, dropout=0)
    torch.testing.assert_close(adapter(value), original)
    adapter(value).square().mean().backward()
    assert all(parameter.grad is None for parameter in adapter.base.parameters())
    assert adapter.residual_out.weight.grad.abs().sum() > 0


class _ImageAsset(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def encode(self, images):
        return images.mean(1, keepdim=True).expand(-1, 128, -1, -1) * self.scale


class _ImageLanguage(nn.Module):
    requires_images = True
    trainable_adapters = True
    auxiliary_weight = 0.1

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.seen = None

    def forward(self, prompts, *, current, history, history_valid, subtasks):
        self.seen = (current.clone(), None if subtasks is None else list(subtasks))
        b = len(prompts)
        features = current.mean((1, 2, 3))[:, None, None].expand(b, 2, 32) * self.weight
        return {"text_tokens": features, "text_valid": torch.ones(b, 2, dtype=torch.bool),
                "auxiliary_loss": self.weight.square()}


def test_asset_wrapper_separate_views_history_and_vl_gradients():
    encoder = _ImageLanguage()
    wrapper = FluxAssetPolicy(make_tiny_policy(), _ImageAsset(), encoder, history_pool_size=2)
    wrapper.train()
    batch = {
        "camera_video": torch.randn(1, 3, 3, 2, 4, 4),
        "history_video": torch.randn(1, 2, 3, 4, 4), "history_valid": torch.tensor([[True, False]]),
        "vl_current": torch.full((1, 3, 4, 4), 0.25), "vl_history": torch.zeros(1, 2, 3, 4, 4),
        "vl_history_valid": torch.tensor([[True, False]]), "prompt": ["place object"],
        "proprio": torch.zeros(1, 16, 14), "action": torch.zeros(1, 16, 14), "subtask": ["pick object"],
    }
    encoded = wrapper._encode(batch, target=True)
    assert encoded["reference_tokens"].shape == (1, 48 + 8, 128)
    assert encoded["future_tokens"].shape == (1, 48, 128)
    assert encoded["reference_ids"][0, :, 0].unique().tolist() == [-2, -1, 10, 11, 12]
    assert not encoded["reference_valid"][0, -4:].any()
    assert encoded["reference_tokens"][0, -4:].count_nonzero() == 0
    result = wrapper(batch)
    flow_gradient = torch.autograd.grad(result["loss_video"] + result["loss_action"], encoder.weight, retain_graph=True)[0]
    assert flow_gradient.abs() > 0
    result["loss"].backward()
    assert encoder.weight.grad is not None and encoder.weight.requires_grad
    assert wrapper.autoencoder.scale.grad is None
    wrapper.eval()
    prediction = wrapper.sample_actions(batch, steps=1)
    assert prediction["future_tokens"].shape == (1, 48, 128)
    assert encoder.seen[1] is None
    # Future targets must not affect language inputs.
    changed = {**batch, "camera_video": batch["camera_video"].clone()}
    changed["camera_video"][:, :, :, -1] += 100
    wrapper._encode(changed, target=True)
    torch.testing.assert_close(encoder.seen[0], batch["vl_current"])


class _TinyProcessor:
    def __init__(self):
        tokenizers = pytest.importorskip("tokenizers")
        transformers = pytest.importorskip("transformers")
        vocabulary = {"[PAD]": 0, "[UNK]": 1, "<vision_start>": 2, "<image>": 3,
                      "<vision_end>": 4, "<assistant>": 5, "<eos>": 6,
                      "place": 7, "object": 8, "pick": 9}
        tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
        self.tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="<eos>", unk_token="[UNK]")
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<vision_start>", "<image>", "<vision_end>", "<assistant>"]})

    def apply_chat_template(self, messages, **kwargs):
        pieces = ["<vision_start> <image> <vision_end>" if item["type"] == "image" else item["text"] for item in messages[0]["content"]]
        return " ".join(pieces) + " <assistant>"

    def image_processor(self, images, **kwargs):
        # Duplicate each 2x2 patch over time.
        patches = []
        for image in images:
            rgb = torch.from_numpy(np.asarray(image).copy()).float().permute(2, 0, 1) / 255
            for y in range(0, 8, 2):
                for x in range(0, 8, 2):
                    patches.append(rgb[:, y:y + 2, x:x + 2].unsqueeze(1).expand(3, 2, 2, 2).flatten())
        return {"pixel_values": torch.stack(patches), "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images))}


def _mini_visual_encoder(trainable=False):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        pytest.skip("Qwen3-VL requires Transformers >=4.57")
    config = transformers.Qwen3VLConfig(
        text_config={"vocab_size": 40, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 4,
                     "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}, "pad_token_id": 0},
        vision_config={"depth": 3, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2,
                       "out_hidden_size": 16, "num_position_embeddings": 16, "deepstack_visual_indexes": [0, 1, 2]},
        image_token_id=3, vision_start_token_id=2, vision_end_token_id=4,
    )
    return VisualInstructionEncoder(
        encoder=transformers.Qwen3VLForConditionalGeneration(config), processor=_TinyProcessor(),
        trainable_adapters=trainable, lora_rank=2, lora_alpha=4, lora_dropout=0,
        instruction_layers=(1, 2, 3), summary_layers=(2, 3, 4), image_size=8, history_pool_size=1,
    )


def test_official_mini_vl_pooling_subtask_causality_and_gradients():
    torch.manual_seed(12)
    encoder = _mini_visual_encoder(trainable=True)
    encoder.train()
    current = torch.zeros(1, 3, 8, 8)
    history = torch.ones(1, 2, 3, 8, 8)
    valid = torch.tensor([[True, False]])
    labeled = encoder(["place object"], current=current, history=history, history_valid=valid, subtasks=["pick object"])
    other_label = encoder(["place object"], current=current, history=history, history_valid=valid, subtasks=["place"])
    plain = encoder(["place object"], current=current, history=history, history_valid=valid)
    torch.testing.assert_close(labeled["text_tokens"], plain["text_tokens"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(labeled["text_tokens"], other_label["text_tokens"], atol=1e-5, rtol=1e-5)
    assert labeled["text_tokens"].shape == (1, 3, 48)
    labeled["auxiliary_loss"].backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for name, parameter in encoder.named_parameters() if "residual_out" in name)
    assert all(parameter.grad is None for parameter in encoder.encoder.model.visual.parameters())
    changed = history.clone()
    changed[:, 1] = float("nan")
    encoder.eval()
    unmasked = encoder(["place object"], current=current, history=changed, history_valid=valid)
    torch.testing.assert_close(unmasked["text_tokens"], plain["text_tokens"], atol=1e-5, rtol=1e-5)
    assert pool_visual_grid(torch.arange(16).float().reshape(4, 4), 2, 2, 1).shape == (1, 4)


def test_dataset_past_clock_clean_vl_images_and_required_subtask_labels(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_data_pipeline import make_root, open_dataset

    root, stats = make_root(tmp_path)
    path = root / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(path)
    pq.write_table(table.append_column("subtask", pa.array(["pick object"] * len(table))), path)
    dataset = open_dataset(
        root, stats, separate_camera_views=True, history_slots=3, history_interval_s=0.25,
        history_jitter_s=0, history_whole_dropout=1, vl_history_jitter_frames=0,
        vl_image_size=8, subtask_column="subtask", require_subtask_labels=True,
        video_augmentation=lambda views: [(view + 0.1).clamp(0, 1) for view in views],
    )
    sample = dataset[19]
    assert sample["subtask"] == "pick object"
    assert sample["camera_video"].shape == (1, 3, 2, 8, 8)
    assert not sample["history_valid"].any() and sample["vl_history_valid"].all()
    assert sample["history_video"].count_nonzero() == 0
    expected = torch.tensor([14, 9, 4]).float() * 10 / 255 * 2 - 1
    torch.testing.assert_close(sample["vl_history"][:, 0, 0, 0], expected)
    torch.testing.assert_close(sample["vl_current"], torch.full((3, 8, 8), 190 / 255 * 2 - 1))
    assert sample["camera_video"][0, :, 0].mean() > sample["vl_current"].mean()
    assert not dataset[0]["vl_history_valid"].any()
    with pytest.raises(KeyError, match="Missing required subtask"):
        open_dataset(root, stats, subtask_column="missing", require_subtask_labels=True)


def test_real_mini_vl_history_self_flow_ema_exact_resume(tmp_path):
    from torch.utils.data import Dataset

    from patchwam.checkpoints import policy_weight_state
    from patchwam.engine import OptimizationRun, RunSettings

    class Observations(Dataset):
        def __len__(self):
            return 4

        def fingerprint(self):
            return "mini-qwen-vl-three-views-causal-history"

        def __getitem__(self, index):
            return {
                "camera_video": torch.zeros(3, 3, 2, 8, 8),
                "history_video": torch.full((1, 3, 8, 8), 0.2),
                "history_valid": torch.tensor([True]),
                "vl_current": torch.zeros(3, 8, 8),
                "vl_history": torch.full((1, 3, 8, 8), 0.2),
                "vl_history_valid": torch.tensor([True]),
                "prompt": "place object", "subtask": "pick object",
                "action": torch.zeros(16, 2), "proprio": torch.zeros(16, 3),
            }

    def create(directory):
        torch.manual_seed(123)
        policy = make_tiny_policy(
            action_dim=2, proprio_dim=3, text_dim=48, depth=4,
            self_flow_variant=3, self_flow_label_warmup=1,
            self_flow_pseudo_label_fraction=1, self_flow_sampling_steps=2,
        )
        wrapper = FluxAssetPolicy(policy, _ImageAsset(), _mini_visual_encoder(trainable=True),
                                  history_pool_size=2)
        calls = {"vl": 0, "policy": 0}

        def record_vl(_module, _args, _output):
            calls["vl"] += 1

        def check_prefix(_module, args):
            batch = args[0]
            # Three 8x8 views plus 2x2 history.
            assert batch["reference_tokens"].shape[1:] == (196, 128)
            assert batch["future_tokens"].shape[1:] == (192, 128)
            calls["policy"] += 1

        wrapper.text_encoder.register_forward_hook(record_vl)
        wrapper.policy.register_forward_pre_hook(check_prefix)
        run = OptimizationRun(
            wrapper, Observations(),
            RunSettings(output_dir=str(directory), epochs=1, max_updates=2, batch_size=1,
                        accumulation=1, precision="no", cpu=True, workers=0,
                        ema_decay=0.9, checkpoint_every=1),
        )
        return run, calls

    baseline, calls = create(tmp_path / "baseline")
    baseline.train()
    assert calls["vl"] == 2 and calls["policy"] > calls["vl"]
    assert baseline.ema.updates == baseline.model.policy._optimizer_updates == 2
    # Update 2 crosses pseudo-label warmup.
    assert any(parameter.count_nonzero() for name, parameter in baseline.model.named_parameters()
               if "text_encoder" in name and "residual_out.weight" in name)

    resumed, resume_calls = create(tmp_path / "resumed")
    resumed.restore(tmp_path / "baseline" / "step_0000001")
    assert resumed.model.policy._optimizer_updates == resumed.ema.updates == 1
    resumed.train()
    assert resume_calls["vl"] == 1 and resume_calls["policy"] > resume_calls["vl"]
    for name, tensor in baseline.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], tensor, rtol=0, atol=0)
    native = policy_weight_state(baseline.model)
    assert any(name.startswith("text_encoder.") for name in native)
    for name, tensor in native.items():
        torch.testing.assert_close(policy_weight_state(resumed.model)[name], tensor, rtol=0, atol=0)
    for name, tensor in baseline.ema.shadows.items():
        torch.testing.assert_close(resumed.ema.shadows[name], tensor, rtol=0, atol=0)
