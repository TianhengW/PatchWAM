import pytest
import torch

from patchwam.data import AppearanceRandomizer

DR4 = {
    "p": 0.8,
    "photometric": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1,
                    "gamma": [0.7, 1.4], "exposure": [-0.5, 0.5], "color_temperature": 0.15,
                    "gaussian_noise_std": 0.03, "blur_sigma": [0, 1.5], "blur_prob": 0.3},
    "style": {"enabled": True, "p": 0.5, "target_mean": [0.2, 0.8], "target_std": [0.05, 0.4], "strength": [0.4, 1]},
    "fourier": {"enabled": True, "p": 0.5, "amp_jitter": [0.5, 1.5], "envelope_strength": [0, 0.6], "envelope_grid": 8},
    "background": {"enabled": False},
}


@pytest.mark.parametrize("seed", [0, 2, 6, 11])
def test_c2r_dr4_preserves_temporal_consistency_shape_and_range(seed):
    frame = torch.linspace(0.05, 0.95, 3 * 16 * 16).reshape(1, 3, 16, 16)
    views = [frame.repeat(2, 1, 1, 1), frame.flip(-1).repeat(2, 1, 1, 1), frame.flip(-2).repeat(2, 1, 1, 1)]
    randomizer = AppearanceRandomizer(**DR4)
    torch.manual_seed(seed)
    result = randomizer(views)
    assert len(result) == 3
    for clip in result:
        assert clip.shape == (2, 3, 16, 16)
        assert clip.isfinite().all()
        assert 0 <= clip.min() <= clip.max() <= 1
        torch.testing.assert_close(clip[0], clip[1], rtol=0, atol=0)
    torch.manual_seed(seed)
    tensor_output = randomizer(torch.stack(views))
    torch.testing.assert_close(torch.stack(result), tensor_output, rtol=0, atol=0)


def test_c2r_dr4_disabled_sample_is_identity_and_background_is_explicitly_unsupported():
    views = [torch.rand(2, 3, 16, 16)]
    assert AppearanceRandomizer(**{**DR4, "p": 0})(views) is views
    with pytest.raises(NotImplementedError, match="background replacement"):
        AppearanceRandomizer(background={"enabled": True})
    with pytest.raises(TypeError, match="Unknown appearance"):
        AppearanceRandomizer(photometric={"unknown_control": 1})


def test_individual_style_and_fourier_stages_preserve_temporal_parameters():
    no_lighting = {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0,
                   "gamma": None, "exposure": None, "color_temperature": 0,
                   "gaussian_noise_std": 0, "blur_sigma": None}
    frames = torch.linspace(0.1, 0.9, 3 * 16 * 16).reshape(1, 3, 16, 16).repeat(2, 1, 1, 1)
    for stage in ("style", "fourier"):
        torch.manual_seed(42)
        result = AppearanceRandomizer(p=1, photometric=no_lighting, **{stage: {"enabled": True, "p": 1}})(frames)
        assert result.isfinite().all()
        torch.testing.assert_close(result[0], result[1], rtol=0, atol=0)
        assert not torch.equal(result, frames)
