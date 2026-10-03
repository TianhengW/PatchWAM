# SPDX-License-Identifier: Apache-2.0
"""Successful-update weight averaging and stateless teacher evaluation."""

import math
from contextlib import contextmanager

import torch


class PolicyWeightAverage:
    """FP32 shadows of trainable weights, including optional encoder adapters.

    ``warmup_updates`` copies the current weights during the first N successful
    updates. Subsequent updates use the fixed decay. Frozen parameters and
    buffers are read from the live model rather than replicated or averaged.
    """

    def __init__(self, model, *, decay=0.999, warmup_updates=0):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite and in [0,1)")
        if type(warmup_updates) is not int or warmup_updates < 0:
            raise ValueError("EMA warmup updates must be a nonnegative integer")
        self.decay, self.warmup_updates, self.updates = decay, warmup_updates, 0
        parameters = self._parameters(model)
        if not parameters:
            raise ValueError("EMA requires trainable parameters")
        self.shadows = {
            name: tensor.detach().float().clone() for name, tensor in parameters.items()
        }
        self._validate_values(self.shadows)

    @staticmethod
    def _parameters(model):
        parameters = {
            name: tensor for name, tensor in model.named_parameters() if tensor.requires_grad
        }
        if any(not tensor.is_floating_point() for tensor in parameters.values()):
            raise ValueError("EMA parameters must be floating-point tensors")
        return parameters

    @staticmethod
    def _validate_values(parameters):
        finite = [torch.isfinite(tensor.detach().float()).all() for tensor in parameters.values()]
        if finite and not torch.stack(finite).all():
            raise ValueError("EMA parameters must be finite in FP32")

    def _matching_parameters(self, model):
        parameters = self._parameters(model)
        if parameters.keys() != self.shadows.keys() or any(
            parameters[name].shape != shadow.shape for name, shadow in self.shadows.items()
        ):
            raise ValueError("EMA trainable parameter names or shapes changed")
        return parameters

    @torch.no_grad()
    def update(self, model, *, check_finite=True):
        parameters = self._matching_parameters(model)
        if check_finite:
            self._validate_values(parameters)
        decay = 0.0 if self.updates < self.warmup_updates else self.decay
        for name, shadow in self.shadows.items():
            shadow.mul_(decay).add_(parameters[name].detach().to(shadow), alpha=1 - decay)
        self.updates += 1

    def state_dict(self):
        return {
            "format_version": 1,
            "updates": self.updates,
            "decay": self.decay,
            "warmup_updates": self.warmup_updates,
            "weights": {
                name: tensor.detach().cpu().clone() for name, tensor in self.shadows.items()
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        if state.get("format_version") != 1:
            raise ValueError("Unsupported EMA state format")
        if state.get("decay") != self.decay or state.get("warmup_updates") != self.warmup_updates:
            raise ValueError("EMA continuation requires unchanged decay and warmup")
        updates, weights = state.get("updates"), state.get("weights", {})
        if type(updates) is not int or updates < 0 or weights.keys() != self.shadows.keys():
            raise ValueError("Invalid EMA update count or parameter names")
        for name, shadow in self.shadows.items():
            value = weights[name]
            if (
                not torch.is_tensor(value)
                or value.shape != shadow.shape
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid EMA weight: " + name)
        for name, shadow in self.shadows.items():
            shadow.copy_(weights[name])
        self.updates = updates

    def parameters_for(self, model):
        """Return detached shadows on each live parameter's device and dtype."""
        parameters = self._matching_parameters(model)
        return {name: shadow.detach().to(parameters[name]) for name, shadow in self.shadows.items()}

    @torch.no_grad()
    def call(self, model, *args, parameter_prefixes=None, submodule=None, **kwargs):
        """Evaluate a teacher without replacing live student parameters.

        This is safe between a student forward and its backward. Buffer state
        is cloned so even an encoder with mutable buffers cannot alter the live
        model. Module training flags are restored, including mixed frozen modes.
        """
        parameters = self._matching_parameters(model)
        selected = self.shadows
        if parameter_prefixes is not None:
            if (
                not isinstance(parameter_prefixes, (list, tuple))
                or not parameter_prefixes
                or any(not isinstance(prefix, str) or not prefix for prefix in parameter_prefixes)
            ):
                raise ValueError("EMA teacher parameter prefixes must be nonempty strings")
            selected = {
                name: tensor
                for name, tensor in selected.items()
                if name.startswith(tuple(parameter_prefixes))
            }
            if not selected:
                raise ValueError("EMA teacher prefixes match no trainable parameters")
        target, prefix = model, ""
        if submodule is not None:
            if not isinstance(submodule, str) or not submodule:
                raise ValueError("EMA target submodule must be a nonempty module path")
            target, prefix = model.get_submodule(submodule), submodule + "."
            selected = {
                name: tensor for name, tensor in selected.items() if name.startswith(prefix)
            }
            if not selected:
                raise ValueError("EMA target submodule has no selected trainable parameters")
        replacement = {
            name.removeprefix(prefix): tensor.detach().to(parameters[name])
            for name, tensor in selected.items()
        }
        replacement.update({name: value.detach().clone() for name, value in target.named_buffers()})
        modes = [(module, module.training) for module in target.modules()]
        try:
            target.eval()
            return torch.func.functional_call(target, replacement, args, kwargs, strict=False)
        finally:
            for module, training in modes:
                module.training = training

    @contextmanager
    def apply_to(self, model):
        """Temporarily use averaged weights for evaluation outside live graphs.

        Use ``call`` for a teacher during training: swapping parameters in this
        context would invalidate an outstanding student autograd graph.
        """
        parameters = self._matching_parameters(model)
        original = {name: tensor.detach().clone() for name, tensor in parameters.items()}
        modes = [(module, module.training) for module in model.modules()]
        with torch.no_grad():
            try:
                for name, tensor in parameters.items():
                    tensor.copy_(self.shadows[name])
                model.eval()
                yield model
            finally:
                for name, tensor in parameters.items():
                    tensor.copy_(original[name])
                for module, training in modes:
                    module.training = training

    def native_weights(self, model):
        """Complete strict-loader weights, retaining unaveraged policy buffers."""
        from .checkpoints import checkpoint_parameter_keys, policy_weight_state

        parameters = self._matching_parameters(model)
        state = {
            name: tensor.detach().cpu().clone().contiguous()
            for name, tensor in policy_weight_state(model).items()
        }
        canonical = {id(tensor): name for name, tensor in parameters.items()}
        native_names = checkpoint_parameter_keys(model)
        for name, tensor in model.named_parameters(remove_duplicate=False):
            if id(tensor) in canonical and name in native_names:
                state[native_names[name]] = (
                    self.shadows[canonical[id(tensor)]].detach().cpu().clone().contiguous()
                )
        return state
