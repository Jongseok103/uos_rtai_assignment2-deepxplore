from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class NeuronCoverageTracker:
    """
    Tracks covered neurons for Conv2d / Linear layers and exposes differentiable
    activation terms for uncovered neurons after each forward pass.
    """

    def __init__(self, model: nn.Module, threshold: float = 0.0):
        self.model = model
        self.threshold = threshold
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_order: List[str] = []
        self.covered: Dict[str, torch.Tensor] = {}
        self.last_activations: Dict[str, torch.Tensor] = {}
        self.total_neurons = 0
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.layer_order.append(name)
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                return

            if output.dim() == 4:
                activation = output.mean(dim=(2, 3))
            elif output.dim() == 2:
                activation = output
            else:
                return

            if layer_name not in self.covered:
                neuron_count = activation.shape[1]
                self.covered[layer_name] = torch.zeros(neuron_count, dtype=torch.bool, device="cpu")
                self.total_neurons += neuron_count

            self.last_activations[layer_name] = activation
            active = (activation.detach() > self.threshold).any(dim=0).cpu()
            self.covered[layer_name] |= active

        return hook

    def reset_current_activations(self) -> None:
        self.last_activations = {}

    def reset_coverage(self) -> None:
        for mask in self.covered.values():
            mask.zero_()

    def coverage_ratio(self) -> float:
        if self.total_neurons == 0:
            return 0.0
        covered_count = sum(mask.sum().item() for mask in self.covered.values())
        return covered_count / self.total_neurons

    def pick_uncovered_neuron(self) -> Optional[Tuple[str, int]]:
        for layer_name in self.layer_order:
            mask = self.covered.get(layer_name)
            if mask is None:
                continue
            uncovered_indices = (~mask).nonzero(as_tuple=False)
            if uncovered_indices.numel() == 0:
                continue
            return layer_name, int(uncovered_indices[0].item())
        return None

    def activation_term(self, selection: Optional[Tuple[str, int]]) -> torch.Tensor:
        if selection is None:
            device = next(self.model.parameters()).device
            return torch.zeros((), device=device)

        layer_name, neuron_index = selection
        activation = self.last_activations.get(layer_name)
        if activation is None:
            device = next(self.model.parameters()).device
            return torch.zeros((), device=device)
        return activation[..., neuron_index].mean()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
