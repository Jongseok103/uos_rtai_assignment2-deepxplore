from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class NeuronCoverage:
    """
    Simple neuron coverage tracker for Conv2d / Linear layers.

    Coverage rule:
    - For Conv2d outputs: spatial mean per channel
    - For Linear outputs: raw activation per neuron
    - A neuron is covered if activation > threshold at least once
    """

    def __init__(self, model: nn.Module, threshold: float = 0.0):
        self.model = model
        self.threshold = threshold

        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_names: List[str] = []

        self.covered: Dict[str, torch.Tensor] = {}
        self.total_neurons: int = 0

        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.layer_names.append(name)
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor):
            if not isinstance(output, torch.Tensor):
                return

            with torch.no_grad():
                if output.dim() == 4:
                    # [B, C, H, W] -> [B, C]
                    act = output.mean(dim=(2, 3))
                elif output.dim() == 2:
                    # [B, D]
                    act = output
                else:
                    return

                if layer_name not in self.covered:
                    num_neurons = act.shape[1]
                    self.covered[layer_name] = torch.zeros(num_neurons, dtype=torch.bool, device="cpu")
                    self.total_neurons += num_neurons

                active = (act > self.threshold).any(dim=0).detach().cpu()
                self.covered[layer_name] |= active

        return hook

    def reset(self) -> None:
        for layer_name in self.covered:
            self.covered[layer_name].zero_()

    def coverage_ratio(self) -> float:
        if self.total_neurons == 0:
            return 0.0
        covered_count = sum(mask.sum().item() for mask in self.covered.values())
        return covered_count / self.total_neurons

    def covered_count(self) -> int:
        return sum(mask.sum().item() for mask in self.covered.values())

    def summary(self) -> Dict[str, float]:
        return {
            "covered_neurons": self.covered_count(),
            "total_neurons": self.total_neurons,
            "coverage_ratio": self.coverage_ratio(),
        }

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()