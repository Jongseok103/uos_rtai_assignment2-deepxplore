from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class NeuronCoverageTracker:
    """
    Conv2d / Linear layer 기준으로 covered 뉴런을 추적하고,
    각 forward 이후 아직 안 켜진 뉴런의 activation을 objective에 넣을 수 있게 해주는 클래스.
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
        """coverage 볼 layer들에 forward hook 거는 부분."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # coverage 계산이랑 activation 추적 둘 다 하려고 hook을 걸어둠.
                self.layer_order.append(name)
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor):
                return

            if output.dim() == 4:
                # Conv layer는 위치가 많아서 channel 평균으로 정리해서 봄.
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
        """직전 forward에서 저장한 activation 캐시 비우는 함수."""
        self.last_activations = {}

    def reset_coverage(self) -> None:
        """뉴런 coverage 누적 상태 초기화하는 함수."""
        for mask in self.covered.values():
            mask.zero_()

    def coverage_ratio(self) -> float:
        """현재까지 활성화된 뉴런 비율 계산하는 부분."""
        if self.total_neurons == 0:
            return 0.0
        covered_count = sum(mask.sum().item() for mask in self.covered.values())
        return covered_count / self.total_neurons

    def pick_uncovered_neuron(self) -> Optional[Tuple[str, int]]:
        """아직 안 켜진 뉴런 하나 골라오는 함수."""
        for layer_name in self.layer_order:
            mask = self.covered.get(layer_name)
            if mask is None:
                continue
            uncovered_indices = (~mask).nonzero(as_tuple=False)
            if uncovered_indices.numel() == 0:
                continue
            # 아직 안 켜진 뉴런 하나 골라서 objective에 넣는 방식임.
            return layer_name, int(uncovered_indices[0].item())
        return None

    def activation_term(self, selection: Optional[Tuple[str, int]]) -> torch.Tensor:
        """고른 뉴런 activation을 objective에 넣기 좋게 scalar로 바꿔줌."""
        if selection is None:
            device = next(self.model.parameters()).device
            return torch.zeros((), device=device)

        layer_name, neuron_index = selection
        activation = self.last_activations.get(layer_name)
        if activation is None:
            device = next(self.model.parameters()).device
            return torch.zeros((), device=device)
        # 선택된 뉴런 activation을 키우면 coverage 늘리는 데 도움 됨.
        return activation[..., neuron_index].mean()

    def remove(self) -> None:
        """등록했던 hook 정리하는 함수."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
