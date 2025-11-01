from abc import ABC, abstractmethod
from typing import Dict
import torch
from torch import nn

Tensor = torch.Tensor


class RepresentationExtractor(ABC):
    """
    Common interface for per-(input, output) representation learning.

    Call flow:
      - fit(input_grid, output_grid): trains internal model on this single pair
      - embed(): returns a 1D tensor embedding (e.g., flattened weights)
      - predict(input_grid): optional grid prediction after training
    """

    @abstractmethod
    def fit(self, input_grid: Tensor, output_grid: Tensor) -> "RepresentationExtractor":
        ...

    @abstractmethod
    def embed(self) -> Tensor:
        ...

    @abstractmethod
    def predict(self, input_grid: Tensor) -> Tensor:
        ...

    # --- Default: flatten (state_dict) parameters as embedding ---
    def _parameters_for_embedding(self) -> Dict[str, Tensor]:
        if hasattr(self, "model") and isinstance(getattr(self, "model"), nn.Module):
            return {k: v for k, v in self.model.state_dict().items()}
        return {}

    def _flatten_params(self) -> Tensor:
        parts = []
        for p in self._parameters_for_embedding().values():
            parts.append(p.detach().view(-1))
        return torch.cat(parts) if parts else torch.empty(0)
