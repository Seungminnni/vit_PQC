from __future__ import annotations

import torch
import torch.nn as nn


class CandidateDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        secret_type: str = "binary",
        integer_values: tuple[int, ...] = (-3, -2, -1, 1, 2, 3),
    ):
        super().__init__()
        self.secret_type = secret_type
        self.integer_values = tuple(integer_values)
        self.support_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
        self.integer_value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, len(self.integer_values) + 1))

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        support_logits = self.support_head(tokens).squeeze(-1)
        out = {"support_logits": support_logits}
        if self.secret_type == "ternary":
            out["value_logits"] = self.value_head(tokens)
        if self.secret_type == "integer":
            out["integer_logits"] = self.integer_value_head(tokens)
        return out


def support_logits_from_output(output: dict[str, torch.Tensor], secret_type: str) -> torch.Tensor:
    if secret_type == "ternary" and "value_logits" in output:
        probs = torch.softmax(output["value_logits"], dim=-1)
        return torch.logit((1.0 - probs[..., 1]).clamp(1e-5, 1 - 1e-5))
    if secret_type == "integer" and "integer_logits" in output:
        probs = torch.softmax(output["integer_logits"], dim=-1)
        return torch.logit((1.0 - probs[..., 0]).clamp(1e-5, 1 - 1e-5))
    return output["support_logits"]

