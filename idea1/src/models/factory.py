from __future__ import annotations

import torch.nn as nn

from src.models.cit_lwe import CITLEWEMultiBranch
from src.models.cnn import SimpleLWEImageCNN
from src.models.column_transformer import ColumnTransformerLWE


def build_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    lwe_cfg = config["lwe"]
    name = model_cfg["name"]
    n = int(lwe_cfg["n"])
    if name == "simple_cnn":
        model = SimpleLWEImageCNN(
            in_channels=int(model_cfg["in_channels"]),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            output_n=n,
        )
        model.consumes_packet = False
        return model
    if name == "column_transformer":
        model = ColumnTransformerLWE(
            in_channels=int(model_cfg["in_channels"]),
            n=n,
            d_model=int(model_cfg.get("d_model", 128)),
            depth=int(model_cfg.get("depth", 4)),
            heads=int(model_cfg.get("heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            output_n=n,
        )
        model.consumes_packet = False
        return model
    if name == "cit_lwe":
        model = CITLEWEMultiBranch(
            rhie_channels=int(model_cfg["rhie_channels"]),
            gram_channels=model_cfg.get("gram_channels"),
            n=n,
            d_model=int(model_cfg.get("d_model", 128)),
            depth=int(model_cfg.get("depth", 4)),
            heads=int(model_cfg.get("heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        model.consumes_packet = True
        return model
    raise ValueError(f"Unsupported model name: {name}")
