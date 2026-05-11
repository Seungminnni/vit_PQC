import torch
import torch.nn as nn


class LWEColumnViT(nn.Module):
    def __init__(self, M=64, n=16, in_channels=1, embed_dim=128, num_heads=4, depth=2, dropout=0.1):
        super().__init__()
        self.M = M
        self.n = n

        self.patch_embed = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=(M, 1))
        self.pos_embed = nn.Parameter(torch.randn(1, n + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.squeeze(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, : self.n, :])
        return self.classifier(x).squeeze(-1)
