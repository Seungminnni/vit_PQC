import torch
import torch.nn as nn
import torch.nn.functional as F


def num_secret_classes(secret_type: str) -> int:
    if secret_type == "binary":
        return 2
    raise ValueError(f"Only binary secret_type is supported, got {secret_type}")


class CNN(nn.Module):
    input_kind = "image"

    def __init__(self, in_channels: int, n: int, secret_type: str, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n = n
        self.num_classes = num_secret_classes(secret_type)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Conv1d(width, self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if z.shape[-1] != self.n:
            raise ValueError(f"Expected width n={self.n}, got {z.shape[-1]}")
        z = z.mean(dim=2)
        z = self.dropout(z)
        out = self.head(z)
        return out.permute(0, 2, 1)


class AlexNet(nn.Module):
    input_kind = "image"

    def __init__(self, in_channels: int, n: int, secret_type: str, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n = n
        self.num_classes = num_secret_classes(secret_type)
        c1 = width
        c2 = width * 3
        c3 = width * 6
        c4 = width * 4
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.Conv2d(c1, c2, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Conv1d(c4, self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        if z.shape[-1] != self.n:
            raise ValueError(f"Expected width n={self.n}, got {z.shape[-1]}")
        z = z.mean(dim=2)
        z = self.dropout(z)
        return self.head(z).permute(0, 2, 1)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (1, 1),
        width_kernel: int = 3,
    ):
        super().__init__()
        if width_kernel not in (1, 3):
            raise ValueError(f"width_kernel must be 1 or 3, got {width_kernel}")
        kernel_size = (3, width_kernel)
        padding = (1, width_kernel // 2)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet(nn.Module):
    input_kind = "image"

    def __init__(
        self,
        in_channels: int,
        n: int,
        secret_type: str,
        width: int = 64,
        blocks: tuple[int, int, int] = (2, 2, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.num_classes = num_secret_classes(secret_type)
        channels = (width, width * 2, width * 4)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(
            channels[0],
            channels[0],
            blocks[0],
            stride=(1, 1),
            first_width_kernel=3,
            remaining_width_kernel=1,
        )
        self.layer2 = self._make_layer(
            channels[0],
            channels[1],
            blocks[1],
            stride=(2, 1),
            first_width_kernel=1,
        )
        self.layer3 = self._make_layer(
            channels[1],
            channels[2],
            blocks[2],
            stride=(2, 1),
            first_width_kernel=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Conv1d(channels[2], self.num_classes, kernel_size=1)

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: tuple[int, int],
        first_width_kernel: int,
        remaining_width_kernel: int = 1,
    ) -> nn.Sequential:
        layers = [
            BasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                width_kernel=first_width_kernel,
            )
        ]
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(
                    out_channels,
                    out_channels,
                    stride=(1, 1),
                    width_kernel=remaining_width_kernel,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        if z.shape[-1] != self.n:
            raise ValueError(f"Expected width n={self.n}, got {z.shape[-1]}")
        z = z.mean(dim=2)
        z = self.dropout(z)
        return self.head(z).permute(0, 2, 1)


class Hybrid(nn.Module):
    input_kind = "image"

    def __init__(
        self,
        in_channels: int,
        m: int,
        n: int,
        secret_type: str,
        embed_dim: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.num_classes = num_secret_classes(secret_type)

        stem_hidden = max(embed_dim // 2, 8)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_hidden, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.coord_proj = nn.Linear(embed_dim * m, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        batch_size, embed_dim, m, n = z.shape
        tokens = z.permute(0, 3, 1, 2).reshape(batch_size, n, embed_dim * m)
        tokens = self.coord_proj(tokens)
        if tokens.shape[1] != self.n:
            raise ValueError(f"Expected token width n={self.n}, got {tokens.shape[1]}")
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens)


def build_model(
    model_name: str,
    in_channels: int,
    m: int,
    n: int,
    secret_type: str,
    embed_dim: int,
    depth: int,
    num_heads: int,
    dropout: float,
) -> nn.Module:
    if model_name == "cnn":
        return CNN(
            in_channels=in_channels,
            n=n,
            secret_type=secret_type,
            width=embed_dim,
            dropout=dropout,
        )
    if model_name == "alexnet":
        return AlexNet(
            in_channels=in_channels,
            n=n,
            secret_type=secret_type,
            width=max(embed_dim // 2, 16),
            dropout=dropout,
        )
    if model_name == "resnet":
        return ResNet(
            in_channels=in_channels,
            n=n,
            secret_type=secret_type,
            width=max(embed_dim // 2, 16),
            dropout=dropout,
        )
    if model_name == "hybrid":
        return Hybrid(
            in_channels=in_channels,
            m=m,
            n=n,
            secret_type=secret_type,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_name: {model_name}")
