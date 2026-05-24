from __future__ import annotations

import math

import torch
from torch import nn


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = out[..., : x.shape[-1]]
        return out + x


class TCNForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, levels: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_dim, hidden_dim, 1)
        self.blocks = nn.Sequential(
            *[TCNBlock(hidden_dim, kernel_size=3, dilation=2**i, dropout=dropout) for i in range(levels)]
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input(x)
        x = self.blocks(x)
        return self.head(x)


class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(self.proj(x))
        x = self.encoder(x)
        return self.head(x[:, -1])


class DLinearForecaster(nn.Module):
    def __init__(self, seq_len: int, input_dim: int) -> None:
        super().__init__()
        self.trend = nn.Linear(seq_len, 1)
        self.seasonal = nn.Linear(seq_len, 1)
        self.target = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, features]
        moving = torch.nn.functional.avg_pool1d(
            x.transpose(1, 2),
            kernel_size=3,
            stride=1,
            padding=1,
        ).transpose(1, 2)
        seasonal = x - moving
        trend_out = self.trend(moving.transpose(1, 2)).squeeze(-1)
        seasonal_out = self.seasonal(seasonal.transpose(1, 2)).squeeze(-1)
        return self.target(trend_out + seasonal_out)


class NLinearForecaster(nn.Module):
    def __init__(self, seq_len: int, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(seq_len, 1)
        self.target = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:, :]
        normalized = x - last
        out = self.linear(normalized.transpose(1, 2)).squeeze(-1) + last.squeeze(1)
        return self.target(out)


class PatchTSTForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        patch_len: int = 8,
        stride: int = 4,
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = 1 + max(0, (seq_len - patch_len) // stride)
        self.patch_proj = nn.Linear(patch_len, hidden_dim)
        self.channel_embed = nn.Parameter(torch.randn(1, input_dim, 1, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(hidden_dim * input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-independent patching: [B, T, C] -> [B*C, P, patch_len]
        x = x.transpose(1, 2)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        encoded = self.patch_proj(patches) + self.channel_embed
        b, c, p, h = encoded.shape
        encoded = self.encoder(encoded.reshape(b * c, p, h)).mean(dim=1).reshape(b, c * h)
        return self.head(encoded)


class ITransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 64, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.value_proj = nn.Linear(seq_len, hidden_dim)
        self.variable_embed = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(hidden_dim * input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inverted tokenization: variables are tokens, history is feature vector.
        x = x.transpose(1, 2)
        x = self.value_proj(x) + self.variable_embed
        x = self.encoder(x)
        return self.head(x.flatten(start_dim=1))


class TimesBlockForecaster(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.period_kernels = nn.ModuleList(
            [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0)),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            ]
        )
        self.seq_len = seq_len
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x.transpose(1, 2))
        period = max(2, int(math.sqrt(self.seq_len)))
        pad = (period - (x.shape[-1] % period)) % period
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        b, h, t = x.shape
        x2 = x.reshape(b, h, t // period, period)
        mixed = sum(kernel(x2) for kernel in self.period_kernels) / len(self.period_kernels)
        return self.head(mixed.reshape(b, h, t))


class SequenceAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 24) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(x)
        latent = self.to_latent(hidden[-1])
        decoded_hidden = self.from_latent(latent).unsqueeze(0)
        zeros = torch.zeros_like(x)
        out, _ = self.decoder(zeros, decoded_hidden)
        return self.output(out)


class SAITSLikeImputer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim * 2, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder_first = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder_second = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, observed_mask: torch.Tensor | None = None) -> torch.Tensor:
        if observed_mask is None:
            observed_mask = torch.ones_like(x)
        h = self.input(torch.cat([x, observed_mask], dim=-1))
        first = self.encoder_first(self.pos(h))
        refined = self.encoder_second(first)
        return self.out(refined)


class TranADLikeAnomalyDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim * 2, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = torch.zeros_like(x)
        h = self.proj(torch.cat([x, context], dim=-1))
        h = self.encoder(self.pos(h))
        first = self.decoder(h)
        residual_context = torch.abs(first - x)
        h2 = self.proj(torch.cat([x, residual_context], dim=-1))
        return self.decoder(self.encoder(self.pos(h2)))


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(self.pos(self.proj(x)))
        return self.head(x[:, -1])


class STGCNForecaster(nn.Module):
    def __init__(self, nodes: int, seq_len: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.temporal = nn.Conv1d(nodes, hidden_dim, kernel_size=3, padding=1)
        self.graph_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.out = nn.Sequential(nn.ReLU(), nn.Flatten(), nn.Linear(hidden_dim * seq_len, nodes))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: [batch, nodes, seq]
        graph_x = torch.einsum("ij,bjt->bit", adjacency, x)
        x = self.temporal(x + graph_x)
        return self.out(x)
