"""ConvLSTM1D classifier (PyTorch port of Figure 3 of the paper).

PyTorch has no built-in ConvLSTM, so a ConvLSTM1D cell is implemented here.
With kernel_size=1 (the paper's setting) the convolution does no spatial mixing,
so this behaves like a position-wise LSTM with 8 hidden channels shared across
the feature axis.

Input is (B, T, A) where T = N - 1 = 4 (frame-difference time steps) and A is
the feature count per frame. It is reshaped to (B, T, 1, A) — one input channel,
A spatial positions — before the recurrence.

Pipeline: ConvLSTM1D(filters=8, kernel=1) → Flatten → Linear(2048) → ReLU
          → Linear(1024) → ReLU → Linear(num_classes).
CrossEntropyLoss applies log-softmax internally, so no softmax layer here.
"""
from __future__ import annotations

import torch
from torch import nn

from .config import TrainConfig


class ConvLSTM1DCell(nn.Module):
    """Single ConvLSTM1D timestep (standard ConvLSTM, no peephole terms).

    Keras' ConvLSTM1D — what the paper actually trained — omits the peephole
    Hadamard terms shown in the paper's eq. 3-7, so this port omits them too.
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        # One conv produces all four gates (input, forget, cell, output).
        self.conv = nn.Conv1d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gates = self.conv(torch.cat([x, h_prev], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM1D(nn.Module):
    """Unrolls a ConvLSTM1DCell over the time axis.

    Input  : (B, T, C_in, L)
    Output : (B, hidden_channels, L)  — last hidden state (return_sequences=False).
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTM1DCell(in_channels, hidden_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _, length = x.shape
        h = x.new_zeros(b, self.hidden_channels, length)
        c = x.new_zeros(b, self.hidden_channels, length)
        for step in range(t):
            h, c = self.cell(x[:, step], (h, c))
        return h


class ConvLSTM1DClassifier(nn.Module):
    def __init__(
        self,
        time_steps: int,
        feature_dim: int,
        num_classes: int,
        cfg: TrainConfig = TrainConfig(),
    ):
        super().__init__()
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.convlstm = ConvLSTM1D(
            in_channels=1,
            hidden_channels=cfg.convlstm_filters,
            kernel_size=cfg.convlstm_kernel,
        )
        flat_dim = cfg.convlstm_filters * feature_dim
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, cfg.dense1),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.dense1, cfg.dense2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.dense2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, A) → (B, T, 1, A)
        x = x.unsqueeze(2)
        x = self.convlstm(x)            # (B, filters, A)
        return self.mlp(x)              # (B, num_classes) logits


def build_model(
    time_steps: int,
    feature_dim: int,
    num_classes: int,
    cfg: TrainConfig = TrainConfig(),
) -> ConvLSTM1DClassifier:
    return ConvLSTM1DClassifier(time_steps, feature_dim, num_classes, cfg)
