# ml_dl/dl_models.py
import torch
import torch.nn as nn


class TemporalConvNet(nn.Module):
    def __init__(self, in_dim, hid=64, levels=4, kernel=3, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        layers = []
        d_in = in_dim
        for l in range(levels):
            dilation = 2 ** l
            pad = (kernel - 1) * dilation
            layers += [
                nn.Conv1d(d_in, hid, kernel, padding=pad, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hid, hid, kernel, padding=pad, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            d_in = hid
        self.net = nn.Sequential(*layers)
        self.head_ret_reg = nn.Linear(hid, 1)
        self.head_ret_cls = nn.Linear(hid, 2)
        self.head_rv_reg  = nn.Linear(hid, 1)

    def forward(self, x):               # x: [B, L, F]
        x = x.transpose(1, 2)           # [B, F, L]
        h = self.net(x)[:, :, -1]       # [B, H]
        return {
            "ret_reg": self.head_ret_reg(h).squeeze(-1),
            "ret_cls_logits": self.head_ret_cls(h),
            "rv_reg": self.head_rv_reg(h).squeeze(-1),
        }


class TinyTransformer(nn.Module):
    def __init__(self, in_dim, d_model=64, nhead=4, nlayers=2, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head_ret_reg = nn.Linear(d_model, 1)
        self.head_ret_cls = nn.Linear(d_model, 2)
        self.head_rv_reg  = nn.Linear(d_model, 1)

    def forward(self, x):               # [B, L, F]
        z = self.proj(x)
        h = self.enc(z)[:, -1, :]       # last token
        return {
            "ret_reg": self.head_ret_reg(h).squeeze(-1),
            "ret_cls_logits": self.head_ret_cls(h),
            "rv_reg": self.head_rv_reg(h).squeeze(-1),
        }


class TinyLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head_ret = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_rv  = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_cls = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):  # x: [B, T, F]
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]  # [B, hidden]
        return {
            "ret_reg": self.head_ret(h).squeeze(-1),
            "rv_reg":  self.head_rv(h).squeeze(-1),
            "ret_cls_logits": self.head_cls(h),
        }
