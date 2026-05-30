import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: MODEL ARCHITECTURES  [CHANGES 4 + 7]
#   [4] Dual-scale CNN: CNNShort (20d) + CNNMedium (60d)
#   [7] Regime-conditioned fusion: ADX regime embedded and fed to dense head
# ══════════════════════════════════════════════════════════════════════════════


class CNNShortBranch(nn.Module):
    """
    Short-window CNN (20-day): learns inflection patterns and crossover shapes.
    Lighter architecture for the shorter window.
    """
    def __init__(self, in_features):
        super().__init__()
        self.conv1   = nn.Conv1d(in_features, 64, kernel_size=3, padding=1)
        self.conv2   = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool    = nn.AdaptiveAvgPool1d(4)
        self.dropout = nn.Dropout(0.3)
        self.out_dim = 128 * 4

    def forward(self, x):
        # x: (B, seq_short, F) → (B, F, seq_short)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x));  x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.flatten(1)   # (B, 128*4=512)

class CNNMediumBranch(nn.Module):
    """
    Medium-window CNN (60-day): learns trend structure and morphological patterns.
    Deeper architecture for the longer window.
    """
    def __init__(self, in_features):
        super().__init__()
        self.conv1   = nn.Conv1d(in_features, 64,  kernel_size=5, padding=2)
        self.conv2   = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3   = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool    = nn.AdaptiveAvgPool1d(8)
        self.dropout = nn.Dropout(0.3)
        self.out_dim = 64 * 8

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x));  x = self.dropout(x)
        x = F.relu(self.conv2(x));  x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.flatten(1)   # (B, 64*8=512)

class GRUBranch(nn.Module):
    """
    GRU (60-day): captures regime persistence and long-horizon dependencies.
    """
    def __init__(self, in_features, hidden=128, n_layers=2):
        super().__init__()
        self.gru     = nn.GRU(in_features, hidden, num_layers=n_layers,
                              batch_first=True,
                              dropout=0.3 if n_layers > 1 else 0)
        self.dropout = nn.Dropout(0.3)
        self.out_dim = hidden

    def forward(self, x):
        out, _ = self.gru(x)
        return self.dropout(out[:, -1, :])   # last hidden state

class DualCNNGRUFusion(nn.Module):
    """
    Full dual-scale CNN-GRU fusion with regime conditioning.  [v2]

    Components:
      cnn_short  : CNN on 20-day window  → inflection signals
      cnn_medium : CNN on 60-day window  → trend structure
      gru        : GRU on 60-day window  → regime persistence
      regime_emb : embedding of ADX regime (3 states)  [CHANGE 7]
      fusion head: concat all → BN → dense → 4-class softmax

    Ablation modes:
      'cnn_only'    : uses only cnn_medium (as in v1)
      'gru_only'    : uses only gru
      'dual_cnn'    : cnn_short + cnn_medium (no GRU)
      'fusion'      : all three + regime embedding  [full model]
    """
    def __init__(self, in_features, n_classes=4, mode="fusion"):
        super().__init__()
        self.mode       = mode
        self.cnn_short  = CNNShortBranch(in_features)
        self.cnn_medium = CNNMediumBranch(in_features)
        self.gru        = GRUBranch(in_features)
        self.regime_emb = nn.Embedding(3, 8)   # 3 ADX regimes → 8-dim embedding

        if mode == "fusion":
            fc_in = self.cnn_short.out_dim + self.cnn_medium.out_dim + self.gru.out_dim + 8
        elif mode == "dual_cnn":
            fc_in = self.cnn_short.out_dim + self.cnn_medium.out_dim
        elif mode == "cnn_only":
            fc_in = self.cnn_medium.out_dim
        elif mode == "gru_only":
            fc_in = self.gru.out_dim

        self.fc1     = nn.Linear(fc_in, 256)
        self.fc2     = nn.Linear(256, 128)
        self.fc3     = nn.Linear(128, 64)
        self.out     = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(0.3)
        self.bn1     = nn.BatchNorm1d(256)
        self.bn2     = nn.BatchNorm1d(128)
        self.bn3     = nn.BatchNorm1d(64)

    def forward(self, xs, xm, regime):
        if self.mode == "fusion":
            cs  = self.cnn_short(xs)
            cm  = self.cnn_medium(xm)
            g   = self.gru(xm)
            re  = self.regime_emb(regime)
            z   = torch.cat([cs, cm, g, re], dim=1)
        elif self.mode == "dual_cnn":
            cs  = self.cnn_short(xs)
            cm  = self.cnn_medium(xm)
            z   = torch.cat([cs, cm], dim=1)
        elif self.mode == "cnn_only":
            z   = self.cnn_medium(xm)
        elif self.mode == "gru_only":
            z   = self.gru(xm)

        z = F.relu(self.bn1(self.fc1(z)));  z = self.dropout(z)
        z = F.relu(self.bn2(self.fc2(z)));  z = self.dropout(z)
        z = F.relu(self.bn3(self.fc3(z)));  z = self.dropout(z)
        return self.out(z)
