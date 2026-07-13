"""
Hybrid architecture for binary stress classification:
  - CNN+LSTM branch reads the raw 30-sample sequence (GSR_z, RR_z, dGSR_z)
  - MLP branch reads the 5 engineered scalar features (RMSSD, SDNN,
    GSR slope, GSR range, GSR peak count)
  - Both branches are concatenated before the final classifier head.

Change from the previous version: instead of taking only the LSTM's
final timestep as the sequence representation (which discards whatever
happened earlier in the 30-second window), an additive attention layer
learns a weighted combination over ALL 15 timesteps. A GSR stress
response can peak anywhere in the window, not necessarily right at the
end, so this should let the model use information it was previously
throwing away.

Input:  x_seq  [batch, window_size, n_channels]
        x_feat [batch, n_scalar_features]
Output: [batch, 2] logits
"""
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Learns a softmax weight per timestep and returns the weighted sum."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: [B, T, H]
        scores = self.attn(lstm_out).squeeze(-1)      # [B, T]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        context = (lstm_out * weights).sum(dim=1)     # [B, H]
        return context, weights.squeeze(-1)


class CNNLSTMHybrid(nn.Module):
    def __init__(self, n_channels=3, cnn_channels=(16, 32), lstm_hidden=64,
                 lstm_layers=1, n_scalar_features=5, feat_hidden=16,
                 n_classes=2, dropout=0.45):
        super().__init__()
        c1, c2 = cnn_channels

        # --- sequence branch: CNN + BiLSTM + attention pooling ---
        self.conv1 = nn.Conv1d(n_channels, c1, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = AttentionPool(lstm_hidden * 2)

        # --- engineered-feature branch: small MLP ---
        self.feat_fc1 = nn.Linear(n_scalar_features, feat_hidden)
        self.feat_bn = nn.BatchNorm1d(feat_hidden)

        # --- combined classifier head ---
        combined_dim = lstm_hidden * 2 + feat_hidden
        self.fc1 = nn.Linear(combined_dim, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x_seq, x_feat, return_attention=False):
        # sequence branch: [B, T, C] -> [B, C, T] for conv
        x = x_seq.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # back to [B, T', C'] for LSTM
        out, _ = self.lstm(x)
        seq_repr, attn_weights = self.attention(out)

        # engineered-feature branch
        feat_repr = self.relu(self.feat_bn(self.feat_fc1(x_feat)))

        # combine
        combined = torch.cat([seq_repr, feat_repr], dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        logits = self.fc2(x)

        if return_attention:
            return logits, attn_weights
        return logits