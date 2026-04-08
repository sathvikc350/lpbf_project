import torch
import torch.nn as nn

class MeltPoolMLP(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4)  # width_um, depth_um, risk_lof_logit, risk_key_logit
        )

    def forward(self, params, geom_feats):
        x = torch.cat([params, geom_feats], dim=-1)
        y = self.net(x)
        width = torch.relu(y[...,0]) + 1e-3
        depth = torch.relu(y[...,1]) + 1e-3
        risks = y[...,2:4]
        return width, depth, risks
