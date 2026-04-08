import torch

class OverhangLoss(torch.nn.Module):
    def __init__(self, build_dir='z', angle_deg=45.0):
        super().__init__()
        self.build_dir = build_dir
        self.angle = torch.deg2rad(torch.tensor(angle_deg, dtype=torch.float32))

    def forward(self, normals):  # normals: (N,3) unit vectors
        # Penalize faces flatter than allowed angle (relative to +Z)
        z = torch.tensor([0.,0.,1.], dtype=normals.dtype, device=normals.device)
        cos_theta = (normals * z).sum(-1).clamp(-1,1)
        theta = torch.acos(cos_theta)  # angle from +Z
        viol = torch.relu(self.angle - theta)  # positive when too flat
        return viol.mean()
