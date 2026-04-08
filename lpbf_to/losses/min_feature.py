import torch

class MinFeatureLoss(torch.nn.Module):
    def __init__(self, min_thickness_um=400.0):
        super().__init__()
        self.min_t = torch.tensor(min_thickness_um, dtype=torch.float32)

    def forward(self, local_thickness_um):  # (N,)
        viol = torch.relu(self.min_t - local_thickness_um)
        return viol.mean()
