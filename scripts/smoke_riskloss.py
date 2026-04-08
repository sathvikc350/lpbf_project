import torch
from lpbf_to.surrogates.mlp import MeltPoolMLP
from lpbf_to.losses.process_risk import ProcessRiskLoss

# Load the trained surrogate
model = MeltPoolMLP(in_dim=6, hidden=128)
sd = torch.load("outputs/surrogate_synth.pt", map_location="cpu")
model.load_state_dict(sd)
model.eval()

# Tiny batch
params = torch.tensor([[200., 400., 100., 40.],
                       [140., 700.,  80., 50.]], dtype=torch.float32)
geom   = torch.tensor([[120., 30.],
                       [180., 50.]], dtype=torch.float32)

loss_fn = ProcessRiskLoss(surrogate=model, width_band_um=(80.,120.), risk_weights=(1.0,1.0))
loss = loss_fn(params, geom)
print(f"ProcessRiskLoss (batch=2) = {loss.item():.3f}")
