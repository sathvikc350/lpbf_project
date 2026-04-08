import math, time, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lpbf_to.surrogates.mlp import MeltPoolMLP

torch.manual_seed(42)
os.makedirs("outputs", exist_ok=True)

# ---- Synthetic DOE (very small & fast) ----
N = 2000
P_W   = torch.rand(N)*180 + 120      # 120..300 W
v_mm  = torch.rand(N)*600 + 200      # 200..800 mm/s
h_um  = torch.rand(N)*80  + 60       # 60..140 µm
t_um  = torch.rand(N)*30  + 30       # 30..60 µm

# two simple "geometry" flags (kept minimal for smoke test)
local_thickness_um = (torch.rand(N)*4 + 2) * t_um       # ~2..6 × layer thickness
overhang_deg       = torch.rand(N)*60 + 15              # 15..75°

params = torch.stack([P_W, v_mm, h_um, t_um], dim=1)
geom   = torch.stack([local_thickness_um, overhang_deg], dim=1)

# Simple physically-inspired targets (just to learn a monotone trend quickly)
line_energy = P_W / v_mm                       # J/mm (scaled)
width_um = (100.0 * line_energy) + 0.4*h_um    # grows with P, shrinks with v
depth_um = (1.8 * line_energy * (t_um/40.0)) + 0.15*h_um

# Risk labels from thresholds (coarse rules)
risk_lof = (width_um < 0.9*h_um) | (depth_um <= t_um)
risk_key = (depth_um >= 2.0*t_um)

Y = torch.stack([width_um, depth_um,
                 risk_lof.float(), risk_key.float()], dim=1)

# Train/val split
idx = torch.randperm(N)
trn, val = idx[:1600], idx[1600:]
Xtr, Gtr, Ytr = params[trn], geom[trn], Y[trn]
Xva, Gva, Yva = params[val], geom[val], Y[val]

ds_tr = TensorDataset(Xtr, Gtr, Ytr)
ds_va = TensorDataset(Xva, Gva, Yva)
dl_tr = DataLoader(ds_tr, batch_size=256, shuffle=True)
dl_va = DataLoader(ds_va, batch_size=512, shuffle=False)

# ---- Model ----
model = MeltPoolMLP(in_dim=6, hidden=128)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
l1 = nn.L1Loss()
bce = nn.BCEWithLogitsLoss()

def step(batch):
    xb, gb, yb = batch
    w_true, d_true = yb[:,0], yb[:,1]
    lof_true, key_true = yb[:,2], yb[:,3]

    w_pred, d_pred, risks = model(xb, gb)
    lof_logit, key_logit = risks[:,0], risks[:,1]

    loss = l1(w_pred, w_true) + l1(d_pred, d_true) \
         + bce(lof_logit, lof_true) + bce(key_logit, key_true)
    return loss, (w_true, d_true, w_pred, d_pred)

def eval_mae():
    model.eval()
    w_mae = d_mae = 0.0
    n = 0
    with torch.no_grad():
        for xb, gb, yb in dl_va:
            w_true, d_true = yb[:,0], yb[:,1]
            w_pred, d_pred, _ = model(xb, gb)
            w_mae += torch.abs(w_pred - w_true).sum().item()
            d_mae += torch.abs(d_pred - d_true).sum().item()
            n += xb.size(0)
    return w_mae/n, d_mae/n

# ---- Train (very short) ----
t0 = time.time()
for epoch in range(10):
    model.train()
    for batch in dl_tr:
        opt.zero_grad(set_to_none=True)
        loss, _ = step(batch)
        loss.backward()
        opt.step()
    if (epoch+1) in (1,5,10):
        w_mae, d_mae = eval_mae()
        print(f"Epoch {epoch+1:02d} | Val MAE: width {w_mae:.2f} µm, depth {d_mae:.2f} µm")
print(f"Training took {time.time()-t0:.1f}s")

# ---- Save small artifacts & a tiny sample table for your update ----
torch.save(model.state_dict(), "outputs/surrogate_synth.pt")
model.eval()
with torch.no_grad():
    w_pred, d_pred, risks = model(Xva[:20], Gva[:20])
    df = pd.DataFrame({
        "P_W": Xva[:20,0].numpy().round(1),
        "v_mm_s": Xva[:20,1].numpy().round(1),
        "h_um": Xva[:20,2].numpy().round(1),
        "t_um": Xva[:20,3].numpy().round(1),
        "pred_width_um": w_pred.numpy().round(1),
        "pred_depth_um": d_pred.numpy().round(1),
    })
    df.to_csv("outputs/synth_eval.csv", index=False)

print("Saved: outputs/surrogate_synth.pt and outputs/synth_eval.csv")
