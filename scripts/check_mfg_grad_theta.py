cd /workspaces/lpbf_project
python - <<'PY'
import torch
from lpbf_to.losses.manufacturing_loss import ManufacturingLoss, Bounds

device="cpu"
dtype=torch.float32
asset_dir="/workspaces/lpbf_project/lpbf_to/surrogates/meltpool_v1"

# load latest theta
theta = torch.load("/workspaces/lpbf_project/scripts/theta_latest.pt", map_location="cpu").to(dtype).squeeze(0)
nx, ny, nz = theta.shape

# load latest P_raw/v_raw
pv = torch.load("/workspaces/lpbf_project/scripts/pv_latest.pt", map_location="cpu")
P_raw = pv["P_raw"].to(dtype).requires_grad_(True)
v_raw = pv["v_raw"].to(dtype).requires_grad_(True)

# tile indices (2x2x2)
tile_indices = torch.empty((nx, ny, nz), dtype=torch.int64)
for ix in range(nx):
    tx = 0 if ix < nx//2 else 1
    for iy in range(ny):
        ty = 0 if iy < ny//2 else 1
        for iz in range(nz):
            tz = 0 if iz < nz//2 else 1
            tile_indices[ix,iy,iz] = tx + 2*ty + 4*tz

mfg = ManufacturingLoss(
    asset_dir=asset_dir,
    tile_shape=(2,2,2),
    bounds_by_material={"IN718": Bounds(P_min=100, P_max=600, v_min=400, v_max=1100)},
    tv_weight=0.05,
    rho_power=1.0,
)

theta = theta.clone().requires_grad_(True)

out = mfg(
    rho=theta,
    P_raw=P_raw,
    v_raw=v_raw,
    tile_indices=tile_indices,
    material="IN718",
    h_mm=0.11,
    t_mm=0.04,
    d_req_um=80.0,
    d_max_um=200.0,
)
loss = out["loss_total_mfg"]
loss.backward()

g = theta.grad
print("[MFG] loss:", float(loss))
print("[grad θ] norm:", float(g.norm()))
print("[grad θ] min/max:", float(g.min()), float(g.max()))
print("[grad θ] mean/std:", float(g.mean()), float(g.std()))
