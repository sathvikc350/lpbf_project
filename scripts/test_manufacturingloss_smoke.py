# pyright: reportMissingImports=false
# pyright: reportOptionalMemberAccess=false

# scripts/test_manufacturingloss_smoke.py

import sys
from pathlib import Path

import torch

# Ensure repo root is on sys.path when running as: python scripts/...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lpbf_to.losses.manufacturing_loss import ManufacturingLoss, Bounds  # noqa: E402


def zero_grads(*params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def require_grad(p: torch.nn.Parameter) -> torch.Tensor:
    """Return grad tensor and assert it exists (prevents 'None' editor warnings)."""
    assert p.grad is not None, "Expected .grad to be populated after backward(), but it is None."
    return p.grad


def make_active_mask_for_tile(tile_indices: torch.Tensor, tile_id: int) -> torch.Tensor:
    return (tile_indices == tile_id).to(dtype=torch.float32)


def neighbor_tiles_2x2x2(tile_id: int):
    nx, ny, nz = 2, 2, 2
    ix = tile_id % nx
    iy = (tile_id // nx) % ny
    iz = tile_id // (nx * ny)

    def tid(x, y, z):
        return x + nx * (y + ny * z)

    neigh = []
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        x, y, z = ix + dx, iy + dy, iz + dz
        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
            neigh.append(tid(x, y, z))
    return sorted(neigh)


def grad_nonzero_tiles(g: torch.Tensor, tol: float = 1e-12):
    return (g.abs() > tol).nonzero(as_tuple=False).flatten().tolist()


def main():
    torch.manual_seed(0)

    tile_shape = (2, 2, 2)
    K = 8
    grid_shape = (32, 32, 16)
    tile_A = 0

    P_raw = torch.nn.Parameter(torch.zeros(K))
    v_raw = torch.nn.Parameter(torch.zeros(K))

    rho = torch.rand(*grid_shape)
    tile_indices = torch.randint(0, K, grid_shape, dtype=torch.int64)

    mfg = ManufacturingLoss(
        asset_dir="lpbf_to/surrogates/meltpool_v1",
        tile_shape=tile_shape,
        bounds_by_material={"IN718": Bounds(P_min=200, P_max=400, v_min=400, v_max=1100)},
        tv_weight=0.0,
        rho_power=4.0,
    )

    material = "IN718"
    h_mm = 0.10
    t_mm = 0.04

    # --- forced LOF thresholds ---
    with torch.no_grad():
        out0 = mfg(
            rho=rho,
            P_raw=P_raw,
            v_raw=v_raw,
            tile_indices=tile_indices,
            material=material,
            h_mm=h_mm,
            t_mm=t_mm,
            d_req_um=0.0,
            d_max_um=1e12,
        )
        depth_mean = float(out0["depth_stats"][1].cpu())

    d_req_um = depth_mean + 1000.0
    d_max_um = depth_mean + 1e12

    print("\n--- Smoke Tests for ManufacturingLoss ---")
    print("tile_shape:", tile_shape, "grid_shape:", grid_shape, "tile_A:", tile_A)
    print("Using forced thresholds:", {"d_req_um": d_req_um, "d_max_um": d_max_um})

    active_mask = make_active_mask_for_tile(tile_indices, tile_A)

    # ============================================================
    print("\n[Test A] Isolation (TV OFF)")

    mfg.tv_weight = 0.0
    zero_grads(P_raw, v_raw)

    outA = mfg(
        rho=rho,
        P_raw=P_raw,
        v_raw=v_raw,
        tile_indices=tile_indices,
        material=material,
        h_mm=h_mm,
        t_mm=t_mm,
        d_req_um=d_req_um,
        d_max_um=d_max_um,
        active_mask=active_mask,
    )
    outA["loss_total_mfg"].backward()

    gP = require_grad(P_raw).detach().cpu()
    gv = require_grad(v_raw).detach().cpu()

    nonzero_P = grad_nonzero_tiles(gP)
    nonzero_v = grad_nonzero_tiles(gv)

    print("Non-zero grad tiles (P_raw):", nonzero_P)
    print("Non-zero grad tiles (v_raw):", nonzero_v)

    assert nonzero_P == [tile_A], f"Expected only tile {tile_A} to have P grad, got {nonzero_P}"
    assert nonzero_v == [tile_A], f"Expected only tile {tile_A} to have v grad, got {nonzero_v}"

    # ============================================================
    
    with torch.no_grad():
        step = 0.1
        P_raw.data[tile_A] += step
        v_raw.data[tile_A] -= step

   
    print("\n[Test B] Coupling (TV ON)")

    mfg.tv_weight = 1.0
    zero_grads(P_raw, v_raw)

    outB = mfg(
        rho=rho,
        P_raw=P_raw,
        v_raw=v_raw,
        tile_indices=tile_indices,
        material=material,
        h_mm=h_mm,
        t_mm=t_mm,
        d_req_um=d_req_um,
        d_max_um=d_max_um,
        active_mask=active_mask,
    )
    outB["loss_total_mfg"].backward()

    gP = require_grad(P_raw).detach().cpu()
    gv = require_grad(v_raw).detach().cpu()

    nonzero_P = grad_nonzero_tiles(gP)
    nonzero_v = grad_nonzero_tiles(gv)

    neigh = neighbor_tiles_2x2x2(tile_A)
    expected = sorted([tile_A] + neigh)

    print("Expected non-zero tiles:", expected, "(tile_A + neighbors)")
    print("Non-zero grad tiles (P_raw):", nonzero_P)
    print("Non-zero grad tiles (v_raw):", nonzero_v)

    for tid in expected:
        assert tid in nonzero_P, f"Expected tile {tid} in P grads, missing"
        assert tid in nonzero_v, f"Expected tile {tid} in v grads, missing"

    far_tiles = sorted(set(range(K)) - set(expected))
    farP = gP[far_tiles].abs().max().item() if far_tiles else 0.0
    farv = gv[far_tiles].abs().max().item() if far_tiles else 0.0
    print("Max |grad| on far tiles: P_raw=", farP, " v_raw=", farv)
    assert farP < 1e-6 and farv < 1e-6, "Far tiles should have ~0 grads when TV couples locally"

    #
    print("\n[Forced LOF] Gradient direction sanity (local tile_A)")

    mfg.tv_weight = 0.0
    zero_grads(P_raw, v_raw)

    outC = mfg(
        rho=rho,
        P_raw=P_raw,
        v_raw=v_raw,
        tile_indices=tile_indices,
        material=material,
        h_mm=h_mm,
        t_mm=t_mm,
        d_req_um=d_req_um,
        d_max_um=d_max_um,
        active_mask=active_mask,
    )

    L_lof = outC["loss_lof"]
    assert float(L_lof.detach().cpu()) > 0.0, "Expected LOF > 0 under forced activation"
    L_lof.backward()

    gP = require_grad(P_raw).detach().cpu()
    gv = require_grad(v_raw).detach().cpu()

    print("tile_A grad signs: dL/dP_raw =", float(gP[tile_A]), " dL/dv_raw =", float(gv[tile_A]))

    print("\nALL TESTS PASSED ")


if __name__ == "__main__":
    main()
