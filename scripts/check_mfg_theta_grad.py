import torch
from typing import Any, Dict, Tuple
from lpbf_to.losses.manufacturing_loss import ManufacturingLoss, Bounds


def _load_any(path: str) -> Any:
    # Try to silence the torch "weights_only" warning when possible
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_pv(obj: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accept pv_latest.pt saved in several possible formats:
      1) dict: {"P_raw": tensor, "v_raw": tensor}  (preferred)
      2) dict: common variants like {"p_raw":..., "v_raw":...}, {"P":...,"v":...}
      3) tuple/list: (P_raw, v_raw)
      4) tensor: shape (2,K) or (K,2) or (2*K,)  <-- IMPORTANT
    """
    # --- dict-like ---
    if isinstance(obj, dict):
        d: Dict[str, Any] = obj

        key_sets = [
            ("P_raw", "v_raw"),
            ("p_raw", "v_raw"),
            ("P", "v"),
            ("p", "v"),
            ("PRAW", "VRAW"),
        ]
        for kp, kv in key_sets:
            if kp in d and kv in d:
                return d[kp], d[kv]

        # fallback: try case-insensitive match
        lower = {str(k).lower(): k for k in d.keys()}
        if "p_raw" in lower and "v_raw" in lower:
            return d[lower["p_raw"]], d[lower["v_raw"]]
        if "p" in lower and "v" in lower:
            return d[lower["p"]], d[lower["v"]]

        raise ValueError(
            f"pv_latest dict keys found: {list(d.keys())} (expected P_raw/v_raw or similar)"
        )

    # --- tuple/list ---
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return obj[0], obj[1]

    # --- tensor ---
    if torch.is_tensor(obj):
        t = obj.detach()

        # (2*K,)  -> split into halves (P_raw first, v_raw second)
        if t.ndim == 1 and t.numel() % 2 == 0:
            K = t.numel() // 2
            return t[:K], t[K:]

        # (2,K)
        if t.ndim == 2 and t.shape[0] == 2:
            return t[0], t[1]

        # (K,2)
        if t.ndim == 2 and t.shape[1] == 2:
            return t[:, 0], t[:, 1]

        raise ValueError(
            f"pv_latest is a Tensor but shape is {tuple(t.shape)}; expected (2*K,), (2,K), or (K,2)"
        )

    raise ValueError(f"pv_latest has unsupported type: {type(obj)}")


def main() -> None:
    device = "cpu"
    dtype = torch.float32

    asset_dir = "/workspaces/lpbf_project/lpbf_to/surrogates/meltpool_v1"
    theta_path = "/workspaces/lpbf_project/scripts/theta_latest.pt"
    pv_path = "/workspaces/lpbf_project/scripts/pv_latest.pt"

    # --- load theta ---
    theta_any = _load_any(theta_path)
    if not torch.is_tensor(theta_any):
        raise TypeError(f"Expected tensor in {theta_path}, got {type(theta_any)}")

    theta = theta_any
    # theta saved as (1,nx,ny,nz) in your script
    if theta.ndim == 4 and theta.shape[0] == 1:
        theta = theta.squeeze(0)
    if theta.ndim != 3:
        raise ValueError(
            f"Expected theta to be 3D (nx,ny,nz) after squeeze, got shape {tuple(theta.shape)}"
        )

    theta = theta.to(dtype=dtype, device=device).clone().requires_grad_(True)
    nx, ny, nz = theta.shape

    # --- load P_raw / v_raw ---
    pv_any = _load_any(pv_path)
    P_raw_any, v_raw_any = _extract_pv(pv_any)

    if not torch.is_tensor(P_raw_any) or not torch.is_tensor(v_raw_any):
        raise TypeError(
            f"P_raw/v_raw must be tensors, got {type(P_raw_any)} and {type(v_raw_any)}"
        )

    P_raw = P_raw_any.to(dtype=dtype, device=device).clone().requires_grad_(True)
    v_raw = v_raw_any.to(dtype=dtype, device=device).clone().requires_grad_(True)

    # --- build tile indices (2x2x2 => K=8) ---
    tile_indices = torch.empty((nx, ny, nz), dtype=torch.int64, device=device)
    for ix in range(nx):
        tx = 0 if ix < nx // 2 else 1
        for iy in range(ny):
            ty = 0 if iy < ny // 2 else 1
            for iz in range(nz):
                tz = 0 if iz < nz // 2 else 1
                tile_indices[ix, iy, iz] = tx + 2 * ty + 4 * tz

    # --- ManufacturingLoss (match your smoke settings) ---
    mfg = ManufacturingLoss(
        asset_dir=asset_dir,
        tile_shape=(2, 2, 2),
        bounds_by_material={"IN718": Bounds(P_min=100, P_max=600, v_min=400, v_max=1100)},
        tv_weight=0.05,
        rho_power=1.0,
    )

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

    # ---- grads (fail fast) ----
    g_theta = theta.grad
    gP = P_raw.grad
    gv = v_raw.grad

    if g_theta is None:
        raise RuntimeError("theta.grad is None (backward didn't produce grads for theta).")
    if gP is None:
        raise RuntimeError("P_raw.grad is None (backward didn't produce grads for P_raw).")
    if gv is None:
        raise RuntimeError("v_raw.grad is None (backward didn't produce grads for v_raw).")

    print("[INFO] pv_latest type:", type(pv_any))
    if isinstance(pv_any, dict):
        print("[INFO] pv_latest keys:", list(pv_any.keys()))
    if torch.is_tensor(pv_any):
        print("[INFO] pv_latest tensor shape:", tuple(pv_any.shape))

    print("[MFG] loss:", float(loss))
    print("[theta] mean/min/max:", float(theta.mean()), float(theta.min()), float(theta.max()))

    print(
        "[GRAD θ]  norm:", float(g_theta.norm()),
        "min/max:", float(g_theta.min()), float(g_theta.max()),
        "mean/std:", float(g_theta.mean()), float(g_theta.std())
    )
    print("[GRAD P]  norm:", float(gP.norm()), "min/max:", float(gP.min()), float(gP.max()))
    print("[GRAD v]  norm:", float(gv.norm()), "min/max:", float(gv.min()), float(gv.max()))


if __name__ == "__main__":
    main()