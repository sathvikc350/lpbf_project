#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

REPO = Path("/workspaces/lpbf_project")
SCRIPTS = REPO / "scripts"
SMOKE_PATH = SCRIPTS / "smoke_joint_opt_fast.py"

# Must match your Phase-1 / smoke script bounds (IN718)
P_MIN, P_MAX = 100.0, 600.0
V_MIN, V_MAX = 400.0, 1100.0

# Diagnostics conventions
OVERHANG_DEG_FROM_HORIZONTAL = 45.0   # violation if angle < 45 deg from horizontal plate
SOLID_THRESH = 0.5                   # used for island connectivity and surface selection
SURFACE_GRAD_THRESH = 1e-6           # avoids division by ~0 normals

# VED assumed constants (documented as placeholders)
ASSUMED_HATCH_MM = 0.1               # 100 um
ASSUMED_LAYER_MM = 0.04              # 40 um

# --- Industrial Patch defaults (locked conventions) ---
VOXEL_SIZE_MM = 0.25                 # placeholder; recorded in manifest
MINFEATURE_EROSION_PASSES = 1        # 1-pass erosion
RECOATER_STEP_VOX = 2                # cliff threshold in voxels (top-z heightmap)
THERMAL_CLUSTER_KERNEL = (5, 5, 5)   # local average window
THERMAL_CLUSTER_THRESH = 0.85        # avg density threshold for "massive chunk"


# -----------------------------
# Helpers: hashing, env, git
# -----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_version(modname: str) -> str:
    try:
        m = __import__(modname)
        return getattr(m, "__version__", "UNKNOWN")
    except Exception:
        return "NOT_INSTALLED"


def env_snapshot() -> Dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "UNKNOWN"),
        "numpy": safe_version("numpy"),
        "dl4to": safe_version("dl4to"),
        "lpbf_to": safe_version("lpbf_to"),
    }


def git_commit(repo: Path) -> str:
    # avoid noisy fatal errors when repo has no .git
    if not (repo / ".git").exists():
        return "UNKNOWN"
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo))
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def fingerprint_dir_sha256(root: Path, relpaths: List[Path]) -> Dict[str, str]:
    """
    Hash important baseline files so the Phase-2 manifest cryptographically binds
    to the exact baseline folder used.
    """
    out: Dict[str, str] = {}
    for rp in relpaths:
        fp = root / rp
        key = str(rp)
        if fp.exists() and fp.is_file():
            out[key] = sha256_file(fp)
        else:
            out[key] = "MISSING"
    return out


def percentiles(x: torch.Tensor, ps=(5, 50, 95)) -> Dict[str, float]:
    v = x.detach().flatten().float().cpu()
    if v.numel() == 0:
        return {f"p{p}": float("nan") for p in ps}
    q = torch.quantile(v, torch.tensor([p / 100.0 for p in ps]))
    return {f"p{p}": float(q[i].item()) for i, p in enumerate(ps)}


# -----------------------------
# Smoke import (projection/eta)
# -----------------------------
def import_smoke_module(smoke_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("smoke_joint_opt_fast", str(smoke_path))
    assert spec and spec.loader
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)  # type: ignore
    return smoke


# -----------------------------
# Phase-1 tile decoding helpers
# -----------------------------
def map_tanh_to_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    u = torch.tanh(x)  # [-1, 1]
    return lo + (u + 1.0) * 0.5 * (hi - lo)


def build_tile_indices(nx: int, ny: int, nz: int) -> torch.Tensor:
    """
    Mirrors your 2x2x2 K-tile mapping:
    tile id = tx + 2*ty + 4*tz
    """
    tile_indices = torch.empty((nx, ny, nz), dtype=torch.int64)
    for ix in range(nx):
        tx = 0 if ix < nx // 2 else 1
        for iy in range(ny):
            ty = 0 if iy < ny // 2 else 1
            for iz in range(nz):
                tz = 0 if iz < nz // 2 else 1
                tile_indices[ix, iy, iz] = tx + 2 * ty + 4 * tz
    return tile_indices


def make_design_mask(theta: torch.Tensor) -> torch.Tensor:
    """
    Matches your smoke Omega_design convention:
    x=0 plane is NOT design (fixed), everything else design.
    """
    mask = torch.ones_like(theta, dtype=torch.bool)
    mask[:, 0, :, :] = False
    return mask


# -----------------------------
# Visualization helpers
# -----------------------------
def save_slices_png(out_png: Path, vol: torch.Tensor, title_prefix: str) -> None:
    """
    vol: (NX, NY, NZ) float tensor in [0,1] or {0,1}
    Saves XY(mid-Z), XZ(mid-Y), YZ(mid-X).
    """
    import matplotlib.pyplot as plt

    v = vol.detach().cpu()
    nx, ny, nz = v.shape
    sx, sy, sz = nx // 2, ny // 2, nz // 2

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(v[:, :, sz].T, origin="lower")
    ax1.set_title(f"{title_prefix} (XY @ mid-Z)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(v[:, sy, :].T, origin="lower")
    ax2.set_title(f"{title_prefix} (XZ @ mid-Y)")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(v[sx, :, :].T, origin="lower")
    ax3.set_title(f"{title_prefix} (YZ @ mid-X)")
    ax3.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_hist_png(out_png: Path, data: torch.Tensor, title: str, xlabel: str) -> None:
    import matplotlib.pyplot as plt

    x = data.detach().cpu().flatten().numpy()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x, bins=40)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_scatter_png(out_png: Path, x: torch.Tensor, y: torch.Tensor, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt
    xv = x.detach().cpu().flatten().numpy()
    yv = y.detach().cpu().flatten().numpy()
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xv, yv, s=6, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_heightmap_png(out_png: Path, z_top: torch.Tensor, title: str) -> None:
    import matplotlib.pyplot as plt
    z = z_top.detach().cpu().numpy()
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(z.T, origin="lower")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# -----------------------------
# Diagnostics: Overhang
# -----------------------------
def central_diff_3d(vol: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v = vol
    gx = torch.zeros_like(v)
    gy = torch.zeros_like(v)
    gz = torch.zeros_like(v)

    gx[1:-1, :, :] = 0.5 * (v[2:, :, :] - v[:-2, :, :])
    gy[:, 1:-1, :] = 0.5 * (v[:, 2:, :] - v[:, :-2, :])
    gz[:, :, 1:-1] = 0.5 * (v[:, :, 2:] - v[:, :, :-2])

    return gx, gy, gz


def overhang_mask_from_theta(theta_phys: torch.Tensor, design_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Overhang convention locked:
      - compute surface normals from grad(theta_phys)
      - angle_from_horizontal = arccos(|n_z|)
      - violation if angle_from_horizontal < 45 deg (i.e., |n_z| > cos45)

    Evaluate voxels that are:
      - in design region
      - solid-ish
      - on/near surface (grad magnitude > threshold)
    """
    rho = theta_phys.detach().float().clamp(0.0, 1.0)[0]  # (NX,NY,NZ)

    gx, gy, gz = central_diff_3d(rho)
    gmag = torch.sqrt(gx * gx + gy * gy + gz * gz)

    nz = torch.zeros_like(rho)
    m = gmag > SURFACE_GRAD_THRESH
    nz[m] = (gz[m] / gmag[m]).abs()

    eval_mask = design_mask[0] & (rho > SOLID_THRESH) & m

    cos_th = math.cos(math.radians(OVERHANG_DEG_FROM_HORIZONTAL))
    violation = eval_mask & (nz > cos_th)

    eval_count = float(eval_mask.sum().item())
    vio_count = float(violation.sum().item())
    frac = (vio_count / eval_count) if eval_count > 0 else 0.0

    stats = {
        "overhang_deg_from_horizontal": float(OVERHANG_DEG_FROM_HORIZONTAL),
        "solid_thresh": float(SOLID_THRESH),
        "eval_surface_voxels": eval_count,
        "violation_surface_voxels": vio_count,
        "violation_fraction_on_surface": float(frac),
    }
    return violation.float().unsqueeze(0), stats


# -----------------------------
# Diagnostics: Floating islands
# -----------------------------
def islands_mask_from_theta(theta_phys: torch.Tensor, design_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Connected components from base plate (Z=0 layer).
    Any solid voxel NOT connected to base is a floating island.
    Connectivity: 6-neighborhood.
    """
    rho = theta_phys.detach().float().clamp(0.0, 1.0)[0]  # (NX,NY,NZ)
    solid = (rho > SOLID_THRESH) & design_mask[0]

    nx, ny, nz = solid.shape
    visited = torch.zeros_like(solid, dtype=torch.bool)
    q = deque()

    z0 = 0
    for ix in range(nx):
        for iy in range(ny):
            if solid[ix, iy, z0]:
                visited[ix, iy, z0] = True
                q.append((ix, iy, z0))

    neigh = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    while q:
        x, y, z = q.popleft()
        for dx, dy, dz in neigh:
            xx, yy, zz = x + dx, y + dy, z + dz
            if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                if solid[xx, yy, zz] and (not visited[xx, yy, zz]):
                    visited[xx, yy, zz] = True
                    q.append((xx, yy, zz))

    islands = solid & (~visited)

    solid_count = float(solid.sum().item())
    islands_count = float(islands.sum().item())
    frac = (islands_count / solid_count) if solid_count > 0 else 0.0

    stats = {
        "solid_thresh": float(SOLID_THRESH),
        "solid_voxels": solid_count,
        "island_voxels": islands_count,
        "island_fraction_of_solid": float(frac),
    }
    return islands.float().unsqueeze(0), stats


# -----------------------------
# Industrial Patch: Min-feature "vanish" proxy (binary erosion)
# -----------------------------
def binary_erosion_3d(x: torch.Tensor, passes: int = 1) -> torch.Tensor:
    """
    x: bool tensor (NX,NY,NZ). 1-pass erosion with 3x3x3 kernel.
    Erosion here = keep voxel only if all neighbors in 3x3x3 are solid.
    Implemented via conv3d: count neighbors == 27.
    """
    if passes <= 0:
        return x

    xi = x.float()[None, None, ...]  # (1,1,NX,NY,NZ)
    k = torch.ones((1, 1, 3, 3, 3), dtype=xi.dtype, device=xi.device)
    for _ in range(passes):
        y = F.conv3d(xi, k, padding=1)
        xi = (y >= 27.0).float()
    return (xi[0, 0] > 0.5)


def minfeature_mask_from_theta(theta_phys: torch.Tensor, design_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    rho = theta_phys.detach().float().clamp(0.0, 1.0)[0]
    solid = (rho > SOLID_THRESH) & design_mask[0]

    eroded = binary_erosion_3d(solid, passes=MINFEATURE_EROSION_PASSES)
    thin = solid & (~eroded)

    solid_count = float(solid.sum().item())
    thin_count = float(thin.sum().item())
    frac = (thin_count / solid_count) if solid_count > 0 else 0.0

    stats = {
        "solid_thresh": float(SOLID_THRESH),
        "erosion_passes": int(MINFEATURE_EROSION_PASSES),
        "kernel": "3x3x3",
        "solid_voxels": solid_count,
        "thin_voxels": thin_count,
        "thin_fraction_of_solid": float(frac),
    }
    return thin.float().unsqueeze(0), stats


# -----------------------------
# Industrial Patch: Recoater risk via Top-Z heightmap (4-neighbor steps)
# -----------------------------
def recoater_from_heightmap(
    theta_phys: torch.Tensor,
    design_mask: torch.Tensor,
    step_vox: int,
    voxel_size_mm: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    For each (x,y), z_top = max z where solid.
    Compute 4-neighbor step: max(|z_top - neighbor|).
    Risk if step >= step_vox.
    Returns: z_top (NX,NY), risk_xy (NX,NY), stats
    """
    rho = theta_phys.detach().float().clamp(0.0, 1.0)[0]
    solid = (rho > SOLID_THRESH) & design_mask[0]
    nx, ny, nz = solid.shape

    z_top = torch.full((nx, ny), -1, dtype=torch.int64)
    for ix in range(nx):
        for iy in range(ny):
            zz = torch.where(solid[ix, iy, :])[0]
            if zz.numel() > 0:
                z_top[ix, iy] = int(zz.max().item())

    zt = z_top.clone()
    zt[zt < 0] = 0

    step = torch.zeros((nx, ny), dtype=torch.int64)
    step[1:, :] = torch.maximum(step[1:, :], (zt[1:, :] - zt[:-1, :]).abs())
    step[:-1, :] = torch.maximum(step[:-1, :], (zt[:-1, :] - zt[1:, :]).abs())
    step[:, 1:] = torch.maximum(step[:, 1:], (zt[:, 1:] - zt[:, :-1]).abs())
    step[:, :-1] = torch.maximum(step[:, :-1], (zt[:, :-1] - zt[:, 1:]).abs())

    risk_xy = (step >= int(step_vox))

    num_xy = float(nx * ny)
    risk_xy_count = float(risk_xy.sum().item())
    stats = {
        "solid_thresh": float(SOLID_THRESH),
        "recoater_step_vox": int(step_vox),
        "recoater_step_mm": float(step_vox) * float(voxel_size_mm),
        "risk_xy_cells": risk_xy_count,
        "total_xy_cells": num_xy,
        "risk_fraction_xy": float(risk_xy_count / num_xy) if num_xy > 0 else 0.0,
    }

    return zt, risk_xy.float(), stats


# -----------------------------
# Industrial Patch: Thermal clustering (local average density)
# -----------------------------
def thermal_cluster_from_theta(
    theta_phys: torch.Tensor,
    design_mask: torch.Tensor,
    kernel_xyz: Tuple[int, int, int],
    thresh: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute local average density via 3D box blur over rho.
    Mark "cluster" where local_avg > thresh AND voxel is solid.
    """
    rho = theta_phys.detach().float().clamp(0.0, 1.0)[0]  # (NX,NY,NZ)
    solid = (rho > SOLID_THRESH) & design_mask[0]

    kx, ky, kz = kernel_xyz
    assert kx % 2 == 1 and ky % 2 == 1 and kz % 2 == 1, "Kernel must be odd in all dims."

    x = rho[None, None, ...]  # (1,1,NX,NY,NZ)
    kernel = torch.ones((1, 1, kx, ky, kz), dtype=x.dtype, device=x.device) / float(kx * ky * kz)
    pad = (kz//2, kz//2, ky//2, ky//2, kx//2, kx//2)  # F.pad order
    xp = F.pad(x, pad, mode="replicate")
    local_avg = F.conv3d(xp, kernel)[0, 0]  # (NX,NY,NZ)

    cluster_mask = (local_avg > float(thresh)) & solid

    solid_count = float(solid.sum().item())
    cluster_count = float(cluster_mask.sum().item())
    frac = (cluster_count / solid_count) if solid_count > 0 else 0.0

    stats = {
        "solid_thresh": float(SOLID_THRESH),
        "kernel_vox": [int(kx), int(ky), int(kz)],
        "cluster_thresh": float(thresh),
        "solid_voxels": solid_count,
        "cluster_voxels": cluster_count,
        "cluster_fraction_of_solid": float(frac),
    }
    return local_avg.unsqueeze(0), cluster_mask.float().unsqueeze(0), stats


# -----------------------------
# VED diagnostics
# -----------------------------
def compute_maps_from_pv(theta_raw: torch.Tensor, pv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns P_map (W) and v_map (mm/s) aligned to voxel grid.
    """
    theta = theta_raw
    pv = pv.flatten().float()
    K = pv.numel() // 2
    P_raw = pv[:K]
    v_raw = pv[K:]

    P_tile = map_tanh_to_range(P_raw, P_MIN, P_MAX)
    v_tile = map_tanh_to_range(v_raw, V_MIN, V_MAX)

    _, NX, NY, NZ = theta.shape
    tile_indices = build_tile_indices(NX, NY, NZ)

    P_map = P_tile[tile_indices]
    v_map = v_tile[tile_indices]
    return P_map, v_map


def compute_ved(P_map: torch.Tensor, v_map: torch.Tensor, h_mm: float, t_mm: float) -> torch.Tensor:
    denom = v_map * h_mm * t_mm
    return P_map / torch.clamp(denom, min=1e-12)


# -----------------------------
# Gate A: baseline parity check
# -----------------------------
def gate_a_recompute_theta_phys(beta_req: float, theta_raw: torch.Tensor, smoke) -> torch.Tensor:
    theta_clean = theta_raw.clamp(0.0, 1.0)
    design_mask = make_design_mask(theta_clean)

    eta, beta_eff, feasible, vf_lo, vf_hi = smoke.solve_eta_with_beta(
        theta_clean.detach(), float(beta_req), float(smoke.TARGET_VF), design_mask
    )
    theta_phys = smoke.projection(theta_clean, float(beta_eff), float(eta))
    return theta_phys


# -----------------------------
# Safe torch.load helper (quiet the warning when available)
# -----------------------------
def torch_load_cpu(path: Path) -> torch.Tensor:
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)  # type: ignore
    except TypeError:
        obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, torch.Tensor):
        raise RuntimeError(f"{path.name} did not contain a torch.Tensor")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 2 diagnostics: overhang, islands, VED + Industrial Patch (minfeature, recoater, clustering) + manifest."
    )
    ap.add_argument("--baseline", type=str, required=True, help="Path to locked Phase-1 baseline folder.")
    ap.add_argument("--out", type=str, required=True, help="Output dir for Phase-2 diagnostics.")
    ap.add_argument("--h_mm", type=float, default=ASSUMED_HATCH_MM, help="ASSUMED hatch spacing in mm (default 0.1).")
    ap.add_argument("--t_mm", type=float, default=ASSUMED_LAYER_MM, help="ASSUMED layer thickness in mm (default 0.04).")
    ap.add_argument("--beta_bin", type=float, default=64.0, help="Beta used for binarized theta_phys (default 64).")
    ap.add_argument("--voxel_size_mm", type=float, default=VOXEL_SIZE_MM, help="Voxel size in mm (placeholder unless confirmed).")
    ap.add_argument("--recoater_step_vox", type=int, default=RECOATER_STEP_VOX, help="Recoater cliff threshold in voxels.")
    ap.add_argument("--cluster_thresh", type=float, default=THERMAL_CLUSTER_THRESH, help="Thermal clustering density threshold.")
    args = ap.parse_args()

    voxel_size_mm = float(args.voxel_size_mm)

    baseline_dir = Path(args.baseline)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load Phase-1 artifacts
    meta_path = baseline_dir / "baseline_metadata.json"
    theta_raw_path = baseline_dir / "theta_raw.pt"
    theta_phys_bin_path = baseline_dir / f"theta_phys_beta{int(args.beta_bin)}.pt"

    if not meta_path.exists():
        raise RuntimeError(f"Missing baseline_metadata.json in {baseline_dir}")
    if not theta_raw_path.exists():
        raise RuntimeError(f"Missing theta_raw.pt in {baseline_dir}")
    if not theta_phys_bin_path.exists():
        raise RuntimeError(f"Missing {theta_phys_bin_path.name} in {baseline_dir}")

    meta = json.loads(meta_path.read_text())
    engine_sha = meta.get("engine_sha256", "UNKNOWN")

    baseline_hashes = fingerprint_dir_sha256(
        baseline_dir,
        relpaths=[
            Path("baseline_metadata.json"),
            Path("theta_raw.pt"),
            Path(f"theta_phys_beta{int(args.beta_bin)}.pt"),
            Path("pv_hist.png"),
            Path("vf_trace.png"),
            Path("theta_phys_beta64_slices.png"),
            Path("run.log"),
        ],
    )

    # Load tensors (safe)
    theta_raw = torch_load_cpu(theta_raw_path).float()
    theta_phys_bin = torch_load_cpu(theta_phys_bin_path).float()

    # pv: prefer baseline folder if present; fallback to scripts
    pv_baseline_path = baseline_dir / "pv_raw.pt"
    pv_latest_path = SCRIPTS / "pv_latest.pt"
    if pv_baseline_path.exists():
        pv = torch_load_cpu(pv_baseline_path).float()
    else:
        if not pv_latest_path.exists():
            raise RuntimeError(f"Missing pv_latest.pt at {pv_latest_path} and no pv_raw.pt in baseline.")
        pv = torch_load_cpu(pv_latest_path).float()

    smoke = import_smoke_module(SMOKE_PATH)

    # Gate A
    theta_phys_re = gate_a_recompute_theta_phys(args.beta_bin, theta_raw, smoke)
    bit_equal = torch.equal(theta_phys_re, theta_phys_bin)
    max_abs_diff = float((theta_phys_re - theta_phys_bin).abs().max().item())
    gate_a_pass = bit_equal or (max_abs_diff <= 1e-12)

    design_mask = make_design_mask(theta_raw.clamp(0.0, 1.0))

    # --- Core diagnostics ---
    overhang_mask, overhang_stats = overhang_mask_from_theta(theta_phys_bin, design_mask)
    torch.save(overhang_mask, str(out_dir / "overhang_mask.pt"))
    save_slices_png(out_dir / "overhang_slices.png", overhang_mask[0], "Overhang violation mask (1=bad)")

    islands_mask, islands_stats = islands_mask_from_theta(theta_phys_bin, design_mask)
    torch.save(islands_mask, str(out_dir / "islands_mask.pt"))
    save_slices_png(out_dir / "islands_slices.png", islands_mask[0], "Floating islands mask (1=bad)")

    P_map, v_map = compute_maps_from_pv(theta_raw, pv)
    ved = compute_ved(P_map, v_map, args.h_mm, args.t_mm)
    torch.save(ved, str(out_dir / "ved_map_J_per_mm3.pt"))
    save_slices_png(out_dir / "ved_slices.png", ved, "VED map (J/mm^3)")
    save_hist_png(out_dir / "ved_hist.png", ved, "VED distribution", "VED (J/mm^3)")
    save_hist_png(out_dir / "P_hist.png", P_map, "P_map distribution", "Power P (W)")
    save_hist_png(out_dir / "v_hist.png", v_map, "v_map distribution", "Speed v (mm/s)")

    ved_min = float(ved.min().item())
    ved_max = float(ved.max().item())
    ved_mean = float(ved.mean().item())
    ved_p = percentiles(ved)

    # --- Industrial Patch diagnostics ---
    minfeature_mask, minfeature_stats = minfeature_mask_from_theta(theta_phys_bin, design_mask)
    torch.save(minfeature_mask, str(out_dir / "minfeature_mask.pt"))
    save_slices_png(out_dir / "minfeature_slices.png", minfeature_mask[0], "Min-feature (vanish) mask (1=thin)")

    z_top, recoater_risk_xy, recoater_stats = recoater_from_heightmap(
        theta_phys_bin, design_mask, args.recoater_step_vox, voxel_size_mm
    )
    torch.save(recoater_risk_xy, str(out_dir / "recoater_risk_mask.pt"))
    save_heightmap_png(out_dir / "recoater_heightmap.png", z_top.float(), "Top-Z heightmap (voxels)")
    save_heightmap_png(out_dir / "recoater_risk.png", recoater_risk_xy, f"Recoater risk (step >= {args.recoater_step_vox} vox)")

    thermal_avg, thermal_cluster_mask, thermal_cluster_stats = thermal_cluster_from_theta(
        theta_phys_bin, design_mask, THERMAL_CLUSTER_KERNEL, args.cluster_thresh
    )
    torch.save(thermal_avg, str(out_dir / "thermal_cluster_map.pt"))
    torch.save(thermal_cluster_mask, str(out_dir / "thermal_cluster_mask.pt"))
    save_slices_png(out_dir / "thermal_cluster_slices.png", thermal_avg[0], "Thermal clustering (local avg rho)")
    save_slices_png(out_dir / "thermal_cluster_mask_slices.png", thermal_cluster_mask[0], "Thermal cluster mask (1=hot chunk)")

    # VED vs clustering scatter: evaluate only solid voxels in design
    rho = theta_phys_bin.detach().float().clamp(0.0, 1.0)[0]
    solid = (rho > SOLID_THRESH) & design_mask[0]
    ved_s = ved[solid]
    clu_s = thermal_avg[0][solid]
    if ved_s.numel() > 0:
        save_scatter_png(
            out_dir / "ved_vs_cluster_scatter.png",
            ved_s,
            clu_s,
            "VED vs Thermal Clustering (solid voxels)",
            "VED (J/mm^3)",
            "Local avg rho",
        )

    cluster_p = percentiles(thermal_avg[0][solid] if solid.any() else thermal_avg[0])

    # Gate B: sha256 of produced outputs
    produced_files = [
        out_dir / "overhang_mask.pt",
        out_dir / "overhang_slices.png",
        out_dir / "islands_mask.pt",
        out_dir / "islands_slices.png",
        out_dir / "ved_map_J_per_mm3.pt",
        out_dir / "ved_slices.png",
        out_dir / "ved_hist.png",
        out_dir / "P_hist.png",
        out_dir / "v_hist.png",
        out_dir / "minfeature_mask.pt",
        out_dir / "minfeature_slices.png",
        out_dir / "recoater_risk_mask.pt",
        out_dir / "recoater_heightmap.png",
        out_dir / "recoater_risk.png",
        out_dir / "thermal_cluster_map.pt",
        out_dir / "thermal_cluster_slices.png",
        out_dir / "thermal_cluster_mask.pt",
        out_dir / "thermal_cluster_mask_slices.png",
        out_dir / "ved_vs_cluster_scatter.png",
    ]
    file_hashes: Dict[str, str] = {}
    for fp in produced_files:
        if fp.exists() and fp.is_file():
            file_hashes[fp.name] = sha256_file(fp)
        else:
            file_hashes[fp.name] = "MISSING"

    manifest = {
        "phase": "Phase-2 Diagnostics",
        "created_unix_s": time.time(),
        "baseline_dir": str(baseline_dir),
        "engine_sha256": engine_sha,
        "baseline_folder_sha256": baseline_hashes,

        "gate_a": {
            "beta_bin": float(args.beta_bin),
            "bit_equal": bool(bit_equal),
            "max_abs_diff": float(max_abs_diff),
            "pass": bool(gate_a_pass),
            "note": "Pass if bit-for-bit identical OR max_abs_diff <= 1e-12.",
        },

        "conventions_locked": {
            "overhang_angle_measured_from": "horizontal_build_plate",
            "overhang_threshold_deg": float(OVERHANG_DEG_FROM_HORIZONTAL),
            "violation_condition": "angle_from_horizontal < threshold (ceiling-like)",
            "solid_threshold": float(SOLID_THRESH),
            "islands_definition": "solid voxels not connected to Z=0 (6-connectivity)",
        },

        "DIAGNOSTIC_THRESHOLDS": {
            "voxel_size_mm": float(voxel_size_mm),
            "minfeature_erosion_passes": int(MINFEATURE_EROSION_PASSES),
            "recoater_step_vox": int(args.recoater_step_vox),
            "recoater_step_mm": float(args.recoater_step_vox) * float(voxel_size_mm),
            "thermal_cluster_kernel_vox": list(THERMAL_CLUSTER_KERNEL),
            "thermal_cluster_thresh": float(args.cluster_thresh),
        },

        "ASSUMED_PROCESS_CONSTANTS": {
            "hatch_spacing_mm": float(args.h_mm),
            "layer_thickness_mm": float(args.t_mm),
            "ved_formula": "VED = P / (v * h * t) (J/mm^3)",
            "note": "These constants are placeholders unless confirmed per machine/material process window.",
        },

        "ved_summary": {
            "ved_min": ved_min,
            "ved_max": ved_max,
            "ved_mean": ved_mean,
            **ved_p,
        },

        "overhang_stats": overhang_stats,
        "islands_stats": islands_stats,
        "minfeature_stats": minfeature_stats,
        "recoater_stats": recoater_stats,
        "thermal_cluster_stats": {
            **thermal_cluster_stats,
            **cluster_p,
        },

        "env": env_snapshot(),
        "git_commit": git_commit(REPO),

        "artifacts_sha256": file_hashes,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    runtime_s = float(time.time() - t0)

    print("\n[SUCCESS] Phase-2 diagnostics written to:")
    print(f"  {out_dir}")

    print("\nGate A (baseline parity):")
    print(f"  bit_equal={bit_equal}  max_abs_diff={max_abs_diff:.3e}  PASS={gate_a_pass}")
    if not gate_a_pass:
        print("  [FAIL] Gate A did not pass. Do NOT proceed until parity is restored.")

    print("\nOverhang (surface-based) red-zone:")
    print(f"  violation_fraction_on_surface={overhang_stats['violation_fraction_on_surface']:.4f}  (threshold={OVERHANG_DEG_FROM_HORIZONTAL} deg from horizontal)")

    print("\nFloating islands:")
    print(f"  island_fraction_of_solid={islands_stats['island_fraction_of_solid']:.6f}  (goal: 0.0)")

    print("\nMin-feature (vanish proxy):")
    print(f"  thin_fraction_of_solid={minfeature_stats['thin_fraction_of_solid']:.6f}  (erosion_passes={MINFEATURE_EROSION_PASSES})")

    print("\nRecoater risk (Top-Z step):")
    print(f"  risk_fraction_xy={recoater_stats['risk_fraction_xy']:.6f}  (step_vox={args.recoater_step_vox}, step_mm={recoater_stats['recoater_step_mm']:.3f})")

    print("\nThermal clustering (heat sink proxy):")
    print(f"  cluster_fraction_of_solid={thermal_cluster_stats['cluster_fraction_of_solid']:.6f}  (kernel={THERMAL_CLUSTER_KERNEL}, thresh={args.cluster_thresh})")

    print("\nVED summary (J/mm^3):")
    print(f"  min={ved_min:.2f}  mean={ved_mean:.2f}  max={ved_max:.2f}  (h={args.h_mm}mm, t={args.t_mm}mm)")
    print(f"  p5={ved_p['p5']:.2f}  p50={ved_p['p50']:.2f}  p95={ved_p['p95']:.2f}")

    print("\nOutputs:")
    for fp in produced_files + [out_dir / "manifest.json"]:
        print(f"  - {fp}")

    print(f"\nruntime_s={runtime_s:.2f}")


if __name__ == "__main__":
    main()
