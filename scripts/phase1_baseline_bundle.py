#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# -----------------------------
# Repo paths
# -----------------------------
REPO = Path("/workspaces/lpbf_project")
SCRIPTS = REPO / "scripts"

SMOKE_PATH = SCRIPTS / "smoke_joint_opt_fast.py"
THETA_PATH = SCRIPTS / "theta_latest.pt"
PV_PATH = SCRIPTS / "pv_latest.pt"

OUT_DIR = REPO / "artifacts" / "phase1_baseline_in718"

# IN718 bounds (must match smoke_joint_opt_fast.py)
P_MIN, P_MAX = 100.0, 600.0
V_MIN, V_MAX = 400.0, 1100.0

# Baseline betas to store
BETA_STAGE = 12.0
BETA_BIN = 64.0

# If P/v gets too close to bounds, warn (fraction of range)
SAT_MARGIN_FRAC = 0.01  # 1% of range

# Regex to parse VF lines from smoke log
VF_RE = re.compile(r"\[VF\]\s+raw=(?P<raw>[0-9.]+)\s+phys=(?P<phys>[0-9.]+)")
ORTH_RE = re.compile(r"\[WD_PREV\].*last_orth=(?P<orth>[-+eE0-9.]+)")

# For repo-wide code hashing
EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    "artifacts", "data", "outputs", "wandb", ".mypy_cache",
}


# -----------------------------
# Helpers
# -----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_code_files(repo: Path) -> List[Path]:
    files: List[Path] = []
    for p in repo.rglob("*.py"):
        rel_parts = p.relative_to(repo).parts
        if any(part in EXCLUDE_DIRS for part in rel_parts):
            continue
        files.append(p)
    return sorted(files)


def sha256_manifest(file_hashes: Dict[str, str]) -> str:
    """
    A single digest for the whole engine: hash of the sorted (path, hash) pairs.
    Changes if ANY file hash changes.
    """
    h = hashlib.sha256()
    for k in sorted(file_hashes.keys()):
        h.update(k.encode("utf-8"))
        h.update(b"\0")
        h.update(file_hashes[k].encode("utf-8"))
        h.update(b"\n")
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
    # Works only if .git exists. If not, we’ll store UNKNOWN and rely on SHA256 fingerprints.
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), stderr=subprocess.STDOUT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def map_tanh_to_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    u = torch.tanh(x)  # [-1,1]
    return lo + (u + 1.0) * 0.5 * (hi - lo)


def monotonicity_check(lo: float, hi: float) -> Dict[str, object]:
    xs = torch.tensor([-6.0, -3.0, 0.0, 3.0, 6.0])
    ys_t = map_tanh_to_range(xs, lo, hi).detach().cpu()
    ys = [float(v) for v in ys_t.tolist()]
    mono = all(ys[i] < ys[i + 1] for i in range(len(ys) - 1))
    in_bounds = (min(ys) >= lo - 1e-9) and (max(ys) <= hi + 1e-9)
    return {
        "xs": [-6, -3, 0, 3, 6],
        "ys": ys,
        "is_strictly_increasing": bool(mono),
        "in_bounds": bool(in_bounds),
        "y_min": float(min(ys)),
        "y_max": float(max(ys)),
    }


def make_design_mask(theta: torch.Tensor) -> torch.Tensor:
    # Mirrors smoke_joint_opt_fast Omega_design:
    # Omega_design is -1 everywhere, with x=0 plane set to +1 (not design)
    mask = torch.ones_like(theta, dtype=torch.bool)
    mask[:, 0, :, :] = False
    return mask


def build_tile_indices(nx: int, ny: int, nz: int) -> torch.Tensor:
    tile_indices = torch.empty((nx, ny, nz), dtype=torch.int64)
    for ix in range(nx):
        tx = 0 if ix < nx // 2 else 1
        for iy in range(ny):
            ty = 0 if iy < ny // 2 else 1
            for iz in range(nz):
                tz = 0 if iz < nz // 2 else 1
                tile_indices[ix, iy, iz] = tx + 2 * ty + 4 * tz
    return tile_indices


def warn_saturation(name: str, arr: torch.Tensor, lo: float, hi: float) -> Optional[str]:
    # Warn if hugging bounds within margin
    span = hi - lo
    margin = SAT_MARGIN_FRAC * span
    a_min = float(arr.min().item())
    a_max = float(arr.max().item())
    if (a_min - lo) < margin or (hi - a_max) < margin:
        return f"{name} near bounds: min={a_min:.3f} max={a_max:.3f} bounds=[{lo},{hi}] margin={margin:.3f}"
    return None


def parse_vf_trace(log_path: Path) -> Tuple[List[float], List[float]]:
    raws, phys = [], []
    if not log_path.exists():
        return raws, phys
    for line in log_path.read_text(errors="ignore").splitlines():
        m = VF_RE.search(line)
        if m:
            raws.append(float(m.group("raw")))
            phys.append(float(m.group("phys")))
    return raws, phys


def parse_orth_trace(log_path: Path) -> List[float]:
    orth = []
    if not log_path.exists():
        return orth
    for line in log_path.read_text(errors="ignore").splitlines():
        m = ORTH_RE.search(line)
        if m:
            try:
                orth.append(float(m.group("orth")))
            except Exception:
                pass
    return orth


def save_visuals(
    out_dir: Path,
    theta_phys_beta64: torch.Tensor,
    P_map: torch.Tensor,
    v_map: torch.Tensor,
    vf_raw: List[float],
    vf_phys: List[float],
) -> None:
    # Create simple PNGs (no seaborn)
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- theta slices
    # theta_phys_beta64 is (1,NX,NY,NZ)
    rho = theta_phys_beta64[0].detach().cpu()
    nx, ny, nz = rho.shape
    sx, sy, sz = nx // 2, ny // 2, nz // 2

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rho[:, :, sz].T, origin="lower")
    ax1.set_title("theta_phys beta64 (XY @ mid-Z)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(rho[:, sy, :].T, origin="lower")
    ax2.set_title("theta_phys beta64 (XZ @ mid-Y)")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(rho[sx, :, :].T, origin="lower")
    ax3.set_title("theta_phys beta64 (YZ @ mid-X)")
    ax3.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "theta_phys_beta64_slices.png", dpi=160)
    plt.close(fig)

    # ---- P/v hist
    p = P_map.detach().cpu().flatten().numpy()
    v = v_map.detach().cpu().flatten().numpy()

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(p, bins=30)
    ax1.set_title("P_map histogram (W)")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(v, bins=30)
    ax2.set_title("v_map histogram (mm/s)")
    fig.tight_layout()
    fig.savefig(out_dir / "pv_hist.png", dpi=160)
    plt.close(fig)

    # ---- VF trace from log
    if len(vf_phys) > 0:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(vf_raw, label="VF raw")
        ax.plot(vf_phys, label="VF phys")
        ax.set_title("VF trace (parsed from run.log)")
        ax.set_xlabel("logged step")
        ax.set_ylabel("VF")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "vf_trace.png", dpi=160)
        plt.close(fig)


def import_smoke_module(smoke_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("smoke_joint_opt_fast", str(smoke_path))
    assert spec and spec.loader
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)  # type: ignore
    return smoke


# -----------------------------
# Streaming subprocess runner (same terminal + tee to log)
# -----------------------------
def run_smoke_stream(smoke_path: Path, repo: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] python {smoke_path} (streaming here; log -> {log_path})")

    # line-buffered text streaming
    p = subprocess.Popen(
        ["python", str(smoke_path)],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert p.stdout is not None
    with log_path.open("w", encoding="utf-8") as f:
        for line in p.stdout:
            # print to terminal
            print(line, end="")
            # write to log
            f.write(line)
        rc = p.wait()

    if rc != 0:
        raise RuntimeError(f"smoke_joint_opt_fast.py failed (exit code {rc}). See log: {log_path}")


# -----------------------------
# Main bundler logic
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 1 baseline bundler: run smoke, capture theta/pv, compute theta_phys, save metadata + visuals."
    )
    ap.add_argument("--run", action="store_true", help="Run smoke_joint_opt_fast.py first (recommended).")
    ap.add_argument("--no-run", action="store_true", help="Do NOT run smoke; just bundle existing theta_latest.pt/pv_latest.pt.")
    ap.add_argument("--seed", type=int, default=123, help="Seed recorded into metadata (engine should also set it).")
    ap.add_argument("--out", type=str, default=str(OUT_DIR), help="Output directory for baseline bundle.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.log"

    t0 = time.time()

    # Decide run mode
    if args.no_run and args.run:
        raise SystemExit("Use only one: --run OR --no-run")

    if not args.no_run:
        # default behavior: run unless --no-run
        run_smoke_stream(SMOKE_PATH, REPO, run_log)

    # Ensure outputs exist
    if not THETA_PATH.exists() or not PV_PATH.exists():
        raise RuntimeError(
            f"Missing outputs. Expected:\n  {THETA_PATH}\n  {PV_PATH}\n"
            f"Tip: run with --run to generate them."
        )

    # Load outputs (safe since we created them)
    theta = torch.load(str(THETA_PATH), map_location="cpu")
    if not isinstance(theta, torch.Tensor):
        raise RuntimeError("theta_latest.pt did not contain a torch.Tensor")
    theta = theta.float()

    pv = torch.load(str(PV_PATH), map_location="cpu")
    if not isinstance(pv, torch.Tensor):
        raise RuntimeError("pv_latest.pt did not contain a torch.Tensor")
    pv = pv.flatten().float()

    K = pv.numel() // 2
    if pv.numel() != 2 * K:
        raise RuntimeError(f"pv_latest.pt length must be even, got {pv.numel()}")

    P_raw = pv[:K]
    v_raw = pv[K:]

    # --- Monotonicity / bounds checks for knob mapping (HARD)
    mono_P = monotonicity_check(P_MIN, P_MAX)
    mono_v = monotonicity_check(V_MIN, V_MAX)
    if (not mono_P["is_strictly_increasing"]) or (not mono_P["in_bounds"]):
        raise RuntimeError(f"[ASSERT FAIL] P knob mapping not monotonic/bounded: {mono_P}")
    if (not mono_v["is_strictly_increasing"]) or (not mono_v["in_bounds"]):
        raise RuntimeError(f"[ASSERT FAIL] v knob mapping not monotonic/bounded: {mono_v}")

    # Decode tile P/v
    P_tile = map_tanh_to_range(P_raw, P_MIN, P_MAX)
    v_tile = map_tanh_to_range(v_raw, V_MIN, V_MAX)

    # Tile -> voxel maps
    _, NX, NY, NZ = theta.shape
    tile_indices = build_tile_indices(NX, NY, NZ)
    P_map = P_tile[tile_indices]
    v_map = v_tile[tile_indices]

    # Import smoke helpers to compute theta_phys consistently
    smoke = import_smoke_module(SMOKE_PATH)

    theta_clean = theta.clamp(0.0, 1.0)
    design_mask = make_design_mask(theta_clean)

    def theta_phys_for_beta(beta_req: float) -> Dict:
        eta, beta_eff, feasible, vf_lo, vf_hi = smoke.solve_eta_with_beta(
            theta_clean.detach(), float(beta_req), float(smoke.TARGET_VF), design_mask
        )
        theta_phys = smoke.projection(theta_clean, float(beta_eff), float(eta))

        vf_raw_local = float(theta_clean[design_mask].mean().item())
        vf_phys_local = float(theta_phys[design_mask].mean().item())
        frac_lo = float((theta_phys[design_mask] < 0.05).float().mean().item())
        frac_hi = float((theta_phys[design_mask] > 0.95).float().mean().item())

        return {
            "beta_req": float(beta_req),
            "beta_eff": float(beta_eff),
            "eta": float(eta),
            "feasible": bool(feasible),
            "vf_lo": float(vf_lo),
            "vf_hi": float(vf_hi),
            "vf_raw": float(vf_raw_local),
            "vf_phys": float(vf_phys_local),
            "theta_phys": theta_phys,
            "theta_phys_min": float(theta_phys.min().item()),
            "theta_phys_max": float(theta_phys.max().item()),
            "frac_lt_005": float(frac_lo),
            "frac_gt_095": float(frac_hi),
        }

    phys_stage = theta_phys_for_beta(BETA_STAGE)
    phys_bin = theta_phys_for_beta(BETA_BIN)

    # Hard assertions for Phase 1 VF
    vf_err_stage = abs(phys_stage["vf_phys"] - float(smoke.TARGET_VF))
    vf_err_bin = abs(phys_bin["vf_phys"] - float(smoke.TARGET_VF))
    if vf_err_stage > 1e-5:
        raise RuntimeError(f"[ASSERT FAIL] VF_phys(beta={BETA_STAGE}) off by {vf_err_stage}")
    if vf_err_bin > 1e-5:
        raise RuntimeError(f"[ASSERT FAIL] VF_phys(beta={BETA_BIN}) off by {vf_err_bin}")

    # P/v saturation warnings
    sat_warnings = []
    w = warn_saturation("P_tile", P_tile, P_MIN, P_MAX)
    if w:
        sat_warnings.append(w)
    w = warn_saturation("v_tile", v_tile, V_MIN, V_MAX)
    if w:
        sat_warnings.append(w)

    # Save tensors
    torch.save(theta, str(out_dir / "theta_raw.pt"))
    torch.save(phys_stage["theta_phys"], str(out_dir / f"theta_phys_beta{int(BETA_STAGE)}.pt"))
    torch.save(phys_bin["theta_phys"], str(out_dir / f"theta_phys_beta{int(BETA_BIN)}.pt"))

    torch.save(P_raw, str(out_dir / "P_raw.pt"))
    torch.save(v_raw, str(out_dir / "v_raw.pt"))
    torch.save(P_tile, str(out_dir / "P_tile_W.pt"))
    torch.save(v_tile, str(out_dir / "v_tile_mms.pt"))
    torch.save(P_map, str(out_dir / "P_map_W.pt"))
    torch.save(v_map, str(out_dir / "v_map_mms.pt"))

    # --- Repo-wide code fingerprints (works even with no .git)
    code_fingerprints: Dict[str, str] = {}
    for fp in iter_code_files(REPO):
        code_fingerprints[str(fp.relative_to(REPO))] = sha256_file(fp)

    engine_sha256 = sha256_manifest(code_fingerprints)

    # VF trace + orth trace from log (if available)
    vf_raw_trace, vf_phys_trace = parse_vf_trace(run_log)
    orth_trace = parse_orth_trace(run_log)

    # --- Orthogonality invariant (HARD)
    orth_max_abs = None
    if orth_trace:
        orth_max_abs = max(abs(x) for x in orth_trace)

    # Numerical tolerance: hooks should be "near-orthogonal".
    # 1e-8 is still extremely strict but avoids false failures from float noise.
    ORTH_TOL = 1e-8
    if orth_max_abs > ORTH_TOL:
        raise RuntimeError(
            f"[ASSERT FAIL] hook orthogonality too high: max|orth|={orth_max_abs:.3e} (tol={ORTH_TOL:.1e})"
        )


    # Metadata
    runtime_s = float(time.time() - t0)
    meta = {
        "git_commit": git_commit(REPO),
        "code_sha256": code_fingerprints,
        "engine_sha256": engine_sha256,
        "num_files_hashed": len(code_fingerprints),
        "env": env_snapshot(),
        "knob_monotonicity": {"P": mono_P, "v": mono_v},
        "material": "IN718",
        "seed": int(args.seed),
        "target_vf": float(smoke.TARGET_VF),
        "theta_raw_min": float(theta.min().item()),
        "theta_raw_max": float(theta.max().item()),
        "pv_numel": int(pv.numel()),
        "K": int(K),
        "P_bounds_W": [P_MIN, P_MAX],
        "v_bounds_mms": [V_MIN, V_MAX],
        "runtime_s": runtime_s,
        "theta_phys_stage_summary": {k: v for k, v in phys_stage.items() if k != "theta_phys"},
        "theta_phys_bin_summary": {k: v for k, v in phys_bin.items() if k != "theta_phys"},
        "vf_trace_len": len(vf_phys_trace),
        "orth_trace_len": len(orth_trace),
        "orth_max_abs": orth_max_abs,
        "saturation_warnings": sat_warnings,
        "log_path": str(run_log),
        "notes": "Phase 1 Step 1 baseline bundle (raw + phys + pv maps + log + visuals + env + repo-wide code fingerprints + monotonicity + orth assert).",
    }
    (out_dir / "baseline_metadata.json").write_text(json.dumps(meta, indent=2))

    # Visuals
    save_visuals(
        out_dir=out_dir,
        theta_phys_beta64=phys_bin["theta_phys"],
        P_map=P_map,
        v_map=v_map,
        vf_raw=vf_raw_trace,
        vf_phys=vf_phys_trace,
    )

    # Summary to terminal
    print("\n[SUCCESS] Phase 1 baseline bundle written to:")
    print(f"  {out_dir}")
    print("\nKey checks (HARD):")
    print(f"  VF_phys(beta12)={phys_stage['vf_phys']:.6f}  beta_eff={phys_stage['beta_eff']:.2f}  eta={phys_stage['eta']:.6f}")
    print(f"  VF_phys(beta64)={phys_bin['vf_phys']:.6f}  beta_eff={phys_bin['beta_eff']:.2f}  eta={phys_bin['eta']:.6f}")
    print(f"  binarization(beta64): frac<0.05={phys_bin['frac_lt_005']:.3f} frac>0.95={phys_bin['frac_gt_095']:.3f}")
    print(f"  orth_max_abs={orth_max_abs if orth_max_abs is not None else 'N/A'}")
    print(f"  num_files_hashed={len(code_fingerprints)}")
    print(f"  engine_sha256={engine_sha256[:16]}...")  # short display
    print(f"  runtime_s={runtime_s:.2f}")

    if sat_warnings:
        print("\n[WARN] Saturation checks:")
        for w in sat_warnings:
            print(f"  - {w}")

    print("\nOutputs:")
    print(f"  log:      {run_log}")
    print(f"  metadata: {out_dir / 'baseline_metadata.json'}")
    print(f"  visuals:  {out_dir / 'theta_phys_beta64_slices.png'}")
    print(f"           {out_dir / 'pv_hist.png'}")
    print(f"           {out_dir / 'vf_trace.png'}")


if __name__ == "__main__":
    main()
