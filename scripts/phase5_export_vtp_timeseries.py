import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from skimage import measure
except ImportError as e:
    raise SystemExit("Missing skimage. Install: pip install scikit-image") from e

try:
    import pyvista as pv
except ImportError as e:
    raise SystemExit("Missing pyvista. Install: pip install pyvista") from e


def load_pt(path: Path) -> torch.Tensor:
    t = torch.load(str(path), map_location="cpu")
    if isinstance(t, dict):
        # common pattern: {"theta": tensor} or similar
        # try best-effort
        for k in ["theta", "theta_phys", "data", "tensor"]:
            if k in t and torch.is_tensor(t[k]):
                return t[k]
        raise RuntimeError(f"{path} is a dict but no known tensor key found.")
    if not torch.is_tensor(t):
        raise RuntimeError(f"{path} did not load as torch.Tensor.")
    return t


def to_numpy_grid(t: torch.Tensor) -> np.ndarray:
    # Accept (1,NX,NY,NZ) or (NX,NY,NZ)
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 3:
        raise RuntimeError(f"Expected 3D grid, got shape {tuple(t.shape)}")
    return t.detach().cpu().float().numpy()


def make_surface_from_voxels(phi: np.ndarray, level: float, voxel_size_mm: float) -> pv.PolyData:
    # marching cubes expects (Z,Y,X) or (N,N,N) - we’ll keep consistent via spacing
    # skimage returns verts in (z,y,x) index space
    verts, faces, normals, values = measure.marching_cubes(phi, level=level, spacing=(voxel_size_mm, voxel_size_mm, voxel_size_mm))
    # faces in skimage are (n,3), pyvista wants a flat array with leading "3" per face
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
    mesh = pv.PolyData(verts, faces_pv)
    mesh.compute_normals(inplace=True)
    return mesh


def sample_grid_to_surface_points(grid: np.ndarray, pts_xyz_mm: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    """
    Nearest-neighbor sample scalar grid (NX,NY,NZ) at mesh points (x,y,z in mm).
    """
    # pts are in mm; convert to index
    ijk = np.round(pts_xyz_mm / voxel_size_mm).astype(np.int64)
    # grid assumed in (z,y,x)?? we used spacing in marching cubes which outputs verts as (z,y,x) coordinates.
    # Those verts are in (z,y,x) order already. So interpret verts columns as (z,y,x) in mm.
    # Convert mm -> index -> (z,y,x)
    z = np.clip(ijk[:, 0], 0, grid.shape[0] - 1)
    y = np.clip(ijk[:, 1], 0, grid.shape[1] - 1)
    x = np.clip(ijk[:, 2], 0, grid.shape[2] - 1)
    return grid[z, y, x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine_dir", required=True)
    ap.add_argument("--diag_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--voxel_size_mm", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    eng = Path(args.engine_dir)
    diag = Path(args.diag_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    theta12 = eng / "theta_phys_beta12.pt"
    theta64 = eng / "theta_phys_beta64.pt"

    ved_pt = diag / "ved_map_J_per_mm3.pt"
    overhang_pt = diag / "overhang_mask.pt"
    thermal_pt = diag / "thermal_cluster_map.pt"

    theta12_np = to_numpy_grid(load_pt(theta12))
    theta64_np = to_numpy_grid(load_pt(theta64))

    ved_np = to_numpy_grid(load_pt(ved_pt))
    overhang_np = to_numpy_grid(load_pt(overhang_pt))
    thermal_np = to_numpy_grid(load_pt(thermal_pt))

    # Marching cubes on theta fields
    m12 = make_surface_from_voxels(theta12_np, level=args.threshold, voxel_size_mm=args.voxel_size_mm)
    m64 = make_surface_from_voxels(theta64_np, level=args.threshold, voxel_size_mm=args.voxel_size_mm)

    # Attach scalars by sampling at surface points
    for tag, mesh in [("0000", m12), ("0001", m64)]:
        pts = mesh.points  # (N,3) in mm, but interpreted as (z,y,x) order
        mesh["VED_J_per_mm3"] = sample_grid_to_surface_points(ved_np, pts, args.voxel_size_mm).astype(np.float32)
        mesh["overhang_mask"] = sample_grid_to_surface_points(overhang_np, pts, args.voxel_size_mm).astype(np.float32)
        mesh["thermal_cluster_map"] = sample_grid_to_surface_points(thermal_np, pts, args.voxel_size_mm).astype(np.float32)

        vtp = out / f"cantilever_{tag}.vtp"
        mesh.save(str(vtp))

    stats = {
        "engine_dir": str(eng),
        "diag_dir": str(diag),
        "out_dir": str(out),
        "frames": ["cantilever_0000.vtp", "cantilever_0001.vtp"],
        "threshold": args.threshold,
        "voxel_size_mm": args.voxel_size_mm,
        "scalars": ["VED_J_per_mm3", "overhang_mask", "thermal_cluster_map"],
    }
    json.dump(stats, open(out / "export_stats.json", "w"), indent=2)
    print("[OK] Wrote VTP frames:")
    print("  ", out / "cantilever_0000.vtp")
    print("  ", out / "cantilever_0001.vtp")
    print("[OK] Wrote:", out / "export_stats.json")


if __name__ == "__main__":
    main()
