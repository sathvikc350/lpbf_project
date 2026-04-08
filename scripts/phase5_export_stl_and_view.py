#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

import trimesh


def load_theta(theta_pt: str) -> torch.Tensor:
    t = torch.load(theta_pt, map_location="cpu")
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {theta_pt}, got {type(t)}")
    return t


def make_mesh_from_voxels(theta: torch.Tensor, voxel_size: float, level: float) -> trimesh.Trimesh:
    """
    Convert a (1,NX,NY,NZ) or (NX,NY,NZ) tensor into a surface mesh via marching cubes.
    Works across trimesh versions (no remove_degenerate_faces()).
    """
    arr = theta.detach().cpu().numpy()

    # Accept (1,NX,NY,NZ) or (NX,NY,NZ)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected theta to be 3D (or 4D with leading 1). Got shape={arr.shape}")

    # Binarize at threshold
    vol = (arr > float(level))

    # VoxelGrid expects boolean 3D volume.
    # Apply scaling transform so the mesh comes out in mm.
    T = trimesh.transformations.scale_matrix(float(voxel_size))
    vg = trimesh.voxel.VoxelGrid(vol, transform=T)

    # Marching cubes mesh
    mesh = vg.marching_cubes

    # Cleanup (version-safe)
    # 1) drop degenerate faces
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass

    # 2) merge duplicates / remove unreferenced
    try:
        mesh.merge_vertices()
    except Exception:
        pass

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # 3) final processing/validation
    try:
        mesh.process(validate=True)
    except Exception:
        pass

    return mesh


def save_preview_png(mesh: trimesh.Trimesh, out_png: Path) -> None:
    """
    Saves a simple rendered preview PNG (offscreen). If rendering isn't available,
    we just skip without failing the STL export.
    """
    try:
        # Prefer scene.save_image (works when pyglet/OSMesa/EGL is available)
        scene = mesh.scene()
        png_bytes = scene.save_image(resolution=(900, 700))
        if png_bytes is None:
            return
        out_png.write_bytes(png_bytes)
    except Exception:
        # Don't hard fail if headless rendering isn't set up
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta_pt", required=True, help="Path to theta_phys_beta*.pt")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--voxel_size", type=float, required=True, help="Voxel size in mm (e.g., 0.25)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold (default 0.5)")
    ap.add_argument("--show", action="store_true", help="Try to open an interactive viewer (may not work headless)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    theta = load_theta(args.theta_pt)
    mesh = make_mesh_from_voxels(theta, voxel_size=args.voxel_size, level=args.threshold)

    # Export STL
    out_stl = out_dir / "theta_mesh.stl"
    mesh.export(out_stl)

    # Save quick stats
    stats = {
        "theta_pt": str(Path(args.theta_pt).resolve()),
        "voxel_size_mm": float(args.voxel_size),
        "threshold": float(args.threshold),
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "is_watertight": bool(getattr(mesh, "is_watertight", False)),
        "bounds_mm": mesh.bounds.tolist() if hasattr(mesh, "bounds") else None,
        "extents_mm": mesh.extents.tolist() if hasattr(mesh, "extents") else None,
    }
    (out_dir / "mesh_stats.json").write_text(
        __import__("json").dumps(stats, indent=2)
    )

    # Preview PNG (optional, best-effort)
    save_preview_png(mesh, out_dir / "preview.png")

    print("[OK] Wrote:")
    print(f"  STL:   {out_stl}")
    print(f"  Stats: {out_dir / 'mesh_stats.json'}")
    if (out_dir / "preview.png").exists():
        print(f"  PNG:   {out_dir / 'preview.png'}")

    # Optional viewer
    if args.show:
        try:
            mesh.show()
        except Exception as e:
            print("[WARN] Could not open interactive viewer (likely headless):", e)


if __name__ == "__main__":
    main()
