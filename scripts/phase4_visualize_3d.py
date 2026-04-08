import argparse
from pathlib import Path
import numpy as np
import torch

def marching_cubes(volume: np.ndarray, level: float):
    # Lazy import so you get a clean error if skimage isn't installed
    from skimage.measure import marching_cubes
    verts, faces, normals, values = marching_cubes(volume, level=level)
    return verts, faces

def write_stl_ascii(path: Path, verts: np.ndarray, faces: np.ndarray):
    # Simple ASCII STL writer (no external deps)
    with open(path, "w") as f:
        f.write("solid lpbf_to\n")
        for tri in faces:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            # normal (rough)
            n = np.cross(v1 - v0, v2 - v0)
            n_norm = np.linalg.norm(n) + 1e-12
            n = n / n_norm
            f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
            f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
            f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid lpbf_to\n")

def write_plotly_html(path: Path, verts: np.ndarray, faces: np.ndarray, title: str):
    import plotly.graph_objects as go

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=1.0
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.write_html(str(path), include_plotlyjs="cdn")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", required=True, help="Path to theta_phys_beta*.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iso", type=float, default=0.5)
    ap.add_argument("--voxel_mm", type=float, default=0.25, help="If you want mm scaling later")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t = torch.load(args.theta, map_location="cpu")

# If checkpoint is a dict, try common keys
    if isinstance(t, dict):
        for key in ("theta", "theta_phys", "tensor", "state", "data"):
            if key in t and isinstance(t[key], torch.Tensor):
                t = t[key]
                break
# Hard fail if still not a tensor (prevents silent weirdness)
    if not isinstance(t, torch.Tensor):
         raise TypeError(f"Expected a torch.Tensor from {args.theta}, got {type(t)} with keys={list(t.keys()) if isinstance(t, dict) else None}")

    t = t.detach().cpu().float()


    # Handle shape: (1,NX,NY,NZ) or (NX,NY,NZ)
    if t.ndim == 4:
        t = t[0]
    vol = t.numpy()

    # marching cubes expects (Z,Y,X) sometimes; we’ll keep it consistent and just render
    verts, faces = marching_cubes(vol, level=args.iso)

    # Save assets
    stl_path = out / "part_iso.stl"
    html_path = out / "part_iso.html"
    write_stl_ascii(stl_path, verts, faces)
    write_plotly_html(html_path, verts, faces, title=f"Isosurface iso={args.iso} | {Path(args.theta).name}")

    print("[OK] Wrote:")
    print(" -", stl_path)
    print(" -", html_path)

if __name__ == "__main__":
    main()
