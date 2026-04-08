from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lpbf_to.surrogates.meltpool_v1.inference import SurrogateInference


# -----------------------------
# Data structures / helpers
# -----------------------------

@dataclass(frozen=True)
class Bounds:
    P_min: float
    P_max: float
    v_min: float
    v_max: float


def sigmoid_bound(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map raw (unbounded) -> [lo, hi] smoothly and differentiably."""
    return lo + (hi - lo) * torch.sigmoid(raw)


def broadcast_tiles_to_voxels(tile_vals: torch.Tensor, tile_indices: torch.Tensor) -> torch.Tensor:
    """
    tile_vals: (K,) or (K,1)
    tile_indices: voxel grid of ints in [0, K-1], shape (X,Y,Z) or (X,Y)
    returns: voxel map same shape as tile_indices
    """
    if tile_vals.ndim == 2 and tile_vals.shape[1] == 1:
        tile_vals = tile_vals[:, 0]
    if tile_vals.ndim != 1:
        raise ValueError(f"tile_vals must be (K,) or (K,1); got {tuple(tile_vals.shape)}")
    if tile_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"tile_indices must be int tensor; got {tile_indices.dtype}")
    return tile_vals[tile_indices]


def make_neighbor_pairs(nx: int, ny: int, nz: int) -> List[Tuple[int, int]]:
    """
    6-neighborhood adjacency on an (nx, ny, nz) tile grid.
    Returns undirected edge list (i, j) with i < j.
    Tile id: i = ix + nx*(iy + ny*iz)
    """
    pairs = set()

    def tid(ix: int, iy: int, iz: int) -> int:
        return ix + nx * (iy + ny * iz)

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                a = tid(ix, iy, iz)
                if ix + 1 < nx:
                    pairs.add(tuple(sorted((a, tid(ix + 1, iy, iz)))))
                if iy + 1 < ny:
                    pairs.add(tuple(sorted((a, tid(ix, iy + 1, iz)))))
                if iz + 1 < nz:
                    pairs.add(tuple(sorted((a, tid(ix, iy, iz + 1)))))
    return sorted(pairs)


def tile_tv_raw(P_raw: torch.Tensor, v_raw: torch.Tensor, neighbor_pairs: List[Tuple[int, int]]) -> torch.Tensor:
    """Tile-space TV on raw params."""
    tv = P_raw.new_zeros(())
    for a, b in neighbor_pairs:
        tv = tv + (P_raw[a] - P_raw[b]).abs() + (v_raw[a] - v_raw[b]).abs()
    return tv / max(len(neighbor_pairs), 1)


# -----------------------------
# ManufacturingLoss
# -----------------------------

class ManufacturingLoss(nn.Module):
    """
    Manufacturing engine:
      - maps K tiles -> voxel P_map / v_map
      - surrogate inference -> width/depth
      - LOF/keyhole risk based on depth thresholds
      - OPTIONAL: aspect-ratio (depth/width) penalty
      - masked reduction using rho^p
      - tile-space TV smoothing on raw params
    """

    def __init__(
        self,
        asset_dir: str,
        tile_shape: Tuple[int, int, int],
        bounds_by_material: Optional[Dict[str, Bounds]] = None,
        tv_weight: float = 0.0,
        rho_power: float = 4.0,
        eps: float = 1e-12,
        device: Optional[torch.device] = None,
        # --- new optional AR controls ---
        ar_weight: float = 0.0,          # set >0 to enable AR penalty
        ar_min: float = 0.0,             # optional lower bound
        ar_max: float = 1.5,             # typical keyhole-ish threshold
    ) -> None:
        super().__init__()

        self.asset_dir = str(asset_dir)
        self.tile_shape = tile_shape
        self.tv_weight = float(tv_weight)
        self.rho_power = float(rho_power)
        self.eps = float(eps)
        self.device_override = device

        self.ar_weight = float(ar_weight)
        self.ar_min = float(ar_min)
        self.ar_max = float(ar_max)

        nx, ny, nz = tile_shape
        self.neighbor_pairs = make_neighbor_pairs(nx, ny, nz)

        cfg_path = Path(self.asset_dir) / "surrogate_config.json"
        self.cfg = json.loads(cfg_path.read_text())

        if device is None:
            self.sur = SurrogateInference(asset_dir=self.asset_dir)
        else:
            self.sur = SurrogateInference(asset_dir=self.asset_dir, device=device)

        self.bounds_by_material = bounds_by_material or {}

    # ---------- bounds ----------
    def _get_bounds(self, material: str) -> Bounds:
        if material in self.bounds_by_material:
            return self.bounds_by_material[material]

        cfg = self.cfg

        vbm = cfg.get("validity_window_by_material", {})
        if isinstance(vbm, dict) and material in vbm:
            mw = vbm[material]
            P_rng = mw.get("P_W", None)
            v_rng = mw.get("v_mm_per_s", None)
            if isinstance(P_rng, list) and len(P_rng) == 2 and isinstance(v_rng, list) and len(v_rng) == 2:
                return Bounds(P_min=float(P_rng[0]), P_max=float(P_rng[1]),
                              v_min=float(v_rng[0]), v_max=float(v_rng[1]))

        vg = cfg.get("validity_window_global", None)
        if isinstance(vg, dict):
            P_rng = vg.get("P_W", None)
            v_rng = vg.get("v_mm_per_s", None)
            if isinstance(P_rng, list) and len(P_rng) == 2 and isinstance(v_rng, list) and len(v_rng) == 2:
                return Bounds(P_min=float(P_rng[0]), P_max=float(P_rng[1]),
                              v_min=float(v_rng[0]), v_max=float(v_rng[1]))

        raise KeyError(
            f"Could not determine P/v bounds for material='{material}'. "
            "Provide bounds_by_material or fix surrogate_config.json validity_window keys."
        )

    def _material_id(self, material: str) -> int:
        m2i = self.cfg.get("material_to_id", None)
        if isinstance(m2i, dict) and material in m2i:
            return int(m2i[material])
        raise KeyError(f"Could not resolve material_id for material='{material}' from surrogate_config.json")

    # ---------- surrogate call ----------
    def _surrogate_forward(
        self,
        P_map: torch.Tensor,
        v_map: torch.Tensor,
        material: str,
        h_mm: float,
        t_mm: float,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        mat_id = self._material_id(material)
        out = self.sur(P_map, v_map, h_mm, t_mm, mat_id)
        return self._parse_surrogate_output(out)

    @staticmethod
    def _parse_surrogate_output(out: Any) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if isinstance(out, dict):
            if "depth" in out:
                depth = out["depth"]
            elif "depth_um" in out:
                depth = out["depth_um"]
            else:
                raise KeyError(f"Surrogate output dict missing depth key. Keys={list(out.keys())}")

            width = None
            if "width" in out:
                width = out["width"]
            elif "width_um" in out:
                width = out["width_um"]
            return width, depth

        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return out[0], out[1]

        if torch.is_tensor(out) and out.ndim >= 1 and out.shape[-1] == 2:
            return out[..., 0], out[..., 1]

        raise TypeError(f"Unrecognized surrogate output type: {type(out)}")

    # ---------- forward ----------
    def forward(
        self,
        rho: torch.Tensor,
        P_raw: torch.Tensor,
        v_raw: torch.Tensor,
        tile_indices: torch.Tensor,
        material: str,
        h_mm: float,
        t_mm: float,
        d_req_um: float,
        d_max_um: float,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = self.device_override if self.device_override is not None else rho.device

        rho = rho.to(device)
        tile_indices = tile_indices.to(device)
        P_raw = P_raw.to(device)
        v_raw = v_raw.to(device)

        if rho.shape != tile_indices.shape:
            raise ValueError(f"rho.shape {tuple(rho.shape)} must match tile_indices.shape {tuple(tile_indices.shape)}")

        # TV in tile space on raw params
        L_tv = tile_tv_raw(P_raw, v_raw, self.neighbor_pairs)

        # bounds + tile->voxel maps
        b = self._get_bounds(material)
        P_tile = sigmoid_bound(P_raw, b.P_min, b.P_max)
        v_tile = sigmoid_bound(v_raw, b.v_min, b.v_max)

        P_map = broadcast_tiles_to_voxels(P_tile, tile_indices)
        v_map = broadcast_tiles_to_voxels(v_tile, tile_indices)

    # surrogate
        width_map, depth_map = self._surrogate_forward(P_map, v_map, material, h_mm, t_mm)

        # depth thresholds
        d_req = torch.as_tensor(d_req_um, device=depth_map.device, dtype=depth_map.dtype)
        d_max = torch.as_tensor(d_max_um, device=depth_map.device, dtype=depth_map.dtype)

        lof_map = F.relu(d_req - depth_map)
        keyhole_map = F.relu(depth_map - d_max)

        # start with depth-only risk
        risk_map = lof_map + keyhole_map

        # ---- aspect ratio term (depth/width) ----
        L_ar = depth_map.new_zeros(())
        ar_map = None
        ar_pen_map = None

        if (width_map is not None) and (float(getattr(self, "ar_weight", 0.0)) > 0.0):
            # AR = depth / width
            ar_map = depth_map / (width_map + self.eps)

            ar_max = float(getattr(self, "ar_max", 1.5))
            ar_min = float(getattr(self, "ar_min", 0.0))
            ar_weight = float(getattr(self, "ar_weight", 0.0))

            # too deep vs width => keyhole-ish
            ar_hi = F.relu(ar_map - ar_max)

            # optional: too wide/shallow (usually keep ar_min=0.0 to disable)
            if ar_min > 0.0:
                ar_lo = F.relu(ar_min - ar_map)
            else:
                ar_lo = depth_map.new_zeros(depth_map.shape)

            ar_pen_map = ar_hi + ar_lo

            # inject into risk
            risk_map = risk_map + ar_weight * ar_pen_map

        # masked reduction
        rho_p = rho.clamp(min=0.0, max=1.0).pow(self.rho_power)
        if active_mask is None:
            am = torch.ones_like(rho_p)
        else:
            am = active_mask.to(device=rho_p.device, dtype=rho_p.dtype)

        mask = rho_p * am
        den = mask.sum() + self.eps

        L_lof = (lof_map * mask).sum() / den
        L_key = (keyhole_map * mask).sum() / den
        L_risk = (risk_map * mask).sum() / den

        if (ar_pen_map is not None):
            L_ar = (ar_pen_map * mask).sum() / den

        L_total_mfg = L_risk + self.tv_weight * L_tv

        def stats(x: torch.Tensor) -> torch.Tensor:
            return torch.stack([x.min(), x.mean(), x.max()])

        out: Dict[str, torch.Tensor] = {
            "loss_total_mfg": L_total_mfg,
            "loss_tv": L_tv,
            "loss_lof": L_lof,
            "loss_keyhole": L_key,
            "loss_risk": L_risk,
            "P_map_stats": stats(P_map),
            "v_map_stats": stats(v_map),
            "depth_stats": stats(depth_map),
            "mask_sum": mask.sum(),
        }

        if width_map is not None:
            out["width_stats"] = stats(width_map)

        if ar_map is not None:
            out["loss_ar"] = L_ar
            out["ar_stats"] = stats(ar_map)

        return out

