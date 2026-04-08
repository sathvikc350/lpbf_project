#!/usr/bin/env bash
set -euo pipefail

echo "[post_create] updating apt libs needed by VTK/OpenGL..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends libgl1
sudo rm -rf /var/lib/apt/lists/*

echo "[post_create] upgrading pip toolchain..."
python -m pip install --upgrade pip setuptools wheel

echo "[post_create] installing core Python deps..."
# CPU-only PyTorch (works on Linux x86, Python 3.10)
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  torch==2.5.1 torchvision==0.20.1

# viz + utils
python -m pip install \
  vtk==9.2.6 pyvista==0.43.9 pyvistaqt==0.11.1 \
  matplotlib tqdm einops loguru pre-commit jupyter ipykernel

# --- lpbf_project: fix dl4to PyVista backend crash (pythreejs removed in newer PyVista) ---
# Safe to run every time; does nothing if already patched.
python /workspaces/lpbf_project/scripts/patch_dl4to_pyvista_backend.py || true

echo "[post_create] registering Jupyter kernel..."
python -m ipykernel install --user --name lpbf-to --display-name "Python (lpbf-to)"

echo "[post_create] smoke-test PyVista offscreen render..."
# tools for headless (virtual display + Mesa)
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends xvfb libgl1-mesa-dri
sudo rm -rf /var/lib/apt/lists/*

mkdir -p outputs
python - <<'PY'
import pyvista as pv


try:
    pv.start_xvfb()  # spins up Xvfb if available
except Exception as e:
    print("Warning: pv.start_xvfb() failed:", e)

# Use an explicit off-screen plotter (API-stable)
mesh = pv.Sphere()
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh)
plotter.show(screenshot="outputs/sphere.png")
print("Smoke test OK: outputs/sphere.png")
PY
