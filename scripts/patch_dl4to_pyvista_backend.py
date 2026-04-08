import importlib.util
import pathlib
import re
import sys

def main() -> int:
    spec = importlib.util.find_spec("dl4to")
    if spec is None or spec.submodule_search_locations is None:
        print("[FAIL] dl4to not found")
        return 1

    dl4to_dir = pathlib.Path(list(spec.submodule_search_locations)[0])
    plotting_path = dl4to_dir / "plotting.py"
    if not plotting_path.exists():
        print("[FAIL] plotting.py not found at:", plotting_path)
        return 1

    txt = plotting_path.read_text(encoding="utf-8")

    # If already patched, do nothing.
    if "Patched by lpbf_project" in txt and "pv.set_jupyter_backend('none')" in txt:
        print("[OK] Already patched:", plotting_path)
        return 0

    # Replace ONLY the exact backend call line if present.
    pat = re.compile(r"(?m)^([ \t]*)pv\.set_jupyter_backend\(['\"]pythreejs['\"]\)\s*$")
    m = pat.search(txt)
    if not m:
        print("[INFO] No pv.set_jupyter_backend('pythreejs') line found. Nothing to patch.")
        print("[INFO] File:", plotting_path)
        return 0

    indent = m.group(1)
    replacement = (
        f"{indent}# Patched by lpbf_project: PyVista backend 'pythreejs' is not supported\n"
        f"{indent}try:\n"
        f"{indent}    pv.set_jupyter_backend('none')\n"
        f"{indent}except Exception:\n"
        f"{indent}    pass\n"
    )

    txt2 = pat.sub(replacement, txt, count=1)
    plotting_path.write_text(txt2, encoding="utf-8")
    print("[OK] Patched:", plotting_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
