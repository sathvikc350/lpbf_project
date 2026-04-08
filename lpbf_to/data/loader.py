import pandas as pd

CANON = ["material","machine","P_W","v_mm_s","h_um","t_um","spot_um",
         "scan_strategy","preheat_C","shield_gas",
         "mp_width_um","mp_depth_um","porosity_pct","risk_lof","risk_keyhole"]

def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in CANON:
        if c not in df.columns:
            df[c] = None
    return df
