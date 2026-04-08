
import os, json
import joblib
import torch
import torch.nn as nn

class MeltPoolSurrogate(nn.Module):
    """
    Must match your trained architecture exactly.
    """
    def __init__(self, num_numeric_features, num_materials=3, emb_dim=3, hidden_sizes=(64,64), dropout=0.10):
        super().__init__()
        self.material_emb = nn.Embedding(num_materials, emb_dim)

        layers = []
        in_dim = num_numeric_features + emb_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(self, X_num, mat_id):
        mat_vec = self.material_emb(mat_id)
        x = torch.cat([X_num, mat_vec], dim=1)
        return self.mlp(x)


class SurrogateInference(nn.Module):
    """
    Differentiable wrapper for DL4TO:
    - Loads config + weights + sklearn scaler
    - Uses torch-only scaling (NO scaler.transform)
    - Computes LED & VED in torch
    """
    def __init__(self, asset_dir: str, device: str | torch.device = "cpu"):
        super().__init__()
        self.asset_dir = asset_dir
        self.device = torch.device(device)

        # --- Load config ---
        cfg_path = os.path.join(asset_dir, "surrogate_config.json")
        with open(cfg_path, "r") as f:
            self.cfg = json.load(f)

        self.numeric_features = self.cfg["numeric_features"]
        self.outputs = self.cfg["outputs"]
        self.material_to_id = self.cfg["material_to_id"]

        # --- Load scaler and convert mean/std to torch tensors (gradient-safe) ---
        scaler_path = os.path.join(asset_dir, "scaler.joblib")
        scaler = joblib.load(scaler_path)

        mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=self.device)
        scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=self.device)

        # Register as buffers so they move with .to(device) and get saved cleanly
        self.register_buffer("scaler_mean", mean)
        self.register_buffer("scaler_scale", scale)

        # --- Build model + load weights ---
       
        model_cfg = self.cfg.get("model", self.cfg)  # if "model" missing, fall back to root

        self.model = MeltPoolSurrogate(
            num_numeric_features=len(self.numeric_features),
            num_materials=len(self.material_to_id),
            emb_dim=int(model_cfg["emb_dim"]),
            hidden_sizes=tuple(model_cfg["hidden_sizes"]),
            dropout=float(model_cfg["dropout"]),
        ).to(self.device)


        w_path = os.path.join(asset_dir, "weights.pt")
        state = torch.load(w_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)

        # Freeze weights (but keep graph for inputs!)
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    def forward(self, P, v, h, t, mat_id: int | torch.Tensor):
        """
        Supports scalars or tensors for P,v,h,t.
        Returns: width_um, depth_um (same shape as P/v)
        """

        # Convert inputs to tensors on device
        P = torch.as_tensor(P, dtype=torch.float32, device=self.device)
        v = torch.as_tensor(v, dtype=torch.float32, device=self.device)
        h = torch.as_tensor(h, dtype=torch.float32, device=self.device)
        t = torch.as_tensor(t, dtype=torch.float32, device=self.device)

        # Avoid divide-by-zero
        eps = 1e-12
        v_safe = torch.clamp(v, min=eps)

        # Physics features (torch)
        LED = P / v_safe
        VED = P / (v_safe * h * t + eps)

        # Broadcast everything to a common shape
        # (works for scalars or voxel grids)
        target_shape = torch.broadcast_shapes(P.shape, v.shape, h.shape, t.shape, LED.shape, VED.shape)
        P   = P.expand(target_shape)
        v   = v.expand(target_shape)
        LED = LED.expand(target_shape)
        VED = VED.expand(target_shape)
        h   = h.expand(target_shape)
        t   = t.expand(target_shape)

        # mat_id tensor expanded to same shape
        if isinstance(mat_id, int):
            mat_id = torch.full(target_shape, mat_id, dtype=torch.long, device=self.device)
        else:
            mat_id = torch.as_tensor(mat_id, dtype=torch.long, device=self.device).expand(target_shape)

        # Flatten to (N, 6) for MLP
        X_raw = torch.stack([P, v, LED, VED, h, t], dim=-1).reshape(-1, len(self.numeric_features))
        mat_flat = mat_id.reshape(-1)

        # Torch-only scaling (keeps gradients)
        X_scaled = (X_raw - self.scaler_mean) / self.scaler_scale

        # Predict (N,2) then reshape back
        y = self.model(X_scaled, mat_flat)
        y = y.reshape(*target_shape, 2)

        width = y[..., 0]
        depth = y[..., 1]
        return width, depth
