import torch

class ProcessRiskLoss(torch.nn.Module):
    """
    Calls a surrogate f̂(P,v,h,t,geom)-> width, depth, risk_logits (lof,keyhole)
    and penalizes out-of-band width and defect risk.
    """
    def __init__(self, surrogate, width_band_um=(80., 120.), risk_weights=(1.0, 1.0)):
        super().__init__()
        self.surrogate = surrogate
        self.wmin, self.wmax = width_band_um
        self.rw_lof, self.rw_key = risk_weights

    def forward(self, params, geom_feats):
        # params: (...,4) = [P_W, v_mm_s, h_um, t_um]; geom_feats: (...,G)
        width_um, depth_um, risk_logits = self.surrogate(params, geom_feats)
        w_pen = torch.relu(self.wmin - width_um) + torch.relu(width_um - self.wmax)
        lof_logit, key_logit = risk_logits.unbind(-1)
        risk_pen = self.rw_lof * torch.nn.functional.softplus(lof_logit) \
                 + self.rw_key * torch.nn.functional.softplus(key_logit)
        return w_pen.mean() + risk_pen.mean()
