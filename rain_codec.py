# rain_codec.py
import torch
from typing import Dict, List, Optional

class RainCompressor:
    """
    L2 clip(C) -> add Gaussian noise (sigma*C) -> (optional) error-feedback ->
    sign (dense) -> pack as (idx,bit) for all d coordinates -> optional per-layer scales.
    """
    def __init__(self, d:int, C:float, sigma:float,
                 use_scale:bool=False, use_EF:bool=False,
                 device:torch.device=torch.device("cpu")):
        self.d = int(d)
        self.C = float(C)
        self.sigma = float(sigma)
        self.use_scale = bool(use_scale)
        self.use_EF = bool(use_EF)
        self.device = device
        self.residual: Dict[int, torch.Tensor] = {}

    def _flatten(self, grads: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in grads], dim=0).to(self.device)

    @torch.no_grad()
    def _l2_clip(self, g: torch.Tensor) -> torch.Tensor:
        if self.C is None or self.C <= 0:
            return g
        n = g.norm(p=2).clamp_min(1e-12)
        return g * (self.C / n) if n > self.C else g

    @torch.no_grad()
    def _dp_noise(self, g: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0:
            return g
        return g + torch.randn_like(g) * (self.sigma * self.C)

    def _layer_scales(self, grads: List[torch.Tensor]) -> Optional[List[float]]:
        if not self.use_scale:
            return None
        return [float(torch.linalg.vector_norm(t).item()) for t in grads]

    def _layer_scale_of_index(self, j:int, grads: List[torch.Tensor], layer_scales: Optional[List[float]]) -> float:
        if (not self.use_scale) or (layer_scales is None):
            return 1.0
        off = 0
        for idx, t in enumerate(grads):
            n = t.numel()
            if off <= j < off + n:
                return float(layer_scales[idx])
            off += n
        return 1.0

    @torch.no_grad()
    def compress(self, client_id:int, round_id:int, grads: List[torch.Tensor]) -> dict:
   
        g = self._dp_noise(self._l2_clip(self._flatten(grads)))
        u = g + self.residual.get(client_id, torch.zeros_like(g)) if self.use_EF else g

        sign = torch.sign(u)
        sign[sign == 0] = 1.0
        idx = torch.arange(self.d, device=sign.device, dtype=torch.long)
        pairs = [(int(j.item()), 1 if sign[j] >= 0 else 0) for j in idx]

        layer_scale = self._layer_scales(grads)

        if self.use_EF:
            u_hat = torch.zeros_like(u)
            for (j, b) in pairs:
                s = 1.0 if b == 1 else -1.0
                u_hat[j] = s * self._layer_scale_of_index(j, grads, layer_scale)
            self.residual[client_id] = u - u_hat

        return dict(round_id=int(round_id), pairs=pairs, layer_scale=layer_scale, mac_tag=b"")
