# client_codec.py
import numpy as np
import torch
from typing import Dict, List, Optional

class ClientCompressor:
    """
    Pipeline: L2 clip (C) -> add Gaussian noise (sigma*C) -> (optional) error-feedback ->
              sign -> (dense) pack all indices as (idx, bit) -> optional layer scales.
    Maintains a per-client residual buffer if use_EF=True.
    If dense_mode=True, sampling is disabled and all d indices are sent.
    """
    def __init__(self, d:int, C:float, sigma:float, k:int,
                 use_scale:bool=False, use_EF:bool=False,
                 dense_mode:bool=False,
                 device:torch.device=torch.device("cpu")):
        self.d = int(d)
        self.C = float(C)
        self.sigma = float(sigma)
        self.k = int(k) 
        self.use_scale = bool(use_scale)
        self.use_EF = bool(use_EF)
        self.dense_mode = bool(dense_mode)
        self.device = device
        self.residual: Dict[int, torch.Tensor] = {}

    def _flatten(self, grads: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in grads], dim=0).to(self.device)

    @torch.no_grad()
    def _l2_clip(self, g: torch.Tensor) -> torch.Tensor:
        if self.C is None or self.C <= 0:
            return g
        n = g.norm(p=2).clamp_min(1e-12)
        if n > self.C:
            g = g * (self.C / n)
        return g

    @torch.no_grad()
    def _dp_noise(self, g: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0:
            return g
        noise = torch.randn_like(g) * (self.sigma * self.C)
        return g + noise

    def _layer_scales(self, grads: List[torch.Tensor]) -> Optional[List[float]]:
        if not self.use_scale:
            return None
        return [float(torch.linalg.vector_norm(t).item()) for t in grads]

    def _layer_scale_of_index(self, j:int, grads: List[torch.Tensor], layer_scales: Optional[List[float]]) -> float:
        if (not self.use_scale) or (layer_scales is None):
            return 1.0
        offset = 0
        for idx, t in enumerate(grads):
            n = t.numel()
            if offset <= j < offset + n:
                return float(layer_scales[idx])
            offset += n
        return 1.0

    @torch.no_grad()
    def compress(self, client_id:int, round_id:int,
                 grads: List[torch.Tensor], seed_i: Optional[int]=None) -> dict:
        """
        grads: list[Tensor], aligned with model.parameters()
        Returns Msg_i: {'round_id','seed_i','pairs','layer_scale','mac_tag'}
          - pairs: List[(idx:int, bit:int{0,1})]，dense 模式下长度 = d
        """
        # Flatten -> DP: clip -> noise
        g = self._flatten(grads)
        g = self._dp_noise(self._l2_clip(g))

        # Error feedback
        if self.use_EF:
            r = self.residual.get(client_id, torch.zeros_like(g))
            u = g + r
        else:
            u = g

        # Sign in {-1,+1}; treat zero as +1
        sign = torch.sign(u)
        sign[sign == 0] = 1.0

   
        J = np.arange(self.d, dtype=np.int64)

        # (idx, bit) pairs
        pairs = [(int(j), 1 if sign[j] >= 0 else 0) for j in J.tolist()]

        # optional layer scales
        layer_scale = self._layer_scales(grads)


        if self.use_EF:
            u_hat = torch.zeros_like(u)
            for (j, b) in pairs:
                s = 1.0 if b == 1 else -1.0
                u_hat[j] = s * self._layer_scale_of_index(j, grads, layer_scale)
            self.residual[client_id] = u - u_hat

        # MAC placeholder
        mac_tag = b""

        return dict(
            round_id=int(round_id),
            seed_i=int(seed_i) if seed_i is not None else 0,
            pairs=pairs,
            layer_scale=layer_scale,
            mac_tag=mac_tag,
        )
