# tssc_relu.py
import numpy as np
import torch
from typing import List, Dict, Optional
from rain_hd_rpc import pack_row_kplus1, s_trust_bits16_from_pm1, call_rain_hd
import os

@torch.no_grad()
def tssc_relu_aggregate(
    msgs: List[dict],
    net: torch.nn.Module,
    s_trust: torch.Tensor,          # shape [d], values in {-1,+1} (dtype int8)
    device: torch.device,
    lambda_mad: float = 2.2,
    tau_override: Optional[float] = None,
    w_max: Optional[float] = None,
    use_scale: bool = False,
    global_step_size: float = 1.0,
) -> torch.Tensor:
  
    rain_bin = os.path.expanduser("~/517/safefl/Secure-Shuffling/rain_hd")
    D = len(s_trust)
    k = len(msgs[0]["pairs"])
    tau_frac = None  


    rows = []
    for m in msgs:
        if "seed16" not in m:
        
            m["seed16"] = bytes([len(rows) % 256]) * 16
        rows.append(pack_row_kplus1(m["pairs"], m["seed16"]))

    # s_trust {-1,+1} → {0,1} → 16B/位
    s_bits16 = s_trust_bits16_from_pm1(s_trust.cpu().numpy())

  
    _, hd_vals = call_rain_hd(rain_bin, D, k, len(msgs[0]["pairs"]), rows, s_bits16, mode="plain")
    d = torch.tensor(hd_vals, dtype=torch.float32) / float(k)


    if tau_override is None:
        med = torch.median(d)
        mad = torch.median(torch.abs(d - med)) + 1e-9
        tau = float(med + 1.4826 * lambda_mad * mad)
    else:
        tau = float(tau_override)

 
    w = torch.clamp(tau - d, min=0.0)
    if w_max is not None:
        w = torch.clamp(w, max=float(w_max))
    sumw = float(w.sum().item()) + 1e-12
    alpha = (w / sumw).cpu().numpy()  # sum alpha_i = 1


    S: Dict[int, float] = {}
    for i, m in enumerate(msgs):
        ai = float(alpha[i])
        if ai <= 0:
            continue
        for (j, b) in m["pairs"]:
            S[j] = S.get(j, 0.0) + ai * (1.0 if b == 1 else -1.0)

    s_trust_next = s_trust.clone()
    for j, Sj in S.items():
        s_trust_next[j] = 1 if Sj >= 0 else -1

    # ----- 5) scale_bar -----
    if use_scale:
        scales = []
        for i, m in enumerate(msgs):
            ls = m.get("layer_scale")
            if ls is None or len(ls) == 0:
                continue
            s = float(np.median(np.asarray(ls, dtype=np.float32)))  # per-client summary
            scales.append((s, float(alpha[i])))
        if len(scales) > 0:
            vals = np.array([v for v, _ in scales], dtype=np.float32)
            ws   = np.array([w for _, w in scales], dtype=np.float32)
            order = np.argsort(vals)
            vals, ws = vals[order], ws[order]
            cum = np.cumsum(ws) / (ws.sum() + 1e-12)
            pos = np.searchsorted(cum, 0.5)
            scale_bar = float(vals[min(pos, len(vals) - 1)])
        else:
            scale_bar = 1.0
    else:
        scale_bar = 1.0

    step = float(global_step_size) * float(scale_bar)
    direction = s_trust_next.to(torch.float32).to(device)  # {-1,+1}
    offset = 0
    for p in net.parameters():
        n = p.numel()
        seg = direction[offset:offset + n].view_as(p)
        p.add_(seg, alpha=-step)
        offset += n

    return s_trust_next

@torch.no_grad()
def rainy_tssc_aggregate(grad_list, net, device, s_trust: torch.Tensor,
                         step_size: float = 1.0):
    
    apply_signed_direction_step(net, s_trust, step_size)