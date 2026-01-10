# aggregation_rules_tssc.py
import torch
from typing import List, Dict, Tuple, Optional
from shuffler import Shuffler 
from client_codec import ClientCompressor
from tssc_relu import tssc_relu_aggregate
from trust_sign import apply_signed_direction_step
from rain_hd_rpc import call_rain_shuffle_perm
import os

@torch.no_grad()
def rainy_tssc_step(
    grad_list: List[List[torch.Tensor]],
    net: torch.nn.Module,
    device: torch.device,
    s_trust: Optional[torch.Tensor] = None,
    round_id: int = 0,
    k_bits: int = 8,           
    dp_clip: float = 1.0,
    dp_sigma: float = 0.0,
    use_scale: bool = False,
    use_EF: bool = False,
    lambda_mad: float = 2.2,
    tau_override: Optional[float] = None,
    w_max: Optional[float] = None,
    global_lr: float = 0.05,
) -> Tuple[torch.Tensor, Dict]:

    d = 0
    for p in net.parameters():
        d += p.numel()

    if s_trust is None:
        s_trust = torch.ones(d, dtype=torch.int8, device=device)

    k_bits = d
    compressor = ClientCompressor(
        d=d, C=dp_clip, sigma=dp_sigma, k=k_bits,
        use_scale=use_scale, use_EF=use_EF,
        dense_mode=True, device=device
    )

    msgs = []
    for cid, grads in enumerate(grad_list):
        msg = compressor.compress(client_id=cid, round_id=round_id, grads=grads, seed_i=None)
        msgs.append(msg)

    try:
        perm = call_rain_shuffle_perm(
            os.environ.get("rain_SHUFFLE_BIN", ""),
            n_rows=len(msgs),
            seed=round_id
        )
        msgs = [msgs[i] for i in perm]
    except Exception:
        pass

    s_trust_next, tau, hd_median, hd_mad, kept = tssc_relu_aggregate(
        msgs=msgs, net=net, s_trust=s_trust, device=device,
        lambda_mad=lambda_mad, tau_override=tau_override, w_max=w_max
    )

    apply_signed_direction_step(net, s_trust_next, global_lr)

    stats = dict(
        round_id=int(round_id),
        d=int(d),
        k_bits=int(k_bits),          
        dp_clip=float(dp_clip),
        dp_sigma=float(dp_sigma),
        lambda_mad=float(lambda_mad),
        tau=float(tau),
        hd_median=float(hd_median),
        hd_mad=float(hd_mad),
        kept=int(kept),
        n_clients=len(grad_list),
        mode="dense",
    )
    return s_trust_next, stats
