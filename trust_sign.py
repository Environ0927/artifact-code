

from typing import Iterator, Tuple
import torch
import torch.nn as nn

@torch.no_grad()
def flatten_params(net: nn.Module, device=None, dtype=torch.float32) -> torch.Tensor:

    flats = []
    for p in net.parameters():
        flats.append(p.detach().reshape(-1).to(dtype=dtype))
    out = torch.cat(flats, dim=0)
    if device is not None:
        out = out.to(device)
    return out  # [D]

@torch.no_grad()
def sign_from_model(net: nn.Module, device=None) -> torch.Tensor:
 
    w = flatten_params(net, device=device, dtype=torch.float32)   # [D], float
    s = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    return s.to(torch.int8)  # [-1,+1] cast int8

@torch.no_grad()
def apply_signed_direction_step(net: nn.Module, s_dir: torch.Tensor, step_size: float) -> None:
  
    if s_dir.dtype != torch.float32 and s_dir.dtype != torch.float16:
        s_dir = s_dir.to(torch.float32)
    off = 0
    for p in net.parameters():
        n = p.numel()
        d = s_dir[off:off+n].view_as(p)
        p.add_(d, alpha=-float(step_size))
        off += n

@torch.no_grad()
def overwrite_model_sign_to(net: nn.Module, s_dir: torch.Tensor) -> None:
  
    if s_dir.dtype != torch.float32 and s_dir.dtype != torch.float16:
        s_dir = s_dir.to(torch.float32)
    off = 0
    for p in net.parameters():
        n = p.numel()
        tgt = s_dir[off:off+n].view_as(p)
        mag = p.data.abs()
        p.data = torch.sign(tgt) * mag
        off += n
