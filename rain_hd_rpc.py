import base64, json, subprocess, struct
from typing import List, Tuple

def _elt16_from_bit(b: int) -> bytes:
    return struct.pack("<Q", int(b)) + b"\x00"*8

def pack_row_kplus1(pairs: List[Tuple[int,int]], seed16: bytes) -> bytes:
    k = len(pairs)
    bits = [_elt16_from_bit(b) for _, b in pairs]
    assert len(seed16) == 16
    return b"".join(bits) + seed16

def s_trust_bits16_from_pm1(pm1) -> bytes:
    out = bytearray()
    for v in pm1: out += _elt16_from_bit(1 if v>0 else 0)
    return bytes(out)

def call_rain_hd(bin_path: str, D: int, k: int, tau_count: int,
                  rows: List[bytes], s_trust_bits16: bytes):
    req = {
        "D": D, "k": k, "tau_count": tau_count,
        "rows_b64": [base64.b64encode(r).decode() for r in rows],
        "s_trust_bits_b64": base64.b64encode(s_trust_bits16).decode(),
    }
    p = subprocess.run([bin_path], input=json.dumps(req).encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = json.loads(p.stdout.decode() or '{}')
    if "pass_mask" not in out:
        raise RuntimeError(f"rain_hd failed: {p.stderr.decode()} {out}")
    return out["pass_mask"], out.get("hd")

def call_rain_hd(bin_path, D, k, tau_count, rows, s_trust_bits16, mode="plain"):
    req = {
        "D": D, "k": k, "tau_count": tau_count, "mode": mode,
        "rows_b64": [base64.b64encode(r).decode() for r in rows],
        "s_trust_bits_b64": base64.b64encode(s_trust_bits16).decode(),
    }
    p = subprocess.run([bin_path], input=json.dumps(req).encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = json.loads(p.stdout.decode())
    if "pass_mask" not in out and "hd" not in out and "msg" in out:
        raise RuntimeError(f"rain_hd error: {out['msg']}")
    return out.get("pass_mask"), out.get("hd")

def call_rain_shuffle(bin_path, flat_bytes: bytes, D: int, k: int):
    req = {
        "D": D,
        "k": k,
        "mode": "shuffle",
        "rows_b64": [base64.b64encode(flat_bytes).decode()],
    }
    p = subprocess.run([bin_path], input=json.dumps(req).encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = json.loads(p.stdout.decode() or "{}")
    shuffled = base64.b64decode(out.get("shuffled_b64", ""))
    perm = out.get("perm", [])
    return shuffled, perm

def call_rain_shuffle_perm(bin_path: str, n_rows: int, blocks_per_row: int, round_id: int):
    req = {
        "D": n_rows,
        "k": blocks_per_row,
        "mode": "shuffle",
        "round_id": round_id,                
        "rows_b64": [],                      
    }
    p = subprocess.run([bin_path], input=json.dumps(req).encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = json.loads(p.stdout.decode() or "{}")
    if "perm" not in out:
        raise RuntimeError(f"rain_hd shuffle error: {out.get('msg') or p.stderr.decode()}")
    return out["perm"]