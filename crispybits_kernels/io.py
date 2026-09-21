import torch
from .ops import PackedLinearWeight


def from_packed_dict(d: dict, device="cuda", dtype=torch.float16) -> PackedLinearWeight:
    return PackedLinearWeight(
        packed=d["packed"].to(device=device,dtype=torch.int32),
        scales=d["scales"].to(device=device,dtype=dtype),
        zeros=d["zeros"].to(device=device,dtype=dtype),
        bits=int(d["bits"]),
        group_size=int(d["group_size"]),
        in_features=int(d["in_features"]),
        out_features=int(d["out_features"]),
        bias=None if d.get("bias") is None else d["bias"].to(device=device,dtype=dtype),
    )
