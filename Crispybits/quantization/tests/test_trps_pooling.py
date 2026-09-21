import torch
from crispybits_quant.trps import TRPSAccumulator, cvar


def test_cvar_is_pooled_over_all_tokens():
    torch.manual_seed(0)
    
    d1 = torch.rand(20)                   
    d2 = torch.rand(2000) * 5.0           
    p1, p2 = torch.rand(20), torch.rand(2000)
    acc = TRPSAccumulator(tau=0.10, lambda_tail=0.25, lambda_prop=0.25)
    acc.add({(0, 2): (d1, p1)})
    acc.add({(0, 2): (d2, p2)})
    (r,) = acc.finalize()
    d = torch.cat([d1, d2]); p = torch.cat([p1, p2])
    assert abs(r.mean_local - d.mean().item()) < 1e-5         
    assert abs(r.cvar_tail - cvar(d, 0.10).item()) < 1e-5      
    assert abs(r.mean_prop - p.mean().item()) < 1e-5
    assert abs(r.trps - (r.mean_local + 0.25*r.cvar_tail + 0.25*r.mean_prop)) < 1e-6
    
    old_cvar = (cvar(d1, 0.10) + cvar(d2, 0.10)).item() / 2
    assert abs(r.cvar_tail - old_cvar) > 1e-3


def test_last_block_has_no_propagation():
    acc = TRPSAccumulator()
    acc.add({(3, 4): (torch.ones(10), None)})
    (r,) = acc.finalize()
    assert r.mean_prop == 0.0
