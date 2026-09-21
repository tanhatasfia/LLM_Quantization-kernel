import pytest, torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_placeholder_import():
    from crispybits_kernels import candidate_split_k
    assert candidate_split_k(1, 5120, 128, 108, 1) >= 1
