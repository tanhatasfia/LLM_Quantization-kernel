
import pytest
import torch

from crispybits_kernels import ops


def _cuda_ok():
    return torch.cuda.is_available() and ops._CUDA is not None


@pytest.fixture(params=["emulator", "cuda"])
def env(request):
    name = request.param
    if name == "cuda" and not _cuda_ok():
        pytest.skip("CUDA extension / GPU not available")
    prev = ops.backend_name()
    ops.set_backend(name)
    ops._sm_count_for.cache_clear()
    yield "cpu" if name == "emulator" else "cuda"
    ops.set_backend(prev if ops._CUDA is not None or prev == "emulator" else "emulator")
    ops._sm_count_for.cache_clear()
