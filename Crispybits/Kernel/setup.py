from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="crispybits_kernels",
    version="0.1.0",
    packages=["crispybits_kernels"],
    ext_modules=[
        CUDAExtension(
            name="crispybits_kernels._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/packed_gemv.cu",
                "csrc/packed_gemm.cu",
                "csrc/fused_gate_up.cu",
                "csrc/fused_qkv.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
