# CrispyBits

Official implementation of **CrispyBits: Packed LLM Quantization Tracks Energy Budget**.


## Repository Structure


Crispybits/
├── quantization/
│   ├── crispybits_quant/
│   │   ├── affine.py        # Group-wise affine W2/W3/W4 quantization
│   │   ├── packing.py       # Physical low-bit weight packing
│   │   ├── trps.py          # TRPS sensitivity computation
│   │   ├── energy.py        # GPU power/energy measurement
│   │   ├── surrogate.py     # Additive energy surrogate
│   │   ├── mckp.py          # Energy-constrained MCKP allocation
│   │   └── model_utils.py
│   ├── scripts/
│   └── tests/
│
└── Kernel/
    ├── crispybits_kernels/
    │   ├── ops.py           # Packed kernel interface and Split-K policy
    │   ├── integration.py   # Transformer integration
    │   ├── autotune.py      # Kernel configuration tuning
    │   ├── emulator.py
    │   └── reference.py
    ├── csrc/
    │   ├── packed_gemv.cu   # Decode packed GEMV
    │   ├── packed_gemm.cu   # Prefill packed GEMM
    │   ├── bitpack.cuh
    │   └── bindings.cpp
    ├── scripts/
    └── tests/



1. Installation
Quantization / Allocation
cd Crispybits/quantization
pip install -r requirements.txt
export PYTHONPATH=$PWD
CUDA Kernels



cd Crispybits/Kernel
pip install -r requirements.txt
pip install -v .
2. TRPS Calibration



cd Crispybits/quantization

python scripts/calibrate_trps.py \
    --model meta-llama/Llama-2-7b-hf \
    --samples 500 \
    --seq-len 2048 \
    --group-size 128 \
    --tau 0.10 \
    --lambda-tail 0.25 \
    --lambda-prop 0.25 \
    --out trps.csv



3. Pack W2/W3/W4 Weights

For hardware-energy calibration, all candidate precisions can be packed:

python scripts/pack_all_precisions.py \
    --model meta-llama/Llama-2-7b-hf \
    --group-size 128 \
    --out packed_llama2_7b


4. Sample Mixed-Precision Configurations



For LLaMA-2 7B:

python scripts/sample_bitmaps.py \
    --blocks 32 \
    --k 700 \
    --out bitmaps.npy




5. Fit the Hardware-Energy Surrogate


python scripts/fit_surrogate.py \
    --npz calibration.npz \
    --out energy_surrogate.json



7. Energy-Constrained Mixed-Precision Allocation


Example:

python scripts/allocate.py \
    --trps-csv trps.csv \
    --surrogate energy_surrogate.json \
    --budget 1098.2 \
    --out allocation.json





6. Kernel Tests and Microbenchmarks

Run correctness tests:

cd Crispybits/Kernel
pytest -q

Example GEMV benchmark:

python scripts/bench_gemv.py \
    --bits 3 \
    --n 4096 \
    --k 4096 \
    --exhaustive

Example prefill benchmark:

python scripts/bench_prefill.py --m 1024



python scripts/ablation_decode.py \
    --model llama2-7b \
    --bits 3
