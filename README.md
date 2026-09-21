

Official implementation of **CrispyBits: Packed LLM Quantization Tracks Energy Budget**.


```text
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
```

## 1. Installation

### Quantization and Allocation

```bash
cd Crispybits/quantization
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

### CUDA Kernels

```bash
cd Crispybits/Kernel
pip install -r requirements.txt
pip install -v .
```



## 2. TRPS Calibration



```bash
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
```



## 3. Pack W2/W3/W4 Weights



```bash
python scripts/pack_all_precisions.py \
    --model meta-llama/Llama-2-7b-hf \
    --group-size 128 \
    --out packed_llama2_7b
```



## 4. Sample Mixed-Precision Configurations



Example for LLaMA-2 7B:

```bash
python scripts/sample_bitmaps.py \
    --blocks 32 \
    --k 700 \
    --out bitmaps.npy
```

## 5. Fit the Hardware-Energy Surrogate



```bash
python scripts/fit_surrogate.py \
    --npz calibration.npz \
    --out energy_surrogate.json
```


## 6. Energy-Constrained Mixed-Precision Allocation


Example:

```bash
python scripts/allocate.py \
    --trps-csv trps.csv \
    --surrogate energy_surrogate.json \
    --budget 1098.2 \
    --out allocation.json
```



## 7. Kernel Tests and Microbenchmarks

### Run Correctness Tests

```bash
cd Crispybits/Kernel
pytest -q
```

### GEMV Microbenchmark

```bash
python scripts/bench_gemv.py \
    --bits 3 \
    --n 4096 \
    --k 4096 \
    --exhaustive
```

### Prefill GEMM Microbenchmark

```bash
python scripts/bench_prefill.py --m 1024
```

### Decode-Kernel Ablation



