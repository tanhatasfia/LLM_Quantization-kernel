# CrispyBits — Quantization / TRPS / Energy Allocation Reference Code



pip install -r requirements.txt
export PYTHONPATH=$PWD

# 1. TRPS scores
python scripts/calibrate_trps.py \
  --model meta-llama/Llama-2-7b-hf \
  --samples 500 --seq-len 2048 --out trps.csv

# 2. Generate K random bitmaps (paper K depends on model)
python scripts/sample_bitmaps.py --blocks 32 --k 700 --out bitmaps.npy

# 4. Fit surrogate
python scripts/fit_surrogate.py --npz calibration.npz --out energy_surrogate.json

# 5. Allocate under a request-energy budget
python scripts/allocate.py --trps-csv trps.csv \
  --surrogate energy_surrogate.json --budget 1098.2 --out allocation.json

# 6. Pack selected model weights
python scripts/pack_checkpoint.py --model meta-llama/Llama-2-7b-hf \
  --bitmap allocation.json --out llama2_7b_crispybits
```

