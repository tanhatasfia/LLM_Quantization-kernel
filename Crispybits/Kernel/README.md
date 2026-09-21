








```bash
pip install -v .          
```


```bash
pytest -q                         
python scripts/bench_gemv.py --bits 3 --n 4096 --k 4096 --exhaustive
python scripts/bench_prefill.py --m 1024
python scripts/ablation_decode.py --model llama2-7b --bits 3     
```




```python
from crispybits_kernels import install_bitmap, tune_model, save_tuning, load_tuning
model = install_bitmap(model, "packed_dir/", bitmap, fuse=True, fuse_rope=True) 
tuning = tune_model(model, batch=1)            
save_tuning(tuning, "tuning_a100.json")

```


