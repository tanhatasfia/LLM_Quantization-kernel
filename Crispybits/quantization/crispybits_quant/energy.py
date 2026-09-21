from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from time import perf_counter, sleep
from typing import Callable, Optional
import os


@dataclass
class EnergyMeasurement:
    joules: float
    seconds: float
    samples: int
    mean_watts: float


class PowerSampler:
    def read_watts(self) -> float:
        raise NotImplementedError


class NVMLPowerSampler(PowerSampler):
    def __init__(self, device_index: int = 0):
        try:
            import pynvml
        except ImportError as e:
            raise RuntimeError("Install nvidia-ml-py for NVML power measurement") from e
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def read_watts(self) -> float:
        
        return float(self.nvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0


class JetsonSysfsPowerSampler(PowerSampler):
   
    def __init__(self, power_path: str, scale_to_watts: float = 1e-3):
        self.path = Path(power_path)
        self.scale = scale_to_watts
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def read_watts(self) -> float:
        return float(self.path.read_text().strip()) * self.scale


def integrate_energy(
    fn: Callable[[], None],
    sampler: PowerSampler,
    hz: float = 50.0,
    synchronize: Optional[Callable[[], None]] = None,
) -> EnergyMeasurement:
    """Integrate sampled power over one end-to-end request using trapezoids."""
    period = 1.0 / hz
    stop = Event()
    samples = []

    def worker():
        while not stop.is_set():
            samples.append((perf_counter(), sampler.read_watts()))
            sleep(period)

    if synchronize:
        synchronize()
    t0 = perf_counter()
    th = Thread(target=worker, daemon=True)
    th.start()
    fn()
    if synchronize:
        synchronize()
    t1 = perf_counter()
    stop.set()
    th.join()


    if not samples:
        p = sampler.read_watts()
        samples = [(t0, p), (t1, p)]
    else:
        samples.insert(0, (t0, samples[0][1]))
        samples.append((t1, samples[-1][1]))

    energy = 0.0
    for (ta, pa), (tb, pb) in zip(samples[:-1], samples[1:]):
        a, b = max(t0, ta), min(t1, tb)
        if b > a:
            energy += 0.5 * (pa + pb) * (b - a)
    mean_watts = energy / max(1e-12, t1 - t0)
    return EnergyMeasurement(energy, t1 - t0, len(samples), mean_watts)
