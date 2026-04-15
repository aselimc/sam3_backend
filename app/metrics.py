from __future__ import annotations

from prometheus_client import Gauge, Histogram

inference_seconds = Histogram(
    "sam3_inference_seconds",
    "Wall-clock duration of SAM3 inference (_run_queries).",
    labelnames=("endpoint",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)

gpu_mem_allocated_bytes = Gauge(
    "sam3_gpu_mem_allocated_bytes",
    "Current CUDA memory allocated (bytes).",
)

gpu_mem_peak_bytes = Gauge(
    "sam3_gpu_mem_peak_bytes",
    "Peak CUDA memory allocated since last reset (bytes).",
)
