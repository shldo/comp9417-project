from __future__ import annotations

import time
from typing import Any


def measure_training_time(train_fn, *args, **kwargs) -> tuple[Any, float]:
    """
    Measure training time for a callable.

    Returns:
    - callable result
    - elapsed time in seconds
    """
    start_time = time.perf_counter()
    result = train_fn(*args, **kwargs)
    elapsed_seconds = time.perf_counter() - start_time
    return result, elapsed_seconds


def measure_inference_time_per_sample(predict_fn, X) -> tuple[Any, float]:
    """
    Measure average inference time per sample for a callable.

    Returns:
    - callable result
    - average time per sample in seconds
    """
    num_samples = len(X)
    if num_samples == 0:
        raise ValueError("X must contain at least one sample.")

    start_time = time.perf_counter()
    predictions = predict_fn(X)
    elapsed_seconds = time.perf_counter() - start_time
    time_per_sample = elapsed_seconds / num_samples

    return predictions, time_per_sample
