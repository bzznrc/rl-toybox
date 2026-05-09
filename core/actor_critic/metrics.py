"""Small actor-critic metric helpers."""

from __future__ import annotations

import numpy as np


def explained_variance(returns: object, values: object, *, variance_epsilon: float = 1e-8) -> float:
    """Return critic explained variance over flattened return/value batches."""
    return_array = np.asarray(returns, dtype=np.float32).reshape(-1)
    value_array = np.asarray(values, dtype=np.float32).reshape(-1)
    if return_array.size == 0 or value_array.size == 0 or return_array.size != value_array.size:
        return 0.0

    return_variance = float(np.var(return_array))
    if return_variance <= float(variance_epsilon):
        return 0.0
    residual_variance = float(np.var(return_array - value_array))
    return float(1.0 - (residual_variance / return_variance))
