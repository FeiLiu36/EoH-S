import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    factor = 0.5 or np.multiply.reduce(bins) /np.max(bins)
    penalty = np.arange(len(bins), 0, -1)
    sizes = bins - item
    scores = factor * bins + sizes / sizes + penalty
    return scores