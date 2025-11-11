import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:


    remaining = bins - item
    scores = (bins / remaining) - (np.arange(1, len(bins) + 1) * bins.mean() / item)

    # Degrade scores for bins that will exceed their capacity
    scores -= np.where(remaining < 0, bins + remaining, 0)

    # Adjust scores for bins that are almost full but closer to their maximum capacity
    scores += np.where((bins >= item) & (remaining <= bins.mean() / 2), bins - item, 0)

    # Give higher priority to bins close to the item size
    scores += np.abs(bins - item)

    # Adjust scores for bins that are already filled with the same item size
    scores -= np.where(bins == item, np.inf, 0)

    # More emphasis on bins that can hold multiple instances of the current item
    scores += np.where((bins - item) % item == 0, bins.max() * bins.size, 0)

    # Prefer bins that have a lot of space and can hold multiple instances of the item
    scores += bins / item * (bins // item)

    # Prefer bins that can hold an additional multiple of the item after adding the current item
    scores += bins / (bins + item)

    # Add a small random factor to the scores to avoid any deterministic bias
    scores += np.random.uniform(-0.01, 0.01, size=bins.shape)

    return scores