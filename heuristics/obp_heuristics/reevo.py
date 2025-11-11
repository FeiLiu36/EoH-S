import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Improved priority function for Online Bin Packing Problem emphasizing fairness, balance, adaptability, controlled randomness, and simplicity.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of remaining capacities for each bin.

    Return:
        Array of the same size as bins with optimized priority score for each bin.
    """
    utilization = (bins.max() - bins) / bins.max()
    remaining_capacities = bins - item
    item_ratio = item / bins

    balanced_item_ratios = utilization * (remaining_capacities - item) / item_ratio

    bin_ratio = bins / (bins - item + 1e-6)
    bin_ratio[remaining_capacities < 0] = 0

    fair_penalty = np.abs(0.5 - item_ratio) * (1 - utilization) * np.sqrt(bins.size)
    dynamic_penalty = np.sqrt(bins.size) * (1 - utilization) * np.abs(np.tanh(item_ratio - 0.5))
    adaptability_penalty = 0.1 * (bins.mean() - bins) * utilization

    balance_factor = 1 - np.std(bins) / np.mean(bins)
    adaptability_penalty *= balance_factor

    scores = bin_ratio + balanced_item_ratios - fair_penalty - dynamic_penalty + adaptability_penalty
    # Introduce controlled randomness with smaller weight for better adaptability
    scores += 0.1 * np.random.rand(*scores.shape)

    return scores