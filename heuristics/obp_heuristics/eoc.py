import numpy as np
def priority(item, bins):
    scores = np.log(item) * (bins ** 2) / (item * np.sqrt(bins - item)) + (bins / item) ** 3
    scores[bins == bins.max()] = -np.inf
    return scores