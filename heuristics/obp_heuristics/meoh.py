import numpy as np

def priority(item, bins):
    differences = np.abs(bins - item)
    penalties = np.where(bins < 1.5 * item, -0.5, 0)
    rewards = np.where(bins > 3 * item, 0.4, 0)
    scores = np.exp(-0.5 * (differences ** 2)) + penalties + rewards
    return scores