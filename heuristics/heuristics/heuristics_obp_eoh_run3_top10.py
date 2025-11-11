# Top 10 functions for eoh run 3

# Function 1 - Score: -0.0396336870566258
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    mask = remaining >= 0
    priorities = np.zeros_like(bins)
    if np.any(mask):
        priorities[mask] = np.log1p(bins[mask]) ** 2 / (remaining[mask] + 1e-10)
    return priorities



# Function 2 - Score: -0.03963454021178764
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    mask = remaining >= 0
    priorities = np.zeros_like(bins)
    if np.any(mask):
        temperature = 0.5  # Tunable parameter
        priorities[mask] = item * np.exp(-remaining[mask] / temperature)
    return priorities



# Function 3 - Score: -0.03966188880762168
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    mask = remaining >= 0
    priorities = np.zeros_like(bins)
    if np.any(mask):
        ratio = bins[mask] / (remaining[mask] + 1e-10)
        priorities[mask] = bins[mask] * (1 / (1 + np.exp(remaining[mask]))) * np.log(ratio + 1)
    return priorities



# Function 4 - Score: -0.039670226587771085
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    mask = remaining >= 0
    priorities = np.zeros_like(bins)
    if np.any(mask):
        ratio = bins[mask] / (remaining[mask] + 1e-10)
        priorities[mask] = bins[mask] * np.exp(-remaining[mask]) * np.power(ratio, 1/3)
    return priorities



# Function 5 - Score: -0.03967761422800768
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    mask = remaining >= 0
    priorities = np.zeros_like(bins)
    if np.any(mask):
        priorities[mask] = item / (1 + np.log(1 + remaining[mask]))
    return priorities



# Function 6 - Score: -0.03967964448943986
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    mask = remaining >= 0
    priorities = np.zeros_like(bins)
    if np.any(mask):
        ratio = bins[mask] / (remaining[mask] + 1e-10)
        priorities[mask] = bins[mask] * np.exp(-remaining[mask]) * np.sqrt(ratio) * (1 - 0.1 * remaining[mask])
    return priorities



# Function 7 - Score: -0.03967964448943986
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    mask = remaining >= 0
    priorities = np.zeros_like(bins)
    if np.any(mask):
        ratio = bins[mask] / (remaining[mask] + 1e-10)
        priorities[mask] = bins[mask] * np.exp(-remaining[mask]) * np.sqrt(ratio) * (1 - 0.1 * remaining[mask])
    return priorities



# Function 8 - Score: -0.03968367543191342
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    mask = remaining >= 0
    priorities = np.zeros_like(bins)
    if np.any(mask):
        priorities[mask] = np.sqrt(bins[mask]) * np.exp(-remaining[mask])
    return priorities



# Function 9 - Score: -0.03968367543191342
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    mask = remaining >= 0
    priorities = np.zeros_like(bins)
    if np.any(mask):
        priorities[mask] = np.sqrt(bins[mask]) * np.exp(-remaining[mask])
    return priorities



# Function 10 - Score: -0.039692410582158014
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    mask = remaining >= 0
    priorities = np.zeros_like(bins)
    if np.any(mask):
        priorities[mask] = item * np.exp(-remaining[mask] ** 2)
    return priorities



