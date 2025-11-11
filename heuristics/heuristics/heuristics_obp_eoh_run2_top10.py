# Top 10 functions for eoh run 2

# Function 1 - Score: -0.03945918882675462
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid = remaining >= 0
    fill_ratio = np.where(bins > 0, item / bins, 0)
    remaining_norm = np.where(bins > 0, remaining / np.max(bins), 0)
    weight = 1.0 - (bins / np.max(bins)) ** 2
    priority_scores = np.where(valid, fill_ratio * 0.3 + remaining_norm * 0.7 * weight, -np.inf)
    return priority_scores



# Function 2 - Score: -0.03945918882675462
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid = remaining >= 0
    fill_ratio = np.where(bins > 0, item / bins, 0)
    remaining_norm = np.where(bins > 0, remaining / np.max(bins), 0)
    weight = 1.0 - (bins / np.max(bins)) ** 2
    priority_scores = np.where(valid, fill_ratio * 0.3 + remaining_norm * 0.7 * weight, -np.inf)
    return priority_scores



# Function 3 - Score: -0.03945918882675462
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid = remaining >= 0
    fill_ratio = np.where(bins > 0, item / bins, 0)
    remaining_norm = np.where(bins > 0, remaining / np.max(bins), 0)
    weight = (bins / np.max(bins)) ** 2
    priority_scores = np.where(valid, fill_ratio * 0.3 + remaining_norm * 0.7 * (1 - weight), -np.inf)
    return priority_scores



# Function 4 - Score: -0.03945918882675462
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid = remaining >= 0
    fill_ratio = np.where(bins > 0, item / bins, 0)
    remaining_norm = np.where(bins > 0, remaining / np.max(bins), 0)
    weight = 1.0 - (bins / np.max(bins)) ** 2
    priority_scores = np.where(valid, fill_ratio * 0.3 + remaining_norm * 0.7 * weight, -np.inf)
    return priority_scores



# Function 5 - Score: -0.03945918882675462
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid = remaining >= 0
    fill_ratio = np.where(bins > 0, item / bins, 0)
    remaining_norm = np.where(bins > 0, remaining / np.max(bins), 0)
    weight = 1.0 - (bins / np.max(bins)) ** 2
    priority_scores = np.where(valid, fill_ratio * 0.3 + remaining_norm * 0.7 * weight, -np.inf)
    return priority_scores



# Function 6 - Score: -0.03976079047298105
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid = remaining >= 0
    dynamic_fraction = 0.2 * (1 + np.log1p(bins))
    target = dynamic_fraction * remaining
    priority_scores = np.where(valid, -np.abs(remaining - target), -np.inf)
    return priority_scores



# Function 7 - Score: -0.03980559162933113
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid = remaining >= 0
    remaining_ratio = np.where(bins > 0, remaining / bins, np.inf)
    penalty = np.where(bins < item * 1.5, 0.5, 1.0)
    priority_scores = np.where(valid, penalty * np.exp(-remaining_ratio * 5), -np.inf)
    return priority_scores



# Function 8 - Score: -0.039836995360891886
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid = remaining >= 0
    fill_ratio = np.where(bins > 0, item / bins, 0)
    remaining_norm = np.where(bins > 0, remaining / np.max(bins), 0)
    weight = 1.0 - (bins / np.max(bins))
    priority_scores = np.where(valid, fill_ratio * 0.4 + remaining_norm * 0.6 * weight, -np.inf)
    return priority_scores



# Function 9 - Score: -0.039836995360891886
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid = remaining >= 0
    fill_ratio = np.where(bins > 0, item / bins, 0)
    remaining_norm = np.where(bins > 0, remaining / np.max(bins), 0)
    weight = (1.0 - (bins / np.max(bins))) ** 2
    priority_scores = np.where(valid, fill_ratio * 0.3 + remaining_norm * 0.7 * weight, -np.inf)
    return priority_scores



# Function 10 - Score: -0.039836995360891886
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid = remaining >= 0
    fill_ratio = np.where(bins > 0, item / bins, 0)
    remaining_norm = np.where(bins > 0, remaining / np.max(bins), 0)
    log_weight = np.where(bins > 0, 1.0 - np.log1p(bins) / np.log1p(np.max(bins)), 0)
    priority_scores = np.where(valid, (fill_ratio ** 2) * 0.3 + remaining_norm * 0.7 * log_weight, -np.inf)
    return priority_scores



