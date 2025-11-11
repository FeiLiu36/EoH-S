# Top 10 functions for reevo run 3

# Function 1 - Score: -0.03914495329595862
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_bins = remaining_space >= 0
    
    # Adaptive scaling with softer parameters
    size_factor = np.log1p(item)
    bin_mean = np.mean(bins)
    bin_std = max(np.std(bins), 0.1*bin_mean, 1e-8)
    
    # Ultra-gentle comfort threshold
    comfort_thresh = max(0.08 * item, 0.01)
    
    # Base priority: ultra-smooth logarithmic function
    base_priority = np.where(
        valid_bins,
        np.log1p(remaining_space) * (1 + 0.01 * np.tanh(remaining_space/(3*item))),
        -np.inf
    )
    
    # Smoother comfort bonus with gradual transition
    comfort_ratio = np.clip((remaining_space - comfort_thresh) / (comfort_thresh + 1e-8), 0, 1)
    comfort_bonus = 0.4 * size_factor * (1 - np.exp(-8 * comfort_ratio))
    base_priority += comfort_bonus
    
    # Ultra-gentle lookahead with balanced decay
    lookahead_factor = min(0.4 + 0.6 * np.log1p(item), 1.0)
    lookahead_space = remaining_space - lookahead_factor*item
    future_score = np.where(
        lookahead_space > 0,
        0.3 * size_factor * (1 - np.exp(-lookahead_space/(1.2*item))),
        -0.8 * size_factor * np.exp(2 * lookahead_space/(0.5*item))
    )
    base_priority += future_score
    
    # Minimal load balancing effect
    load_imbalance = (bins - bin_mean) / bin_std
    balance_factor = 0.008 * size_factor
    base_priority -= balance_factor * np.tanh(0.2 * load_imbalance)
    
    # Final very mild logarithmic scaling
    priority = base_priority * (1 + 0.04 * np.log1p(item))
    return np.where(valid_bins, -priority, -np.inf)



# Function 2 - Score: -0.03915210179804532
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_bins = remaining_space >= 0
    
    # Optimized dynamic parameters
    size_factor = np.log1p(item)
    bin_mean = np.mean(bins)
    bin_std = max(np.std(bins), 0.1*bin_mean, 1e-8)
    
    # Fine-tuned comfort threshold (28% with minimum floor)
    comfort_thresh = max(0.28 * item, 0.02)
    
    # Enhanced base priority with stronger curvature
    base_priority = np.where(
        valid_bins,
        np.log1p(remaining_space) * (1 + 0.25 * np.tanh(remaining_space/(1.8*item))),
        -np.inf
    )
    
    # More aggressive comfort bonus
    comfort_ratio = np.clip(remaining_space / comfort_thresh, 0, 2.5)
    comfort_bonus = 1.0 * size_factor * (1 - np.exp(-3 * comfort_ratio))
    base_priority += comfort_bonus
    
    # Stronger lookahead with optimized transitions
    lookahead_factor = min(0.75 + 0.25 * np.log1p(item), 1.0)
    lookahead_space = remaining_space - lookahead_factor*item
    
    future_score = np.where(
        lookahead_space > 0,
        0.7 * size_factor * (1 - np.exp(-lookahead_space/(0.8*item))),  # Faster reward decay
        -1.2 * size_factor * np.exp(lookahead_space/(0.2*item))         # Sharper penalty
    )
    base_priority += future_score
    
    # More aggressive load balancing
    load_imbalance = (bins - bin_mean) / bin_std
    balance_factor = 0.08 * size_factor * (1 - np.exp(-np.abs(load_imbalance)/2.5))
    base_priority -= balance_factor * np.tanh(2.0 * load_imbalance)
    
    # Final scaling with stronger size adaptation
    priority = base_priority * (1 + 0.12 * np.log1p(item + 0.03))
    return np.where(valid_bins, -priority, -np.inf)



# Function 3 - Score: -0.03916746638580355
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_bins = remaining_space >= 0
    
    # Dynamic scaling parameters
    size_factor = np.log1p(item)
    bin_mean = np.mean(bins)
    bin_std = max(np.std(bins), 0.1*bin_mean, 1e-8)
    
    # Adaptive threshold with lower base value
    comfortable_thresh = max(0.08 * item, 0.01)
    
    # Base priority: smooth concave function with gentler slope
    base_priority = np.where(
        valid_bins,
        np.sqrt(remaining_space + 1) * (1 + 0.05 * np.tanh(remaining_space/(3*item))),
        -np.inf
    )
    
    # Unified reward system with exponential decay
    comfort_ratio = np.clip((remaining_space - comfortable_thresh) / (comfortable_thresh + 1e-8), 0, 1)
    comfort_bonus = 0.7 * size_factor * (1 - np.exp(-4 * comfort_ratio))
    base_priority += comfort_bonus
    
    # Balanced lookahead with asymmetric decay
    lookahead_factor = min(0.4 + 0.6 * np.log1p(item), 1.0)
    lookahead_space = remaining_space - lookahead_factor*item
    future_score = np.where(
        lookahead_space > 0,
        0.5 * size_factor * (1 - np.exp(-lookahead_space/(0.6*item))),
        -1.2 * size_factor * np.exp(2 * lookahead_space/(0.3*item))
    )
    base_priority += future_score
    
    # Very gentle load balancing
    load_imbalance = (bins - bin_mean) / bin_std
    balance_factor = 0.02 * size_factor
    base_priority -= balance_factor * np.tanh(0.5 * load_imbalance)
    
    # Final scaling with logarithmic growth
    priority = base_priority * (1 + 0.08 * np.log1p(item))
    return np.where(valid_bins, -priority, -np.inf)



# Function 4 - Score: -0.039169877706064095
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_bins = remaining_space >= 0
    
    # Adaptive scaling with ultra-gentle parameters
    size_factor = np.log1p(item)
    bin_mean = np.mean(bins)
    bin_std = max(np.std(bins), 0.1*bin_mean, 1e-8)
    
    # Ultra-soft comfort threshold
    comfort_thresh = max(0.1 * item, 0.02)
    
    # Base priority: ultra-smooth logarithmic function with gentle adjustment
    base_priority = np.where(
        valid_bins,
        np.log1p(remaining_space) * (1 + 0.005 * np.tanh(remaining_space/(5*item))),
        -np.inf
    )
    
    # Ultra-gradual comfort bonus with smooth transition
    comfort_ratio = np.clip((remaining_space - comfort_thresh) / (comfort_thresh + 1e-8), 0, 1)
    comfort_bonus = 0.3 * size_factor * (1 - np.exp(-10 * comfort_ratio))
    base_priority += comfort_bonus
    
    # Balanced lookahead with ultra-gentle decay
    lookahead_factor = min(0.5 + 0.5 * np.log1p(item), 1.0)
    lookahead_space = remaining_space - lookahead_factor*item
    future_score = np.where(
        lookahead_space > 0,
        0.2 * size_factor * (1 - np.exp(-lookahead_space/(2*item))),
        -0.6 * size_factor * np.exp(3 * lookahead_space/(0.8*item))
    )
    base_priority += future_score
    
    # Minimal load balancing with ultra-gentle effect
    load_imbalance = (bins - bin_mean) / bin_std
    balance_factor = 0.005 * size_factor
    base_priority -= balance_factor * np.tanh(0.1 * load_imbalance)
    
    # Final ultra-mild logarithmic scaling
    priority = base_priority * (1 + 0.03 * np.log1p(item))
    return np.where(valid_bins, -priority, -np.inf)



# Function 5 - Score: -0.03918031218627439
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_bins = remaining_space >= 0
    
    # Adaptive parameters with logarithmic scaling
    size_factor = np.log1p(item)
    bin_mean = np.mean(bins)
    bin_std = max(np.std(bins), 0.1*bin_mean, 1e-8)
    
    # Dynamic comfort threshold with smooth minimum
    comfort_thresh = max(0.08 * item, 0.01)
    
    # Ultra-smooth base priority using log-sigmoid combination
    base_priority = np.where(
        valid_bins,
        np.log1p(remaining_space) * (1 + 0.01 / (1 + np.exp(-remaining_space/(item + 1e-8)))),
        -np.inf
    )
    
    # Sharp but smooth comfort bonus with adaptive transition
    comfort_ratio = np.clip((remaining_space - comfort_thresh) / (comfort_thresh + 1e-8), 0, 1)
    comfort_bonus = 0.5 * size_factor * (1 - np.exp(-8 * comfort_ratio))
    base_priority += comfort_bonus
    
    # Adaptive lookahead with item-size dependent decay
    lookahead_factor = min(0.4 + 0.6 * np.log1p(item + 0.1), 1.0)
    lookahead_space = remaining_space - lookahead_factor*item
    future_score = np.where(
        lookahead_space > 0,
        0.3 * size_factor * (1 - np.exp(-lookahead_space/(0.7*item))),
        -0.8 * size_factor * np.exp(2 * lookahead_space/(0.3*item))
    )
    base_priority += future_score
    
    # Minimal load balancing with ultra-gentle coefficients
    load_imbalance = (bins - bin_mean) / bin_std
    balance_factor = 0.01 * size_factor
    base_priority -= balance_factor * np.tanh(0.3 * load_imbalance)
    
    # Final logarithmic normalization
    priority = base_priority * (1 + 0.04 * np.log1p(item + 0.1))
    return np.where(valid_bins, -priority, -np.inf)



# Function 6 - Score: -0.03918482838834179
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_bins = remaining_space >= 0
    
    # Adaptive scaling parameters
    size_factor = np.log1p(item)
    bin_mean = np.mean(bins)
    bin_std = max(np.std(bins), 0.1*bin_mean, 1e-8)
    
    # Dynamic comfort threshold based on item size
    comfort_thresh = max(0.1 * item, 0.02)
    
    # Base priority: smooth natural log function with gentle slope
    base_priority = np.where(
        valid_bins,
        np.log1p(remaining_space) * (1 + 0.03 * np.tanh(remaining_space/(2*item))),
        -np.inf
    )
    
    # Exponential comfort bonus with smooth transition
    comfort_ratio = np.clip((remaining_space - comfort_thresh) / (comfort_thresh + 1e-8), 0, 1)
    comfort_bonus = 0.6 * size_factor * (1 - np.exp(-5 * comfort_ratio))
    base_priority += comfort_bonus
    
    # Balanced lookahead with smooth exponential decay
    lookahead_factor = min(0.5 + 0.5 * np.log1p(item), 1.0)
    lookahead_space = remaining_space - lookahead_factor*item
    future_score = np.where(
        lookahead_space > 0,
        0.4 * size_factor * (1 - np.exp(-lookahead_space/(0.8*item))),
        -1.0 * size_factor * np.exp(1.5 * lookahead_space/(0.4*item))
    )
    base_priority += future_score
    
    # Very mild load balancing
    load_imbalance = (bins - bin_mean) / bin_std
    balance_factor = 0.015 * size_factor
    base_priority -= balance_factor * np.tanh(0.4 * load_imbalance)
    
    # Final logarithmic scaling
    priority = base_priority * (1 + 0.06 * np.log1p(item))
    return np.where(valid_bins, -priority, -np.inf)



# Function 7 - Score: -0.03919017569319545
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_bins = remaining_space >= 0
    
    # Dynamic parameters with adaptive scaling
    size_factor = np.log1p(item)
    bin_mean = np.mean(bins)
    bin_std = max(np.std(bins), 0.1*bin_mean, 1e-8)
    
    # Unified comfort threshold with adaptive minimum
    comfort_thresh = max(0.3 * item, 0.02)
    
    # Smoother base priority using log-tanh combination
    base_priority = np.where(
        valid_bins,
        np.log1p(remaining_space) * (1 + 0.2 * np.tanh(remaining_space/(2*item))),
        -np.inf
    )
    
    # Single-zone reward system with exponential decay
    comfort_ratio = np.clip(remaining_space / comfort_thresh, 0, 3)
    comfort_bonus = 0.8 * size_factor * (1 - np.exp(-1.5 * comfort_ratio))
    base_priority += comfort_bonus
    
    # Balanced lookahead with adaptive decay
    lookahead_factor = min(0.7 + 0.3 * np.log1p(item), 1.0)
    lookahead_space = remaining_space - lookahead_factor*item
    future_score = np.where(
        lookahead_space > 0,
        0.6 * size_factor * (1 - np.exp(-lookahead_space/(0.8*item))),
        -1.0 * size_factor * np.exp(lookahead_space/(0.4*item))
    )
    base_priority += future_score
    
    # Adaptive load balancing with tanh smoothing
    load_imbalance = (bins - bin_mean) / bin_std
    balance_factor = 0.03 * size_factor * (1 - np.exp(-np.abs(load_imbalance)/3))
    base_priority -= balance_factor * np.tanh(load_imbalance)
    
    # Final scaling with size-adaptive weight
    priority = base_priority * (1 + 0.1 * np.log1p(item + 0.2))
    return np.where(valid_bins, -priority, -np.inf)



# Function 8 - Score: -0.03919017569319545
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_bins = remaining_space >= 0
    
    # Dynamic parameters with adaptive scaling
    size_factor = np.log1p(item)
    bin_mean = np.mean(bins)
    bin_std = max(np.std(bins), 0.1*bin_mean, 1e-8)
    
    # Unified comfort threshold with adaptive minimum
    comfort_thresh = max(0.25 * item, 0.01)
    
    # Base priority with smooth logarithmic scaling
    base_priority = np.where(
        valid_bins,
        np.log1p(remaining_space) * (1 + 0.15 * np.tanh(remaining_space/(1.5*item))),
        -np.inf
    )
    
    # Unified reward system with smooth exponential transition
    space_ratio = remaining_space / comfort_thresh
    reward = 0.7 * size_factor * (1 - np.exp(-1.8 * np.clip(space_ratio, 0, 2)))
    base_priority += reward
    
    # Balanced lookahead with adaptive decay
    lookahead_factor = min(0.6 + 0.4 * np.log1p(item + 0.5), 1.0)
    lookahead_space = remaining_space - lookahead_factor*item
    future_score = np.where(
        lookahead_space > 0,
        0.5 * size_factor * (1 - np.exp(-lookahead_space/(0.7*item))),
        -0.8 * size_factor * np.exp(lookahead_space/(0.3*item))
    )
    base_priority += future_score
    
    # Load balancing with smooth tanh transition
    load_imbalance = (bins - bin_mean) / bin_std
    balance_factor = 0.02 * size_factor * np.exp(-np.abs(load_imbalance)/2)
    base_priority -= balance_factor * np.tanh(load_imbalance)
    
    # Final scaling with adaptive weight
    priority = base_priority * (1 + 0.08 * np.log1p(item + 0.1))
    return np.where(valid_bins, -priority, -np.inf)



# Function 9 - Score: -0.03920412122086044
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_bins = remaining_space >= 0
    
    # Simplified dynamic scaling
    size_factor = np.log1p(item)
    bin_mean = np.mean(bins)
    bin_std = max(np.std(bins), 0.1*bin_mean, 1e-8)
    
    # Minimal comfort threshold
    comfort_thresh = max(0.05 * item, 0.005)
    
    # Smoother base priority with cubic root
    base_priority = np.where(
        valid_bins,
        np.cbrt(remaining_space + 1) * (1 + 0.02 * np.tanh(remaining_space/(4*item))),
        -np.inf
    )
    
    # Tight comfort ratio with linear decay
    comfort_ratio = np.clip((remaining_space - comfort_thresh) / (comfort_thresh + 1e-8), 0, 1)
    comfort_bonus = 0.5 * size_factor * comfort_ratio
    base_priority += comfort_bonus
    
    # Balanced lookahead with minimal penalty
    lookahead_factor = min(0.3 + 0.7 * np.log1p(item), 1.0)
    lookahead_space = remaining_space - lookahead_factor*item
    future_score = np.where(
        lookahead_space > 0,
        0.4 * size_factor * (1 - np.exp(-lookahead_space/(0.5*item))),
        -0.8 * size_factor * np.exp(lookahead_space/(0.2*item))
    )
    base_priority += future_score
    
    # Minimal load balancing
    load_imbalance = (bins - bin_mean) / bin_std
    balance_factor = 0.01 * size_factor
    base_priority -= balance_factor * np.tanh(0.3 * load_imbalance)
    
    # Final scaling with minimal growth
    priority = base_priority * (1 + 0.05 * np.log1p(item))
    return np.where(valid_bins, -priority, -np.inf)



# Function 10 - Score: -0.039204459166321225
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_bins = remaining_space >= 0
    
    # Adaptive parameters with softer scaling
    size_factor = np.log1p(item)
    bin_mean = np.mean(bins)
    bin_std = max(np.std(bins), 0.1*bin_mean, 1e-8)
    
    # Dynamic comfort threshold with tighter bounds
    comfort_thresh = max(0.08 * item, 0.01)
    
    # Base priority: ultra-smooth logarithmic function
    base_priority = np.where(
        valid_bins,
        np.log1p(remaining_space) * (1 + 0.01 * np.tanh(remaining_space/(3*item))),
        -np.inf
    )
    
    # Sharper but smoother comfort bonus
    comfort_ratio = np.clip((remaining_space - comfort_thresh) / (comfort_thresh + 1e-8), 0, 1)
    comfort_bonus = 0.4 * size_factor * (1 - np.exp(-8 * comfort_ratio))
    base_priority += comfort_bonus
    
    # Balanced lookahead with gentler exponential decay
    lookahead_factor = min(0.4 + 0.6 * np.log1p(item), 1.0)
    lookahead_space = remaining_space - lookahead_factor*item
    future_score = np.where(
        lookahead_space > 0,
        0.3 * size_factor * (1 - np.exp(-lookahead_space/(1.0*item))),
        -0.8 * size_factor * np.exp(1.2 * lookahead_space/(0.5*item))
    )
    base_priority += future_score
    
    # Minimal load balancing with ultra-smooth tanh
    load_imbalance = (bins - bin_mean) / bin_std
    balance_factor = 0.008 * size_factor
    base_priority -= balance_factor * np.tanh(0.3 * load_imbalance)
    
    # Final scaling with logarithmic damping
    priority = base_priority * (1 + 0.04 * np.log1p(item))
    return np.where(valid_bins, -priority, -np.inf)



