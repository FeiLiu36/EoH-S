# Top 10 functions for reevo run 1

# Function 1 - Score: -0.03834987489345694
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_mask = remaining_space >= 0
    
    if not np.any(valid_mask):
        return np.zeros_like(bins)
    
    # Moderate exact fit boost
    exact_fit = (remaining_space == 0).astype(float) * 1e4
    
    # Adaptive exponent based on global fill ratio
    residuals = remaining_space[valid_mask] / bins[valid_mask]
    global_fill = 1 - np.mean(residuals)
    exponent = 3 + 8 * global_fill  # Smoothly varies from 3 to 11
    
    # Core penalty terms
    tight_fit = 1 / (1e-6 + residuals**exponent)
    slack_penalty = 1 / (1e-6 + (1 - residuals)**exponent)
    
    # Gradual blending with optimized transition
    transition_point = 0.6 + 0.2 * global_fill  # Adaptive transition
    blend_weight = 1 / (1 + np.exp(-15*(global_fill-transition_point)))
    
    # Simple harmonic mean
    harmonic = (2 * tight_fit * slack_penalty) / (1e-6 + tight_fit + slack_penalty)
    
    # Final priorities
    priorities = np.zeros_like(bins)
    priorities[valid_mask] = (
        exact_fit[valid_mask] +
        blend_weight * tight_fit +
        (1 - blend_weight) * harmonic
    )
    
    return priorities



# Function 2 - Score: -0.03837741543956535
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_mask = remaining_space >= 0
    
    if not np.any(valid_mask):
        return np.zeros_like(bins)
    
    # Absolute dominance for exact fits (1e9 boost)
    exact_fit = (remaining_space == 0).astype(float) * 1e9
    
    # Core metrics with septic penalties (steeper than quintic)
    residuals = remaining_space[valid_mask] / bins[valid_mask]
    tight_fit = 1 / (1e-14 + residuals**7)  # Septic penalty
    slack_penalty = 1 / (1e-14 + (1 - residuals)**7)  # Septic penalty
    
    # Near-step weight transition (very sharp sigmoid)
    global_fill = 1 - np.mean(residuals)
    tight_weight = 1 / (1 + np.exp(-30*(global_fill-0.6)))  # Near-step transition
    slack_weight = 1 - tight_weight
    
    # Pure harmonic combination with tighter epsilon
    combined = (2 * tight_fit * slack_penalty) / (1e-14 + tight_fit + slack_penalty)
    
    # Final priorities with absolute exact fit dominance
    priorities = np.zeros_like(bins)
    priorities[valid_mask] = (
        exact_fit[valid_mask] +
        tight_weight * tight_fit +
        slack_weight * combined
    )
    
    return priorities



# Function 3 - Score: -0.03847229844015503
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_mask = remaining_space >= 0
    
    if not np.any(valid_mask):
        return np.zeros_like(bins)
    
    # Moderate exact-fit boost (1e5 multiplier)
    exact_fit = (remaining_space == 0).astype(float) * 1e5
    
    # Core metrics with cubic/quartic penalties
    residuals = remaining_space[valid_mask] / bins[valid_mask]
    tight_fit = 1 / (1e-12 + residuals**4)  # Quartic penalty
    slack_penalty = 1 / (1e-12 + (1 - residuals)**3)  # Cubic penalty
    
    # Smoother sigmoid transition (¦Å=1e-12, threshold ~0.65)
    global_fill = 1 - np.mean(residuals)
    tight_weight = 1 / (1 + np.exp(-15*(global_fill-0.65)))  # Smoother transition
    
    # Harmonic blending with balanced weights
    combined = (2 * tight_fit * slack_penalty) / (1e-12 + tight_fit + slack_penalty)
    
    # Final priorities with exact-fit dominance
    priorities = np.zeros_like(bins)
    priorities[valid_mask] = (
        exact_fit[valid_mask] +
        tight_weight * tight_fit +
        (1 - tight_weight) * combined
    )
    
    return priorities



# Function 4 - Score: -0.03847229844015503
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_mask = remaining_space >= 0
    
    if not np.any(valid_mask):
        return np.zeros_like(bins)
    
    # Strong but not extreme boost for exact fits (1e5 multiplier)
    exact_fit = (remaining_space == 0).astype(float) * 1e5
    
    # Core metrics with moderate penalties (quartic/cubic)
    residuals = remaining_space[valid_mask] / bins[valid_mask]
    tight_fit = 1 / (1e-12 + residuals**4)  # Quartic penalty
    slack_penalty = 1 / (1e-12 + (1 - residuals)**3)  # Cubic penalty
    
    # Smoother sigmoid transition (¦Å=1e-12, threshold ~0.65)
    global_fill = 1 - np.mean(residuals)
    tight_weight = 1 / (1 + np.exp(-15*(global_fill-0.65)))  # Smoother transition
    slack_weight = 1 - tight_weight
    
    # Harmonic blending with balanced weights
    combined = (2 * tight_fit * slack_penalty) / (1e-12 + tight_fit + slack_penalty)
    
    # Final priorities with exact-fit dominance
    priorities = np.zeros_like(bins)
    priorities[valid_mask] = (
        exact_fit[valid_mask] +
        tight_weight * tight_fit +
        slack_weight * combined
    )
    
    return priorities



# Function 5 - Score: -0.03847229844015503
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_mask = remaining_space >= 0
    
    if not np.any(valid_mask):
        return np.zeros_like(bins)
    
    # Moderate exact fit boost (1e5 multiplier)
    exact_fit = (remaining_space == 0).astype(float) * 1e5
    
    # Core metrics with cubic/quartic penalties
    residuals = remaining_space[valid_mask] / bins[valid_mask]
    tight_fit = 1 / (1e-12 + residuals**4)  # Quartic penalty
    slack_penalty = 1 / (1e-12 + (1 - residuals)**3)  # Cubic penalty
    
    # Smoother transition (sigmoid with ¦Å=1e-12, threshold ~0.65)
    global_fill = 1 - np.mean(residuals)
    tight_weight = 1 / (1 + np.exp(-15*(global_fill-0.65)))  # Smoother transition
    slack_weight = 1 - tight_weight
    
    # Harmonic blending with balanced weights
    combined = (2 * tight_fit * slack_penalty) / (1e-12 + tight_fit + slack_penalty)
    
    # Final priorities with exact fit dominance
    priorities = np.zeros_like(bins)
    priorities[valid_mask] = (
        exact_fit[valid_mask] +
        tight_weight * tight_fit +
        slack_weight * combined
    )
    
    return priorities



# Function 6 - Score: -0.03847229844015503
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_mask = remaining_space >= 0
    
    if not np.any(valid_mask):
        return np.zeros_like(bins)
    
    # Stronger exact fit boost (quartic range)
    exact_fit = (remaining_space == 0).astype(float) * 1e5
    
    # Fixed moderate exponents (quartic/cubic)
    residuals = remaining_space[valid_mask] / bins[valid_mask]
    tight_penalty = 1 / (1e-12 + residuals**4)  # Quartic penalty
    slack_penalty = 1 / (1e-12 + (1 - residuals)**3)  # Cubic penalty
    
    # Sigmoid transition with fixed threshold
    transition_threshold = 0.65
    global_fill = 1 - np.mean(residuals)
    blend_weight = 1 / (1 + np.exp(-15*(global_fill - transition_threshold)))
    
    # Harmonic blending
    harmonic = (2 * tight_penalty * slack_penalty) / (1e-12 + tight_penalty + slack_penalty)
    
    # Final priorities with balanced blending
    priorities = np.zeros_like(bins)
    priorities[valid_mask] = (
        exact_fit[valid_mask] +
        blend_weight * tight_penalty +
        (1 - blend_weight) * harmonic
    )
    
    return priorities



# Function 7 - Score: -0.03855655932468689
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_mask = remaining_space >= 0
    
    if not np.any(valid_mask):
        return np.zeros_like(bins)
    
    # Exact fit gets absolute priority (1e6+ boost)
    exact_fit = (remaining_space == 0).astype(float) * 1e6
    
    # Dynamic exponent based on global fill ratio
    global_fill = 1 - np.mean(remaining_space[valid_mask] / bins[valid_mask])
    exponent = 4 + int(6 * global_fill)  # Ranges from 4 to 10
    
    # Core penalty terms with dynamic exponent
    residuals = remaining_space[valid_mask] / bins[valid_mask]
    tight_fit = 1 / (1e-8 + residuals**exponent)
    slack_penalty = 1 / (1e-8 + (1 - residuals)**exponent)
    
    # Sigmoid-weighted blending (sharp transition at 0.65 fill)
    blend_weight = 1 / (1 + np.exp(-20*(global_fill-0.65)))
    
    # Harmonic mean blending
    harmonic = (2 * tight_fit * slack_penalty) / (1e-8 + tight_fit + slack_penalty)
    
    # Final priorities
    priorities = np.zeros_like(bins)
    priorities[valid_mask] = (
        exact_fit[valid_mask] +
        blend_weight * tight_fit +
        (1 - blend_weight) * harmonic
    )
    
    return priorities



# Function 8 - Score: -0.038671904898955084
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_mask = remaining_space >= 0
    
    if not np.any(valid_mask):
        return np.zeros_like(bins)
    
    # Absolute dominance for exact fits (1e12 boost)
    exact_fit = (remaining_space == 0).astype(float) * 1e12
    
    # Ultra-steep nonic penalties (9th power)
    residuals = remaining_space[valid_mask] / bins[valid_mask]
    tight_fit = 1 / (1e-16 + residuals**9)  # Nonic penalty
    slack_penalty = 1 / (1e-16 + (1 - residuals)**9)  # Nonic penalty
    
    # Step-like transition (very sharp sigmoid)
    global_fill = 1 - np.mean(residuals)
    tight_weight = 1 / (1 + np.exp(-50*(global_fill-0.55)))  # Near-step transition
    
    # Pure harmonic combination with minimal epsilon
    combined = (2 * tight_fit * slack_penalty) / (1e-16 + tight_fit + slack_penalty)
    
    # Final priorities with exact fit dominance
    priorities = np.zeros_like(bins)
    priorities[valid_mask] = (
        exact_fit[valid_mask] +
        tight_weight * tight_fit +
        (1 - tight_weight) * combined
    )
    
    return priorities



# Function 9 - Score: -0.038671904898955084
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_mask = remaining_space >= 0
    
    if not np.any(valid_mask):
        return np.zeros_like(bins)
    
    # Extreme exact-fit reward (1e9+ boost)
    exact_fit = (remaining_space <= 1e-14).astype(float) * 1e18
    
    # Ultra-tight epsilon (1e-14) with septic+ penalties (x^9)
    residuals = remaining_space[valid_mask] / bins[valid_mask]
    tight_fit = 1 / (1e-14 + residuals**9)  # Septic+ penalty
    slack_penalty = 1 / (1e-14 + (1 - residuals)**9)  # Septic+ penalty
    
    # Dynamic tight/slack balance with ultra-sharp transition (k=50)
    global_fill = 1 - np.mean(residuals)
    tight_weight = 1 / (1 + np.exp(-50*(global_fill-0.55)))  # Ultra-sharp transition
    
    # Aggressive adaptation: pure min(tight,slack) when tight_weight > 0.99
    combined = np.where(tight_weight > 0.99,
                       np.minimum(tight_fit, slack_penalty),
                       (2 * tight_fit * slack_penalty) / (1e-14 + tight_fit + slack_penalty))
    
    # Final priorities with exact fit dominance
    priorities = np.zeros_like(bins)
    priorities[valid_mask] = (
        exact_fit[valid_mask] +
        tight_weight * tight_fit +
        (1 - tight_weight) * combined
    )
    
    return priorities



# Function 10 - Score: -0.038671904898955084
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    valid_mask = remaining_space >= 0
    
    if not np.any(valid_mask):
        return np.zeros_like(bins)
    
    # Absolute dominance for exact fits (1e12 boost)
    exact_fit = (remaining_space == 0).astype(float) * 1e12
    
    # Core metrics with nonic penalties (steeper than septic)
    residuals = remaining_space[valid_mask] / bins[valid_mask]
    tight_fit = 1 / (1e-16 + residuals**9)  # Nonic penalty
    slack_penalty = 1 / (1e-16 + (1 - residuals)**9)  # Nonic penalty
    
    # Step-like weight transition (extremely sharp sigmoid)
    global_fill = 1 - np.mean(residuals)
    tight_weight = 1 / (1 + np.exp(-50*(global_fill-0.55)))  # Near-step transition
    slack_weight = 1 - tight_weight
    
    # Pure harmonic combination with minimal epsilon
    combined = (2 * tight_fit * slack_penalty) / (1e-16 + tight_fit + slack_penalty)
    
    # Final priorities with absolute exact fit dominance
    priorities = np.zeros_like(bins)
    priorities[valid_mask] = (
        exact_fit[valid_mask] +
        tight_weight * tight_fit +
        slack_weight * combined
    )
    
    return priorities



