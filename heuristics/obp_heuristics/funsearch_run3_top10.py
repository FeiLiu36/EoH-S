# Top 10 functions for funsearch run 3

# Function 1 - Score: -0.038956363623731
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    EPS = 1e-10
    valid_mask = bins >= item
    priorities = np.full_like(bins, -np.inf)
    
    if not np.any(valid_mask):
        return priorities
    
    valid_bins = bins[valid_mask]
    remaining = valid_bins - item
    normalized_remaining = remaining / (valid_bins + EPS)
    utilization = 1 - normalized_remaining
    fit_ratio = item / valid_bins
    
    # 1. Enhanced exact fit detection with multiple tolerance levels
    exact_fit_bonus = np.zeros_like(valid_bins)
    tol_levels = [1e-6, 1e-4, 1e-3, 1e-2]
    bonus_levels = [20.0, 15.0, 10.0, 5.0]
    for tol, bonus in zip(tol_levels, bonus_levels):
        exact_fit_bonus = np.maximum(exact_fit_bonus, 
                                    bonus * np.exp(-(normalized_remaining**2)/tol))
    
    # 2. Piecewise tightness priority
    tightness_priority = np.where(
        normalized_remaining < 0.01,
        5.0,  # Highest priority for extremely tight fits
        np.where(
            normalized_remaining < 0.05,
            4.0 - 20.0 * normalized_remaining,  # Strong preference
            np.where(
                normalized_remaining < 0.2,
                2.0 - 5.0 * normalized_remaining,  # Moderate preference
                1.0 - 2.0 * normalized_remaining  # Mild preference
            )
        )
    )
    
    # 3. Utilization incentive with sigmoid curve
    utilization_priority = 3.0 / (1 + np.exp(-12.0 * (utilization - 0.65)))
    
    # 4. Future packing potential with size distribution awareness
    common_sizes = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    remaining_rep = remaining.reshape(-1, 1)
    mod_results = np.min(remaining_rep % common_sizes, axis=1) / (common_sizes[0] + EPS)
    future_potential = 2.0 * np.exp(-5.0 * mod_results)
    
    # 5. Dynamic bin size preference
    bin_median = np.median(bins)
    bin_75 = np.percentile(bins, 75)
    size_pref = np.where(
        valid_bins < bin_median,
        1.0 + 0.3 * (bin_median - valid_bins) / (bin_median + EPS),  # Prefer smaller bins
        1.0 - 0.1 * (valid_bins - bin_median) / (bin_75 - bin_median + EPS)  # Slight penalty for very large bins
    )
    
    # 6. Stability penalty with adaptive thresholds
    stability_penalty = np.where(
        utilization > 0.95,
        -4.0 * (utilization - 0.95) * 20,  # Strong penalty for nearly full
        np.where(
            utilization > 0.85,
            -1.0 * (utilization - 0.85) * 10,  # Moderate penalty
            0.0
        )
    )
    
    # 7. System state awareness
    system_fill = np.sum(bins - valid_bins) / (np.sum(bins) + EPS)
    avg_item_size = np.mean(item)
    
    # Dynamic weights based on system state and item characteristics
    weights = np.array([
        1.0,  # exact_fit (always important)
        3.5 - 0.5 * system_fill + 0.3 * avg_item_size,  # tightness
        2.0 + 0.8 * system_fill - 0.2 * avg_item_size,  # utilization
        1.5 - 0.4 * system_fill + 0.1 * avg_item_size,  # future potential
        0.8 + 0.1 * system_fill - 0.05 * avg_item_size  # size preference
    ])
    
    # Combine all factors
    combined_priority = (
        weights[0] * exact_fit_bonus +
        weights[1] * tightness_priority +
        weights[2] * utilization_priority +
        weights[3] * future_potential +
        weights[4] * size_pref +
        stability_penalty +
        1e-8 * valid_bins  # Final tiebreaker (slight preference for larger bins)
    )
    
    priorities[valid_mask] = combined_priority
    return priorities



# Function 2 - Score: -0.039527326677224625
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    if item <= 0:
        raise ValueError("Item size must be positive")
    if np.any(bins <= 0):
        raise ValueError("All bin capacities must be positive")
    
    # Initialize priorities and calculate remaining capacity
    priorities = np.full_like(bins, -np.inf)
    remaining = bins - item
    valid_mask = remaining >= 0
    
    # Short-circuit if no valid bins
    if not np.any(valid_mask):
        return priorities
    
    # Extract valid bins and calculate basic metrics
    valid_bins = bins[valid_mask]
    remaining_valid = remaining[valid_mask]
    current_fill = 1 - (remaining_valid / valid_bins)
    
    # System state analysis
    avg_fill = np.mean(current_fill)
    median_fill = np.median(current_fill)
    fill_std = np.std(current_fill) if len(current_fill) > 1 else 0.0
    system_skew = avg_fill - median_fill
    system_size = np.mean(valid_bins)
    item_ratio_system = item / system_size
    
    # Dynamic weight calculation with more sophisticated adaptation
    base_weights = np.array([0.4, 0.35, 0.25])  # remaining, ratio, fill
    
    # Adjust weights based on system state
    fill_weight_adj = (0.2 * system_skew - 0.1 * fill_std + 
                      0.05 * np.tanh(10 * (item_ratio_system - 0.4)))
    ratio_weight_adj = (0.1 * (1 - fill_std) - 0.05 * system_skew + 
                      0.1 * np.tanh(5 * (0.5 - item_ratio_system)))
    
    weights = base_weights + np.array([-fill_weight_adj - ratio_weight_adj,
                                     ratio_weight_adj,
                                     fill_weight_adj])
    weights = np.clip(weights, 0.1, 0.8)  # Prevent extreme weights
    weights = weights / np.sum(weights)   # Renormalize
    
    # Feature 1: Remaining capacity score (non-linear with adaptive steepness)
    remaining_ratio = remaining_valid / valid_bins
    steepness = 10 - 6 * avg_fill + 3 * item_ratio_system
    remaining_score = 1 / (1 + np.exp(steepness * remaining_ratio - 3))
    
    # Feature 2: Size ratio score (logistic with adaptive midpoint)
    item_ratio = np.clip(item / valid_bins, 1e-5, 1-1e-5)
    ideal_ratio = 0.4 + 0.2 * avg_fill - 0.1 * system_skew
    ratio_score = 1 - 2 * np.abs(item_ratio - ideal_ratio)
    
    # Feature 3: Fill score (adaptive power function)
    fill_power = 0.7 + 0.3 * avg_fill - 0.15 * system_skew
    fill_score = np.power(current_fill, fill_power)
    
    # Combined base score
    combined_score = (weights[0] * remaining_score +
                     weights[1] * ratio_score +
                     weights[2] * fill_score)
    
    # Special case adjustments with smooth transitions
    # 1. Exact fits
    exact_fit = np.isclose(remaining_valid, 0, atol=1e-10)
    combined_score[exact_fit] = 3.0
    
    # 2. Progressive boost for nearly-full bins
    nearly_full = (current_fill > 0.75)
    boost_factor = 0.6 * (1 + np.tanh(5 * (avg_fill - 0.5)))
    fill_boost = np.maximum(current_fill[nearly_full] - 0.75, 0)
    combined_score[nearly_full] += boost_factor * fill_boost * 4
    
    # 3. Size-based bonuses/penalties
    size_bonus = 0.3 * np.tanh(8 * (item_ratio - 0.25))
    combined_score += size_bonus
    
    # 4. Fragmentation penalty for small items
    small_item = (item_ratio < 0.15)
    frag_penalty = 0.2 * np.exp(-10 * item_ratio[small_item])
    combined_score[small_item] -= frag_penalty
    
    # 5. System balancing bonus
    balance_bonus = 0.15 * np.exp(-8 * np.abs(current_fill - median_fill))
    combined_score += balance_bonus
    
    # Apply scores to valid bins
    priorities[valid_mask] = combined_score
    
    return priorities



# Function 3 - Score: -0.03955641930282427
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    valid_mask = bins >= item
    priorities = np.full_like(bins, -np.inf)
    
    if np.any(valid_mask):
        valid_bins = bins[valid_mask]
        remaining = valid_bins - item
        
        # Normalization factors
        mean_bin = valid_bins.mean()
        min_bin = valid_bins.min()
        max_bin = valid_bins.max()
        
        # Item size characteristics (normalized)
        relative_item = item / mean_bin
        item_size_factor = np.clip(relative_item, 0.01, 0.99)
        
        # Perfect fit gets strongest bonus (but not infinite)
        perfect_fit = (remaining == 0) * (10.0 + 5*item_size_factor)
        
        # Enhanced tightness measure with adaptive curve
        tightness_steepness = 4 + 6 * item_size_factor
        normalized_remaining = remaining / valid_bins
        tightness = 1 / (1 + np.exp(-tightness_steepness * (1 - normalized_remaining)))
        
        # Current utilization with cubic scaling (stronger preference for fuller bins)
        current_util = ((valid_bins - remaining) / valid_bins) ** 3
        
        # Bin size preference with logarithmic scaling
        # Balances preference for smaller bins without extreme bias
        size_pref = 1 / (1 + np.log1p(valid_bins / min_bin))
        
        # Enhanced fragmentation avoidance
        # Considers both current and potential future fragmentation
        ideal_remaining = np.clip(item, 0.2*min_bin, 0.5*max_bin)
        frag_score = np.exp(-((remaining - ideal_remaining) / (0.3*ideal_remaining))**2)
        
        # Future packing potential estimate
        # Favors bins where remaining space could fit typical future items
        future_fit_prob = 1 - np.abs(remaining - ideal_remaining) / (ideal_remaining + 1e-6)
        
        # Combine factors with dynamic weights
        combined_priority = (
            (10.0 + 3*item_size_factor) * perfect_fit +      # Perfect fit
            (6.0 - 2*item_size_factor) * tightness +         # Tight fit
            (4.0 + item_size_factor) * current_util +        # Current utilization
            1.5 * size_pref +                                # Bin size preference
            1.0 * frag_score +                               # Fragmentation avoidance
            0.8 * future_fit_prob +                          # Future packing potential
            valid_bins * 1e-10                               # Tie-breaker
        )
        
        priorities[valid_mask] = combined_priority
    
    return priorities



# Function 4 - Score: -0.03959933216984135
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    if item <= 0:
        raise ValueError("Item size must be positive")
    if np.any(bins <= 0):
        raise ValueError("All bin capacities must be positive")
    
    remaining = bins - item
    priorities = np.full_like(bins, -np.inf)  # Default to invalid
    
    # Only consider bins that can fit the item
    valid_mask = remaining >= 0
    valid_bins = bins[valid_mask]
    
    if len(valid_bins) == 0:
        return priorities
    
    # Numerical stability safeguards
    valid_bins = np.maximum(valid_bins, 1e-10)
    remaining_valid = remaining[valid_mask]
    
    # Current fill state calculations
    current_fill = 1 - (remaining_valid / valid_bins)
    avg_fill = np.mean(current_fill)
    median_fill = np.median(current_fill)
    fill_std = np.std(current_fill) if len(current_fill) > 1 else 0.0
    
    # System state analysis - more sophisticated metrics
    system_skew = avg_fill - median_fill
    fill_entropy = -np.sum(current_fill * np.log(current_fill + 1e-10)) / len(current_fill)
    normalized_entropy = fill_entropy / -np.log(avg_fill + 1e-10) if avg_fill > 0 else 1.0
    
    # Item-to-system ratio analysis
    item_ratio = np.clip(item / valid_bins, 1e-5, 1-1e-5)
    system_item_ratio = item / np.mean(bins) if len(bins) > 0 else 1.0
    
    # Dynamic weight adjustments - more responsive to system state
    # When entropy is low (uneven distribution), prioritize balancing
    balance_factor = 0.5 * (1 - normalized_entropy)
    # When system is skewed toward full bins, prioritize completing them
    completion_factor = 0.5 * system_skew
    
    fill_weight = 0.4 + 0.3 * completion_factor - 0.2 * balance_factor
    ratio_weight = 0.3 + 0.3 * balance_factor - 0.1 * completion_factor
    remaining_weight = 1.0 - fill_weight - ratio_weight
    
    # 1. Remaining capacity score (adaptive sigmoid)
    remaining_ratio = remaining_valid / valid_bins
    steepness = 10 - 6 * avg_fill  # more aggressive when system is fuller
    remaining_score = 2 / (1 + np.exp(steepness * remaining_ratio)) - 1
    
    # 2. Size ratio score (enhanced with system awareness)
    log_ratio = np.log(item_ratio) - np.log(1 - item_ratio)
    ratio_score = np.tanh(log_ratio / (3 + 2 * system_item_ratio))  # smoother for large items
    
    # 3. Fill score with adaptive non-linearity and entropy awareness
    fill_power = 0.4 + 0.6 * avg_fill  # more aggressive as system fills up
    fill_score = np.power(current_fill, fill_power)
    
    # Combined base score
    combined_score = (
        remaining_weight * remaining_score +
        ratio_weight * ratio_score +
        fill_weight * fill_score
    )
    
    # Special case handling with more nuanced boosts
    # 1. Exact fits (with floating point tolerance)
    exact_fit = np.isclose(remaining_valid, 0, atol=1e-10)
    combined_score[exact_fit] = 3.0  # highest priority
    
    # 2. Nearly-full bins (dynamic threshold based on system state)
    nearly_full_threshold = 0.8 + 0.1 * system_skew  # lower when system is skewed
    nearly_full = (current_fill > nearly_full_threshold)
    boost_factor = 0.5 * (1 + system_skew - 0.5 * fill_std)
    combined_score[nearly_full] += boost_factor * (
        (current_fill[nearly_full] - nearly_full_threshold) * 
        (10 / (1 - nearly_full_threshold))
    )
    
    # 3. Large items relative to bin size
    large_item = (item_ratio > 0.3)
    combined_score[large_item] += 0.2 * np.sqrt(item_ratio[large_item])
    
    # 4. Small items with anti-fragmentation logic
    small_item = (item_ratio < 0.1)
    penalty_factor = 0.15 * (1 - avg_fill)  # stronger penalty in emptier systems
    combined_score[small_item] -= penalty_factor * (0.1 - item_ratio[small_item]) * 10
    
    # 5. Items that would leave very small remaining capacity
    small_remaining = (remaining_valid / valid_bins < 0.05)
    combined_score[small_remaining] += 0.1 * (1 - remaining_valid[small_remaining]/valid_bins[small_remaining]) * 20
    
    priorities[valid_mask] = combined_score
    
    return priorities



# Function 5 - Score: -0.039619022285058716
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    if item == 0:
        return np.zeros_like(bins)
    
    remaining = bins - item
    priorities = np.full_like(bins, -np.inf)
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return priorities
    
    valid_bins = bins[valid_mask]
    valid_remaining = remaining[valid_mask]
    
    # Exact fit gets highest priority (100x boost)
    exact_fit = (valid_remaining == 0)
    
    # For non-exact fits, calculate two components:
    # 1. Inverse of remaining space (strong preference for small remaining)
    #    Using log1p for numerical stability and smooth decay
    space_efficiency = -np.log1p(valid_remaining)
    
    # 2. Bin size preference (slight preference for larger bins)
    #    Helps maintain flexibility for future items
    bin_size_preference = np.log1p(valid_bins) * 0.1
    
    # Combine components with exact fit getting massive boost
    priorities[valid_mask] = (
        space_efficiency + 
        bin_size_preference + 
        exact_fit * 100.0  # Exact fits dominate all other considerations
    )
    
    return priorities



# Function 6 - Score: -0.03963454021178764
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    priorities = np.full_like(bins, -np.inf)
    valid_mask = remaining >= 0
    valid_bins = bins[valid_mask]
    valid_remaining = remaining[valid_mask]
    
    if not valid_mask.any():
        return priorities
    
    # System statistics
    total_capacity = np.sum(valid_bins)
    used_space = np.sum(valid_bins - valid_remaining)
    system_fill = used_space / total_capacity
    avg_bin_size = np.mean(valid_bins)
    item_ratio = item / avg_bin_size
    
    # Current fill ratios
    current_fill = (valid_bins - valid_remaining) / valid_bins
    new_fill = current_fill + item / valid_bins
    
    # Dynamic weighting factors
    if item_ratio < 0.05:
        # Micro items - prioritize tight packing
        priorities[valid_mask] = 1 / (valid_remaining + 1e-6)
    elif item_ratio > 0.9:
        # Large items - prioritize exact fits
        exact_fit = (valid_remaining < 1e-6).astype(float)
        priorities[valid_mask] = exact_fit * 10 + (1 / (valid_remaining + 1e-6))
    else:
        # Medium items - adaptive combination
        if system_fill > 0.85:
            # When nearly full, focus on utilization
            weights = [0.2, 0.7, 0.1]  # utilization, exact fit, gap avoidance
        else:
            # Normal operation - balanced approach
            weights = [0.4, 0.4, 0.2]  # utilization, exact fit, gap avoidance
            
        exact_fit = 1 / (valid_remaining + 1e-6)
        utilization = new_fill
        gap_avoidance = -valid_remaining / valid_bins
        
        priorities[valid_mask] = (
            weights[0] * utilization +
            weights[1] * exact_fit +
            weights[2] * gap_avoidance
        )
    
    # Normalize priorities to (0,1] range for valid bins
    valid_prio = priorities[valid_mask]
    if valid_prio.size > 0:
        min_p, max_p = np.min(valid_prio), np.max(valid_prio)
        if max_p > min_p:
            priorities[valid_mask] = (valid_prio - min_p) / (max_p - min_p)
    
    return priorities



# Function 7 - Score: -0.03963454021178764
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    if item <= 0:
        raise ValueError("Item size must be positive")
    if np.any(bins <= 0):
        raise ValueError("All bin capacities must be positive")
    
    remaining = bins - item
    priorities = np.full_like(bins, -np.inf)  # Default to negative infinity for invalid
    
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return priorities
    
    valid_bins = bins[valid_mask]
    remaining_valid = remaining[valid_mask]
    
    # Core metrics (simpler than v0 but more robust than v1)
    new_fill = 1 - remaining_valid / valid_bins  # How full bin would be after adding
    fit_quality = 1 / (remaining_valid + 1e-8)   # Prefer tighter fits (with small epsilon)
    
    # Item-to-bin size ratio (important for decision making)
    size_ratio = item / valid_bins
    
    # Combined score - weights determined empirically
    combined_score = (
        0.6 * new_fill +          # Prefer filling bins
        0.4 * fit_quality +       # But with preference for good fits
        0.2 * np.log1p(size_ratio)  # Bonus for matching item to bin size
    )
    
    # Special case: exact fits get highest priority
    exact_fit = np.isclose(valid_bins, item, rtol=1e-8, atol=1e-10)
    combined_score[exact_fit] = 10.0
    
    # Apply the scores to valid bins
    priorities[valid_mask] = combined_score
    
    # For invalid bins, provide meaningful negative scores
    # (better than v1's simple negative ratio)
    over_capacity = item - bins[~valid_mask]
    priorities[~valid_mask] = -np.log1p(over_capacity) - 1
    
    # Numerical stability
    priorities = np.nan_to_num(priorities, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    priorities = np.clip(priorities, -10, 10)
    
    return priorities



# Function 8 - Score: -0.03964670825952325
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    if item == 0:
        return np.zeros_like(bins)
    
    remaining = bins - item
    priorities = np.full_like(bins, -np.inf)
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return priorities
    
    valid_bins = bins[valid_mask]
    valid_remaining = remaining[valid_mask]
    utilization = 1 - (valid_remaining / valid_bins)
    
    # Advanced bin statistics
    avg_bin_size = np.mean(bins)
    std_bin_size = np.std(bins)
    median_bin_size = np.median(bins)
    relative_item_size = item / avg_bin_size
    
    # System state characterization
    is_small_item = relative_item_size < 0.05
    is_medium_item = (0.05 <= relative_item_size) & (relative_item_size <= 0.3)
    is_large_item = relative_item_size > 0.3
    is_uniform_bins = std_bin_size < 0.05 * avg_bin_size
    total_remaining = np.sum(remaining[valid_mask])
    packing_density = 1 - (total_remaining / (np.sum(valid_bins) + 1e-10))
    
    # Component 1: Utilization priority with adaptive curve
    if is_uniform_bins:
        util_steepness = 20 + 10 * relative_item_size
        util_center = 0.92 if is_small_item else 0.88
    else:
        util_steepness = 10 + 8 * relative_item_size
        util_center = 0.85 - (0.05 * packing_density)  # Adjust based on overall packing
    
    utilization_priority = 1 / (1 + np.exp(-util_steepness * (utilization - util_center)))
    
    # Component 2: Enhanced exact fit detection
    near_exact_threshold = 0.005 * avg_bin_size  # Consider very close fits
    exact_fit_bonus = np.where(
        valid_remaining <= near_exact_threshold,
        # Scale bonus with item size and system state
        10.0 * (1 + relative_item_size) * (1 + packing_density),
        0.0
    )
    
    # Component 3: Remaining space penalty with non-linear scaling
    remaining_penalty = -np.where(
        valid_remaining > 0,
        np.log1p(valid_remaining) * (1 + 3*relative_item_size),
        0
    )
    
    # Component 4: Small item optimization (cluster in few bins)
    small_item_bonus = np.where(
        is_small_item,
        # Strong preference for bins that are almost full
        np.exp(8 * (utilization - 0.97)) + (utilization > 0.95) * 2.0,
        0
    )
    
    # Component 5: Large item strategy (avoid awkward remaining space)
    large_item_penalty = np.where(
        is_large_item,
        # Penalize bins that would leave very small remaining space
        -np.exp(-valid_remaining/(0.05 * median_bin_size)) * 3,
        0
    )
    
    # Component 6: Smart distribution balancing
    if len(bins) > 5:  # Only apply when we have enough bins
        if packing_density < 0.7:
            # Early stage: encourage even distribution
            avg_fill = np.mean(utilization)
            distribution_bonus = -1.0 * np.abs(utilization - avg_fill)
        else:
            # Late stage: consolidate to fill bins
            distribution_bonus = 0.5 * utilization
    else:
        distribution_bonus = 0
    
    # Component 7: Bin size adaptation (prefer smaller bins for better density)
    size_adaptation = -0.3 * (valid_bins / avg_bin_size - 1) * (1 - utilization)
    
    # Component 8: Controlled randomness with adaptive scale
    random_scale = 0.02 * (1 + relative_item_size) * (1 - packing_density)
    random_component = np.random.uniform(0, random_scale, size=valid_remaining.shape)
    
    # Dynamic weighting based on system state
    if is_small_item:
        weights = [0.7, 0.3, 0.2, 2.5, 0, 0.5, 0.5, 0.1]
    elif is_large_item:
        weights = [1.5, 2.5, 1.8, 0, 2.0, 0.8, -0.2, 0.1]
    else:
        weights = [1.2, 1.8, 1.2, 0, 0, 1.0, 0.3, 0.1]
    
    # Combine all components
    priorities[valid_mask] = (
        weights[0] * utilization_priority +
        weights[1] * exact_fit_bonus +
        weights[2] * remaining_penalty +
        weights[3] * small_item_bonus +
        weights[4] * large_item_penalty +
        weights[5] * distribution_bonus +
        weights[6] * size_adaptation +
        weights[7] * random_component
    )
    
    return priorities



# Function 9 - Score: -0.03965321730844011
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    
    # Initialize priorities with -inf for invalid bins
    priorities = np.full_like(bins, -np.inf)
    valid_mask = remaining >= 0
    valid_remaining = remaining[valid_mask]
    valid_bins = bins[valid_mask]
    
    if len(valid_bins) > 0:
        # Main components
        perfect_fit = (valid_remaining == 0)  # Highest priority
        will_be_full = 1 - valid_remaining    # How full bin will be after adding
        
        # Penalties
        empty_penalty = 0.3 * (1 - valid_bins)  # Penalize currently empty bins
        nearly_full_penalty = 0.1 * ((valid_bins > 0.85) & (valid_bins < 1.0))
        
        # Combined score
        combined = (10 * perfect_fit +          # Strong boost for perfect fits
                   will_be_full -               # Prefer fuller bins
                   empty_penalty -              # Avoid empty bins
                   nearly_full_penalty)         # Avoid creating many almost-full bins
        
        # Apply exponential scaling to emphasize differences
        priorities[valid_mask] = np.exp(combined)
        
        # Normalize to prevent numerical overflow
        max_priority = np.max(priorities[valid_mask])
        if max_priority > 1e6:
            priorities[valid_mask] = priorities[valid_mask] / (max_priority / 1e6)
    
    return priorities



# Function 10 - Score: -0.03966228345716389
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    valid_mask = bins >= item
    priorities = np.full_like(bins, -np.inf)
    
    if np.any(valid_mask):
        valid_bins = bins[valid_mask]
        remaining = valid_bins - item
        
        # Key metrics (all in [0,1] range where possible)
        normalized_remaining = remaining / valid_bins
        utilization = 1 - normalized_remaining
        fit_ratio = item / valid_bins
        
        # Exact fit bonus (smooth transition near exact fits)
        exact_fit_bonus = 10.0 * np.exp(-50.0 * normalized_remaining**2)
        
        # Main priority components
        remaining_priority = -np.log1p(remaining)  # Strong preference for less remaining
        utilization_priority = 0.5 * np.log1p(utilization * 100)  # Reward higher utilization
        
        # Tiebreakers
        bin_size_tiebreaker = np.log1p(valid_bins) * 1e-6  # Slight preference for larger bins
        fit_tiebreaker = 0.1 * fit_ratio**0.5  # Mild preference for better fitting items
        
        priorities[valid_mask] = (
            exact_fit_bonus +
            remaining_priority +
            utilization_priority +
            bin_size_tiebreaker +
            fit_tiebreaker
        )
    
    return priorities



