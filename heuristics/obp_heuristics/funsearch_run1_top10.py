# Top 10 functions for funsearch run 1

# Function 1 - Score: -0.03957205173697356
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    can_fit = (remaining >= 0)
    priorities = np.full_like(bins, -np.inf)  # Default to -inf for non-fittable bins
    
    if not np.any(can_fit):
        return priorities
    
    remaining_fittable = remaining[can_fit]
    bins_fittable = bins[can_fit]
    eps = np.finfo(float).eps  # Machine epsilon
    
    # System state analysis
    total_capacity = np.sum(bins)
    used_capacity = total_capacity - np.sum(remaining[remaining > 0])
    system_utilization = used_capacity / (total_capacity + eps)
    
    # Dynamic weights based on system state
    consolidation_weight = 0.5 + 0.5 * system_utilization  # More consolidation when system is fuller
    tight_fit_weight = 1.2 - 0.5 * system_utilization  # More tight fit when system is emptier
    
    # 1. Exact fit bonus (absolute priority)
    exact_fit = (remaining_fittable == 0).astype(float)
    
    # 2. Base score - enhanced logarithmic scaling
    base_score = -np.log1p(remaining_fittable / (item + eps))
    
    # 3. Utilization component - dynamic curve based on system state
    utilization = 1 - (remaining_fittable / bins_fittable)
    k = 10 + 10 * system_utilization  # Sharper transition when system is fuller
    utilization_bonus = 1 / (1 + np.exp(-k * (utilization - 0.7)))  # Transition around 70%
    
    # 4. Fit quality - adaptive exponential bonus
    fit_quality = np.exp(-remaining_fittable / (np.sqrt(item) + eps))
    
    # 5. Bin size preference - favors consolidation in larger bins
    median_bin = np.median(bins_fittable) + eps
    size_pref = np.log1p(bins_fittable / median_bin) * consolidation_weight
    
    # 6. Anti-fragmentation bonus - helps prevent many nearly-empty bins
    frag_threshold = 0.1 * item
    anti_frag = np.where(remaining_fittable < frag_threshold, 
                         np.exp(-remaining_fittable / (frag_threshold + eps)), 
                         0)
    
    # 7. Adaptive tie-breaking noise
    current_scores = base_score + utilization_bonus + fit_quality + size_pref
    score_range = np.ptp(current_scores) if len(current_scores) > 1 else 1.0
    noise_scale = max(1e-6, score_range * 1e-2)  # Proportional to score range
    noise = np.random.uniform(-noise_scale, noise_scale, size=remaining_fittable.shape)
    
    # Combine components with dynamic weights
    priorities[can_fit] = (
        1000.0 * exact_fit +               # Absolute priority for exact fits
        1.5 * base_score +                 # Main capacity consideration
        1.0 * utilization_bonus +          # Utilization preference
        tight_fit_weight * fit_quality +   # Adaptive fit quality bonus
        0.5 * size_pref +                  # Bin size preference
        0.8 * anti_frag +                  # Anti-fragmentation
        noise                              # Adaptive tie-breaker
    )
    
    return priorities



# Function 2 - Score: -0.03957860745588624
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    epsilon = 1e-12  # For numerical stability
    
    # Basic calculations and masks
    remaining = bins - item
    fits_mask = remaining >= -epsilon
    perfect_fit_mask = np.abs(remaining) <= epsilon
    
    # Dynamic tolerance calculation (adaptive to bin sizes)
    median_bin = np.median(bins[bins > epsilon])
    relative_tol = 0.001 + 0.004 * (item / median_bin)
    absolute_tol = max(0.003, 0.015 * item)
    dynamic_tolerance = np.maximum(bins * relative_tol, absolute_tol)
    near_perfect_mask = np.abs(remaining) <= dynamic_tolerance
    
    # Bin and item characteristics
    max_bin = np.max(bins)
    avg_bin = np.mean(bins[bins > epsilon])
    bin_ratio = bins / max_bin
    item_ratio = item / max_bin
    
    # Item classification
    is_small = item < 0.1 * avg_bin
    is_medium = (item >= 0.1 * avg_bin) & (item <= 0.6 * max_bin)
    is_large = item > 0.6 * max_bin
    
    # Utilization metrics
    utilization = 1 - (remaining / (bins + epsilon))
    remaining_ratio = remaining / (bins + epsilon)
    
    # 1. Base capacity priority (non-linear favoring bins with less remaining)
    capacity_priority = np.exp(-5 * remaining / avg_bin)
    
    # 2. Advanced utilization boost (smooth curve with multiple plateaus)
    util_boost = np.where(
        utilization > 0.95,
        1.8,
        np.where(
            utilization > 0.85,
            1.4 + 0.4 * (utilization - 0.85) / 0.1,
            np.where(
                utilization > 0.7,
                1.2 + 0.2 * (utilization - 0.7) / 0.15,
                1.0 + 0.2 * (utilization / 0.7)
            )
        )
    )
    
    # 3. Fragmentation avoidance (three zones with smooth transitions)
    frag_threshold = 0.1 * avg_bin
    fragmentation = np.where(
        remaining <= epsilon,
        1.0,
        np.where(
            remaining < frag_threshold,
            0.6 + 0.4 * (remaining / frag_threshold)**0.7,  # Sub-linear ramp
            1.0
        )
    )
    
    # 4. Adaptive size strategy
    size_strategy = np.ones_like(bins)
    if is_small:
        # Small items: balanced approach between large and medium bins
        size_strategy = 0.7 + 0.3 * np.sqrt(bin_ratio)
    elif is_large:
        # Large items: strong preference for largest bins
        size_strategy = 0.2 + 0.8 * bin_ratio**1.5
    else:
        # Medium items: slight preference for medium bins
        mid_range = 0.3 + 0.4 * bin_ratio
        size_strategy = 0.8 + 0.2 * np.cos(2 * np.pi * (bin_ratio - 0.5))
    
    # 5. Density bonus (encourages similar-sized items)
    ideal_fill = 0.5  # Target fill ratio for density bonus
    density_bonus = 1.0 + 0.4 * np.exp(-8 * np.abs((item/(bins + epsilon)) - ideal_fill))
    
    # 6. Load balancing factor (avoids overloading large bins)
    load_balance = 1.0 - 0.2 * (bin_ratio**2)
    
    # Combine all factors with learned weights
    base_priority = (
        0.4 * capacity_priority +
        0.3 * util_boost +
        0.15 * fragmentation +
        0.1 * size_strategy +
        0.05 * load_balance
    ) * density_bonus
    
    # Hierarchical priority assignment
    priorities = np.full_like(bins, -np.inf)  # Default for non-fitting bins
    priorities[fits_mask] = np.where(
        perfect_fit_mask[fits_mask],
        4.0,  # Highest priority for perfect fit
        np.where(
            near_perfect_mask[fits_mask],
            3.0,  # Very high for near-perfect fit
            np.where(
                utilization[fits_mask] > 0.97,
                2.5,  # Extreme high for nearly full
                np.where(
                    utilization[fits_mask] > 0.9,
                    2.0,  # High for very utilized bins
                    np.where(
                        utilization[fits_mask] > 0.8,
                        1.5,  # Boost for well-utilized bins
                        base_priority[fits_mask]  # Default case
                    )
                )
            )
        )
    )
    
    return priorities



# Function 3 - Score: -0.039597435364797665
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    
    # Initialize scores with large negative value for bins that can't fit the item
    scores = np.full_like(bins, -np.inf)
    valid_mask = remaining >= 0
    valid_remaining = remaining[valid_mask]
    valid_bins = bins[valid_mask]
    
    if valid_remaining.size == 0:
        return scores  # No bins can fit the item
    
    # Calculate base scores with more nuanced factors
    perfect_fit = (valid_remaining == 0)
    nearly_full = valid_remaining < 0.05 * valid_bins
    medium_full = valid_remaining < 0.2 * valid_bins
    
    # Space utilization factor (more weight to less remaining space)
    space_factor = 1.0 / (1.0 + valid_remaining)
    
    # Bin size preference (slight preference for larger bins)
    size_factor = 0.05 * np.log1p(valid_bins)
    
    # Progressive bonuses based on how full the bin is
    fullness_bonus = np.where(
        perfect_fit,
        3.0,  # Highest priority for perfect fit
        np.where(
            nearly_full,
            2.0 + 0.5 * (0.05 * valid_bins - valid_remaining) / (0.05 * valid_bins),
            np.where(
                medium_full,
                1.0 + 0.5 * (0.2 * valid_bins - valid_remaining) / (0.2 * valid_bins),
                0.5 * (1.0 - valid_remaining / valid_bins)
            )
        )
    )
    
    # Combine factors with different weights
    combined_score = (
        2.0 * space_factor + 
        1.0 * size_factor + 
        3.0 * fullness_bonus
    )
    
    # Add small random noise to break ties (seed for reproducibility)
    rng = np.random.RandomState(42)
    noise = 0.001 * rng.rand(*valid_remaining.shape)
    
    scores[valid_mask] = combined_score + noise
    
    return scores



# Function 4 - Score: -0.03965582828899743
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    priorities = np.full_like(bins, -np.inf)  # Default to -inf for invalid bins
    
    valid_mask = remaining >= 0
    remaining_valid = remaining[valid_mask]
    bins_valid = bins[valid_mask]
    
    if remaining_valid.size > 0:
        # Special cases get maximum priority
        perfect_fit = (remaining_valid == 0)
        empty_bin = (bins_valid == item)  # Item exactly fills empty bin
        near_perfect = (remaining_valid <= 0.01 * bins_valid)  # Within 1% of perfect
        
        # Normalized metrics
        norm_remaining = remaining_valid / np.maximum(bins_valid, 1e-10)  # Avoid division by zero
        new_utilization = (bins_valid - remaining_valid) / np.maximum(bins_valid, 1e-10)
        
        # Dynamic weighting based on item size
        item_size_ratio = item / np.median(bins_valid)
        util_weight = 5.0 + 2.0 * np.tanh(3.0 * item_size_ratio)  # [5-7] range
        fragment_weight = 2.0 + 1.0 * np.tanh(5.0 * (1 - item_size_ratio))  # [2-3] range
        
        # Strategic components
        utilization_bonus = new_utilization**3  # Non-linear preference for full bins
        fragment_penalty = np.exp(-5 / (norm_remaining + 1e-10))  # Penalize small fragments
        near_perfect_bonus = 10 * np.exp(-100 * norm_remaining)
        
        # Additional considerations
        bin_size_adjustment = 0.1 * (bins_valid / np.max(bins_valid))  # Slight preference for larger bins
        future_fit_penalty = np.where(
            remaining_valid > 0,
            np.exp(-2 / (remaining_valid + 1e-10)),  # Penalize bins that would be hard to fill later
            0
        )
        
        priorities[valid_mask] = (
            util_weight * utilization_bonus +
            -fragment_weight * fragment_penalty +
            3.0 * near_perfect_bonus +
            -1.5 * future_fit_penalty +
            1000.0 * perfect_fit +
            800.0 * empty_bin +
            bin_size_adjustment
        )
        
        # Special handling for very small items
        if item < 0.01 * np.median(bins_valid[bins_valid > 0]):
            tiny_fragment_penalty = np.where(
                remaining_valid > 0,
                np.exp(-10 / (remaining_valid + 1e-10)),
                0
            )
            priorities[valid_mask] -= 5.0 * tiny_fragment_penalty
    
    return priorities



# Function 5 - Score: -0.039679424861097684
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid_mask = remaining >= 0
    priorities = np.full_like(bins, -np.inf)
    
    if not np.any(valid_mask):
        return priorities
    
    remaining_valid = remaining[valid_mask]
    bins_valid = bins[valid_mask]
    
    # Calculate metrics more efficiently
    perfect_fit = np.isclose(remaining_valid, 0, atol=max(1e-10, item*1e-8))  # Relative tolerance
    future_utilization = 1 - (remaining_valid / bins_valid)
    current_utilization = 1 - (remaining_valid / bins_valid)  # Same as future here
    total_capacity = np.sum(bins)
    used_capacity = total_capacity - np.sum(remaining_valid)
    global_utilization = used_capacity / total_capacity
    
    # Enhanced stage detection with hysteresis
    if global_utilization < 0.15:
        stage = 'early'
    elif global_utilization < 0.65:
        stage = 'mid'
    else:
        stage = 'late'
    
    # Perfect fit gets absolute priority with smarter tie-breaking
    perfect_fit_priority = np.where(
        perfect_fit,
        1000.0 + (1 - (bins_valid/np.max(bins_valid))) * 0.1,  # Prefer smaller perfect fits
        0.0
    )
    
    # Space priority - adaptive curve based on stage
    if stage == 'early':
        space_priority = future_utilization ** 2  # Gentle preference
    elif stage == 'mid':
        space_priority = np.exp(3 * future_utilization) - 1  # Moderate
    else:
        space_priority = np.exp(8 * future_utilization) - 1  # Aggressive
    
    # Dynamic size penalty considering bin size distribution
    median_bin = np.median(bins)
    bin_size_ratio = bins_valid / median_bin
    
    if stage == 'early':
        size_penalty = 0.4 * bin_size_ratio
    elif stage == 'mid':
        size_penalty = 0.2 * np.sqrt(bin_size_ratio)
    else:
        size_penalty = 0.05 * np.log1p(bin_size_ratio)
    
    # Balance component - considers both current and target distribution
    utilization_std = np.std(current_utilization)
    if utilization_std > 0.2:
        # When utilization is uneven, reward balancing
        balance_bonus = 0.7 * (1 - current_utilization)
    else:
        # Otherwise focus on filling
        balance_bonus = 0.3 * (1 - current_utilization) ** 2
    
    # Combine components with stage-aware weights
    weights = {
        'early': {'perfect': 5, 'space': 1, 'size': -1, 'balance': 0.8},
        'mid': {'perfect': 10, 'space': 2, 'size': -0.5, 'balance': 0.5},
        'late': {'perfect': 20, 'space': 5, 'size': -0.1, 'balance': 0.2}
    }
    w = weights[stage]
    
    priorities_valid = (
        w['perfect'] * perfect_fit_priority +
        w['space'] * space_priority +
        w['size'] * size_penalty +
        w['balance'] * balance_bonus
    )
    
    # Adaptive noise scaling
    priority_scale = np.std(priorities_valid) if len(priorities_valid) > 1 else 1
    noise_magnitude = max(1e-6, priority_scale * 1e-4)
    priorities_valid += np.random.uniform(-noise_magnitude, noise_magnitude, 
                                        size=priorities_valid.shape)
    
    priorities[valid_mask] = priorities_valid
    return priorities



# Function 6 - Score: -0.0396823791147907
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid_mask = remaining >= -1e-10  # Slightly more tolerant than >= 0
    priorities = np.full_like(bins, -np.inf)
    
    if not np.any(valid_mask):
        return priorities
    
    remaining_valid = remaining[valid_mask]
    bins_valid = bins[valid_mask]
    current_utilization = 1 - (remaining_valid + item) / bins_valid
    future_utilization = 1 - remaining_valid / bins_valid
    
    # Global state analysis
    mean_util = np.mean(current_utilization)
    median_bin = np.median(bins_valid)
    std_util = np.std(current_utilization) if len(current_utilization) > 1 else 0.5
    total_remaining = np.sum(remaining_valid)
    system_entropy = -np.sum(current_utilization * np.log(current_utilization + 1e-10))
    
    # Perfect fit handling (with tie-breakers)
    perfect_fit = np.isclose(remaining_valid, 0, atol=1e-10)
    perfect_fit_priority = np.where(
        perfect_fit,
        10.0 + np.random.uniform(0, 0.001, size=perfect_fit.shape),  # Higher base value for perfect fits
        0.0
    )
    
    # Phase detection and strategy selection
    if mean_util < 0.3:
        # EARLY PHASE: Focus on creating diverse utilization
        space_priority = np.exp(1 - (remaining_valid / bins_valid)) - np.e
        size_penalty = 0.4 * (bins_valid / median_bin) ** 0.7
        threshold_bonus = 0.5 * (future_utilization >= 0.25)
        util_penalty = 0.1 * current_utilization
        
        # Strong encouragement for first items in bins
        empty_bonus = 0.8 * (current_utilization < 0.05)
        
    elif mean_util < 0.7:
        # MIDDLE PHASE: Balance between utilization and preparation for final phase
        space_priority = 2 * (np.exp(1 - (remaining_valid / bins_valid)) - np.e)
        size_penalty = 0.2 * (bins_valid / median_bin) ** 0.5
        threshold_bonus = 0.3 * ((future_utilization >= 0.5) + 
                                (future_utilization >= 0.75))
        util_penalty = 0.05 * current_utilization
        empty_bonus = 0.0
        
        # Encourage reducing number of active bins
        active_bin_penalty = 0.1 * (1 - np.sqrt(current_utilization))
        
    else:
        # FINAL PHASE: Maximize utilization of existing bins
        space_priority = 3 * (np.exp(2 * (1 - remaining_valid / bins_valid)) - np.e**2)
        size_penalty = 0.1 * (bins_valid / median_bin) ** 0.3
        threshold_bonus = 0.6 * (future_utilization >= 0.9)
        util_penalty = 0.02 * current_utilization
        empty_bonus = 0.0
        
        # Strong penalty for creating new utilization points
        active_bin_penalty = 0.3 * (1 - current_utilization)
    
    # Adaptive components based on system state
    entropy_factor = 0.1 * (1 - np.exp(-system_entropy))
    diversity_bonus = 0.0
    
    # Bin size diversity encouragement (only in early/middle phases)
    if mean_util < 0.6 and len(bins_valid) > 3:
        unique_sizes, counts = np.unique(bins_valid, return_counts=True)
        size_rarity = 1 / np.interp(bins_valid, unique_sizes, counts)
        diversity_bonus = 0.002 * (size_rarity / size_rarity.max())
    
    # Combine all components
    base_priority = (
        perfect_fit_priority +
        space_priority +
        threshold_bonus +
        empty_bonus -
        size_penalty -
        util_penalty -
        (active_bin_penalty if 'active_bin_penalty' in locals() else 0)
    )
    
    # Final adjustments
    priorities_valid = base_priority * (1 + entropy_factor) + diversity_bonus
    
    # Micro-adjustments for tie-breaking
    tie_breaker = np.random.uniform(-0.001, 0.001, size=priorities_valid.shape)
    priorities_valid += tie_breaker
    
    priorities[valid_mask] = priorities_valid
    
    return priorities



# Function 7 - Score: -0.03968367543191342
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    priorities = np.full_like(bins, -np.inf)  # Default to negative infinity
    
    valid_mask = remaining >= 0
    remaining_valid = remaining[valid_mask]
    bins_valid = bins[valid_mask]
    
    if remaining_valid.size > 0:
        # Perfect fits get highest priority (scaled by bin size)
        perfect_fit = (remaining_valid == 0)
        
        # For non-perfect fits, balance between remaining space and bin size
        # exp(-remaining) gives strong preference to tighter fits
        # sqrt(bins) gives moderate preference to larger bins (less aggressive than linear)
        priorities[valid_mask] = np.where(
            perfect_fit,
            # For perfect fits: bin_size + 1 to ensure they're above all non-perfect
            bins_valid + 1,
            # For others: combination of tightness and bin size
            np.exp(-remaining_valid) * np.sqrt(bins_valid)
        )
    
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
    
    # Initialize scores with -inf for bins that can't fit the item
    scores = np.full_like(bins, -np.inf)
    valid_mask = remaining >= 0
    valid_remaining = remaining[valid_mask]
    valid_bins = bins[valid_mask]
    
    if valid_remaining.size == 0:
        return scores  # No bins can fit the item
    
    # Perfect fits get highest priority
    perfect_fit = (valid_remaining == 0)
    
    # For non-perfect fits, calculate multiple factors:
    # 1. Space utilization (primary factor)
    space_utilization = 1.0 / (1.0 + valid_remaining)
    
    # 2. Bin size preference (secondary factor, logarithmic to prevent dominance)
    size_preference = 0.2 * np.log1p(valid_bins)
    
    # 3. Nearly-full bonus (smooth transition using sigmoid)
    fill_ratio = 1 - (valid_remaining / valid_bins)
    nearly_full_bonus = 1.0 / (1.0 + np.exp(-10 * (fill_ratio - 0.85)))  # sigmoid centered at 85% full
    
    # 4. Medium-term packing potential (prefer bins where remaining space is divisible by common item sizes)
    # Assuming common item sizes around 0.1-0.3 of bin sizes
    packing_potential = 0.3 * np.exp(-(valid_remaining % 0.2)**2 / 0.02)
    
    scores[valid_mask] = np.where(
        perfect_fit,
        10.0,  # Even higher priority for perfect fits
        space_utilization * (1 + size_preference + nearly_full_bonus + packing_potential)
    )
    
    return scores



# Function 9 - Score: -0.039689072194308415
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    priorities = np.full_like(bins, -np.inf)  # Default to -inf for invalid bins
    
    valid_mask = remaining >= 0
    remaining_valid = remaining[valid_mask]
    bins_valid = bins[valid_mask]
    
    if remaining_valid.size > 0:
        # Calculate various metrics
        perfect_fit = (remaining_valid == 0) & (bins_valid > 0)
        will_be_empty = (remaining_valid == bins_valid)  # Item fills the bin completely
        norm_remaining = remaining_valid / bins_valid
        new_utilization = (bins_valid - remaining_valid) / bins_valid
        
        # Component scores
        utilization_score = new_utilization ** 2  # Quadratic bonus for high utilization
        fragmentation_penalty = np.exp(-8 * norm_remaining)  # Strong penalty for leftover space
        perfect_fit_bonus = 100 * perfect_fit
        empty_bin_bonus = 0.5 * will_be_empty  # Moderate bonus for filling empty bins
        
        # Special case: if item exactly matches bin size (new bin creation)
        new_bin_case = (bins_valid == item)
        new_bin_bonus = 0.3 * new_bin_case  # Small bonus for perfect new bins
        
        # Combine all components
        combined_score = (
            utilization_score +
            fragmentation_penalty +
            perfect_fit_bonus +
            empty_bin_bonus +
            new_bin_bonus
        )
        
        # Special handling for empty bins to encourage consolidation
        empty_bins = (bins_valid == bins)  # Originally empty bins
        combined_score[empty_bins] *= 0.7  # Slightly reduce priority for empty bins
        
        priorities[valid_mask] = combined_score
    
    return priorities



# Function 10 - Score: -0.03968957773999936
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    valid_mask = remaining >= -1e-10  # Slightly more tolerant than v1
    priorities = np.full_like(bins, -np.inf)
    
    if not np.any(valid_mask):
        return priorities
    
    remaining_valid = remaining[valid_mask]
    bins_valid = bins[valid_mask]
    
    # Fundamental metrics
    perfect_fit = np.isclose(remaining_valid, 0, atol=1e-10)
    future_utilization = 1 - (remaining_valid / bins_valid)
    current_utilization = 1 - (bins_valid - item) / bins_valid
    
    # System-wide metrics
    median_bin = np.median(bins_valid)
    mean_utilization = np.mean(current_utilization)
    system_fill_ratio = np.sum(bins - remaining_valid) / np.sum(bins)
    num_bins = len(bins_valid)
    
    # Perfect fit gets absolute priority with tie-breaker
    perfect_fit_priority = np.where(
        perfect_fit,
        3.0 + np.random.uniform(0, 0.0005, size=perfect_fit.shape),
        0.0
    )
    
    # Enhanced utilization scoring with sigmoid-like rewards
    util_exponent = 2.5 + 3 * system_fill_ratio  # More dynamic range than v1
    space_priority = np.power(future_utilization, util_exponent)
    
    # Dynamic bin size adjustment with more nuanced behavior
    size_ratio = bins_valid / median_bin
    size_adjustment = np.zeros_like(size_ratio)
    
    if mean_utilization < 0.2:
        # Early stage: strong preference for smaller bins
        size_adjustment = -0.4 * size_ratio
    elif mean_utilization > 0.9:
        # Late stage: minimal size preference
        size_adjustment = -0.02 * size_ratio
    else:
        # Normal operation: gradual transition
        # Quadratic interpolation between early and late stages
        t = (mean_utilization - 0.2) / 0.7
        penalty_strength = 0.4 * (1 - t) + 0.02 * t
        size_adjustment = -penalty_strength * size_ratio
    
    # Utilization bonuses with non-linear rewards
    util_bonus = np.zeros_like(future_utilization)
    thresholds = [0.5, 0.7, 0.85, 0.925, 0.975]
    bonuses = [0.02, 0.05, 0.1, 0.15, 0.25]  # More aggressive top bonuses
    
    # Apply bonuses with smoothing around thresholds
    for threshold, bonus in zip(thresholds, bonuses):
        # Smooth transition around threshold
        distance = np.maximum(0, future_utilization - (threshold - 0.05))
        transition = np.minimum(1, distance / 0.1)
        util_bonus += bonus * transition
    
    # Diversity bonus to occasionally favor less-utilized bins
    if num_bins > 5 and system_fill_ratio < 0.7:
        diversity_bonus = 0.05 * (1 - current_utilization)
    else:
        diversity_bonus = 0.0
    
    # Combine all components with adaptive weights
    base_priority = (
        perfect_fit_priority 
        + space_priority 
        + util_bonus 
        + diversity_bonus
        + size_adjustment
    )
    
    # Adaptive noise scaling based on system state and priority magnitude
    avg_priority = np.mean(base_priority[~perfect_fit]) if np.any(~perfect_fit) else 0
    noise_scale = 0.0005 * (1 + 2*system_fill_ratio) * (1 + avg_priority)
    priorities_valid = base_priority + np.random.uniform(
        -noise_scale, noise_scale, size=base_priority.shape
    )
    
    priorities[valid_mask] = priorities_valid
    return priorities



