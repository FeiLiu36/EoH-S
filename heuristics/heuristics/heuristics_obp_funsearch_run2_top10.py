# Top 10 functions for funsearch run 2

# Function 1 - Score: -0.03911015984090625
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    EPSILON = 1e-10
    can_fit = (bins >= item)
    remaining = np.maximum(bins - item, 0)
    
    # 1. Dynamic ideal remaining calculation (from v1 but enhanced)
    if item < 0.05:  # Very small items
        ideal_remaining = np.percentile(bins[bins >= item], 10) if np.any(bins >= item) else item
    elif item > 0.95:  # Very large items
        ideal_remaining = item * 0.1
    else:
        # Base ideal combines v0's size-adaptation with v1's distribution awareness
        base_ideal = item / np.e  # Information-theoretic optimal
        
        # Distribution-aware adjustment (from v1)
        if len(bins) > 1 and np.any(bins >= item):
            avg_capacity = np.mean(bins[bins >= item])
            std_capacity = np.std(bins[bins >= item])
            distribution_factor = np.clip(std_capacity / (avg_capacity + EPSILON), 0, 2)
            ideal_remaining = base_ideal * (1 + 0.2 * distribution_factor)
        else:
            ideal_remaining = base_ideal
    
    # 2. Priority components (combining best of v0 and v1)
    # Exact fits get maximum priority
    exact_fit = (remaining < EPSILON)
    
    # Smooth remaining capacity priority (Gaussian-like from v0)
    sigma = ideal_remaining * 0.5
    remaining_priority = np.exp(-0.5 * ((remaining - ideal_remaining) / sigma) ** 2)
    
    # Capacity bonus system (from v1)
    capacity_bonus = np.where(
        bins >= (item * 1.5), 0.05,
        np.where(bins >= (item * 1.2), 0.02, 0.0)
    )
    
    # Penalty for nearly-full bins (from v1)
    nearly_full = remaining < (item * 0.15)
    
    # 3. Combined priority calculation
    priorities = np.where(
        exact_fit,
        1000.0,  # Highest priority for exact fits
        np.where(
            can_fit,
            (remaining_priority * 2.0 +      # Main factor
             capacity_bonus * 1.0 -          # Future capacity bonus
             np.where(nearly_full, 0.5, 0.0)),  # Nearly-full penalty
            -np.inf  # Can't fit
        )
    )
    
    return priorities



# Function 2 - Score: -0.03922084412315429
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
    
    if not np.any(valid_mask):
        return priorities
    
    # Calculate system statistics
    bins_valid = bins[valid_mask]
    remaining_valid = remaining[valid_mask]
    current_fill = 1 - (remaining_valid / bins_valid)
    global_fill = np.mean(current_fill) if len(current_fill) > 0 else 0
    mean_bin_size = np.mean(bins)
    item_ratio = item / mean_bin_size
    
    # Adaptive parameters with more robust scaling
    params = {
        # Fit quality parameters (dynamic based on system state)
        'exact_fit_boost': 150.0 * (1 + global_fill**2),
        'near_fit_threshold': max(0.01, 0.05 * (1 + global_fill)),
        'near_fit_boost': 40.0 * (1 - 0.5*global_fill),
        'good_fit_threshold': 0.15 + 0.1*global_fill,
        'fit_tightness_scale': 4.0 + 4*global_fill,
        
        # Size-aware parameters
        'size_ratio_weight': 0.2 * min(1.0, item_ratio),
        'bin_size_preference': 0.1 * (1 + np.log1p(item_ratio)),
        
        # Penalty system with smoother transitions
        'penalty_threshold': 0.1 + 0.1*global_fill,
        'penalty_scale': 1.5 + 2*global_fill,
        'penalty_smoothness': 3.0,
        
        # Global considerations
        'fill_balance_weight': 0.3 * (1 - 0.5*global_fill),
        'diversity_weight': 0.15 + 0.1*(1-global_fill),
        
        # Stability parameters
        'min_normalization': 1e-6,
        'max_boost': 200.0
    }
    
    # Normalize remaining space (more robust handling)
    remaining_ratio = remaining_valid / np.maximum(bins_valid, params['min_normalization'])
    fill_ratio = 1 - remaining_ratio
    
    # 1. Fit quality components (with more gradual transitions)
    exact_fit_mask = remaining_valid < params['min_normalization']
    near_fit_mask = remaining_ratio < params['near_fit_threshold']
    good_fit_mask = remaining_ratio < params['good_fit_threshold']
    
    # 2. Dynamic fit priority (sigmoid instead of exponential for better control)
    fit_priority = 1 / (1 + np.exp(-params['fit_tightness_scale'] * (1 - remaining_ratio)))
    
    # 3. Enhanced penalty system (smooth polynomial instead of sharp cutoff)
    penalty = np.where(
        remaining_ratio < params['penalty_threshold'],
        params['penalty_scale'] * 
        np.power((params['penalty_threshold'] - remaining_ratio), params['penalty_smoothness']),
        0
    )
    
    # 4. Size-aware bin preference (logarithmic scaling)
    max_bin = np.max(bins)
    bin_size_preference = (
        np.log1p(bins_valid / mean_bin_size) * params['bin_size_preference']
    )
    
    # 5. Fill balance and diversity components
    fill_balance = 1 - np.abs(fill_ratio - global_fill) * params['fill_balance_weight']
    fill_diversity = 1 - np.abs(fill_ratio - 0.5) * params['diversity_weight']
    
    # 6. Size ratio consideration (accounts for relative item size)
    size_ratio_factor = 1 + params['size_ratio_weight'] * (1 - remaining_ratio)
    
    # Combine all components with controlled boosting
    base_score = (
        (exact_fit_mask * params['exact_fit_boost']) +
        (near_fit_mask * params['near_fit_boost']) +
        (good_fit_mask * fit_priority * 3) +
        (fit_priority * 2) -
        penalty +
        bin_size_preference
    )
    
    # Apply balancing factors with clipping to prevent extreme values
    priorities[valid_mask] = np.clip(
        base_score * fill_balance * fill_diversity * size_ratio_factor,
        -params['max_boost'],
        params['max_boost']
    )
    
    return priorities



# Function 3 - Score: -0.03961726775178735
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    priorities = np.full_like(bins, -np.inf)
    remaining_space = bins - item
    
    # Handle edge cases
    if item <= 0:
        return np.zeros_like(bins)  # Zero/negative items can go anywhere
    if len(bins) == 0:
        return priorities
    
    # Only consider bins that can fit the item
    valid_mask = remaining_space >= 0
    valid_remaining = remaining_space[valid_mask]
    
    if valid_remaining.size == 0:
        return priorities
    
    # Dynamic normalization factors
    max_bin = np.max(bins)
    scale = max(max_bin, item, 1.0)  # Ensure scale is at least 1.0
    norm_remaining = valid_remaining / scale
    norm_bins = bins[valid_mask] / scale
    norm_item = item / scale
    
    # 1. Primary best-fit component (inverse remaining space)
    # Smoother gradient with log transform
    best_fit = np.log1p(1.0 / (norm_remaining + 1e-12))
    
    # 2. Exact fit detection (with dynamic tolerance)
    exact_fit_tol = min(1e-8, norm_item * 1e-4)
    exact_fit = np.isclose(norm_remaining, 0, atol=exact_fit_tol)
    
    # 3. Nearly-full bonus (smooth transition with adjustable steepness)
    nearly_full_factor = 12 / max(norm_item, 0.01)
    nearly_full = 1 / (1 + np.exp(nearly_full_factor * (norm_remaining - norm_item*1.5)))
    
    # 4. Bin utilization score (encourage balanced utilization)
    # Dual peaks at 30% and 70% to prevent fragmentation
    utilization = 0.5 * (np.exp(-((norm_bins - 0.3)/0.2)**2) + 
                        np.exp(-((norm_bins - 0.7)/0.2)**2))
    
    # 5. Size-proportional bonus (prefer smaller bins when equal fit)
    size_adjusted = 1 / (norm_bins + norm_item + 1e-5)
    
    # 6. Diversity bonus (spread items across bins)
    # Calculate based on position to encourage using different bins
    valid_indices = np.where(valid_mask)[0]
    diversity = 0.5 / (1 + valid_indices)  # Gradually decreasing bonus
    
    # Combine components with optimized weights
    combined = (
        0.65 * best_fit +                    # Primary best-fit (65%)
        1e6 * exact_fit +                    # Absolute priority for exact fits
        0.20 * nearly_full +                 # Nearly-full bonus (20%)
        0.10 * utilization +                 # Utilization balancing (10%)
        0.03 * size_adjusted +               # Size awareness (3%)
        0.02 * diversity                     # Diversity (2%)
    )
    
    # Numerical stability transformations
    priorities[valid_mask] = np.clip(combined, 0, 1e10)  # More efficient than nested where
    
    return priorities



# Function 4 - Score: -0.03963454021178764
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_space = bins - item
    
    # Initialize priorities (can't fit by default)
    priorities = np.full_like(bins, -np.inf)
    valid_mask = remaining_space >= 0
    
    if np.any(valid_mask):
        valid_bins = bins[valid_mask]
        valid_remaining = remaining_space[valid_mask]
        
        # 1. Space efficiency (primary factor)
        # Using inverse with smoothing for better gradient
        epsilon = np.finfo(float).eps * 100  # Slightly larger epsilon
        space_efficiency = 1 / (valid_remaining + epsilon)
        
        # 2. Current utilization (secondary factor)
        current_utilization = 1 - (valid_remaining / valid_bins)
        # Non-linear scaling that emphasizes high utilization
        utilization_factor = np.power(current_utilization, 1.2)
        
        # 3. Bin size factor (tie-breaker)
        # Prefer smaller bins to keep larger bins available
        bin_size_factor = 1 - (valid_bins / np.max(valid_bins))
        
        # 4. Future packing potential
        # Estimate how many average items could fit after this one
        avg_item_size = np.clip(item * 0.5, 0.01, 0.1)  # Conservative estimate
        future_packing = np.floor(valid_remaining / avg_item_size)
        future_factor = np.log1p(future_packing)  # Logarithmic scaling
        
        # Combine factors with optimized weights
        combined_priority = (
            0.55 * space_efficiency +
            0.30 * utilization_factor +
            0.10 * bin_size_factor +
            0.05 * future_factor
        )
        
        priorities[valid_mask] = combined_priority
    
    # Highest priority for exact fits (with tolerance)
    exact_fit_mask = np.abs(remaining_space) < (1e-10 * bins)
    priorities = np.where(exact_fit_mask, np.inf, priorities)
    
    # Special case: if item is very small compared to bin size
    # we should slightly prefer bins that are already more full
    if item < 0.01 * np.max(bins):
        full_bins_mask = (bins - remaining_space) > 0.9 * bins
        priorities = np.where(full_bins_mask & valid_mask, 
                            priorities[valid_mask] + 0.2, 
                            priorities)
    
    return priorities



# Function 5 - Score: -0.03963595769650076
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    priorities = np.full_like(bins, -np.inf)
    remaining_space = bins - item
    
    # Only consider bins that can fit the item
    valid_mask = remaining_space >= 0
    valid_remaining = remaining_space[valid_mask]
    
    if not np.any(valid_mask):
        return priorities
    
    # Normalization based on problem scale
    max_bin = np.max(bins)
    scale_factor = max(max_bin, 1.0)  # Ensure we don't divide by 0
    
    # Normalized values
    norm_remaining = valid_remaining / scale_factor
    norm_bins = bins[valid_mask] / scale_factor
    norm_item = item / scale_factor
    
    # 1. Core best-fit component (primary driver)
    best_fit = 1.0 / (norm_remaining + 1e-10)
    
    # 2. Exact fit bonus with adaptive tolerance
    exact_fit_tol = max(1e-10, 1e-6 * norm_item)
    exact_fit_bonus = np.where(norm_remaining <= exact_fit_tol, 1000.0, 0.0)
    
    # 3. Size-aware adjustment (prefer appropriately sized bins)
    size_adjustment = 1.0 / (1 + np.power(norm_bins, 0.8))
    
    # 4. Utilization bonus (encourage using existing bins)
    utilization = 1 - norm_remaining / (norm_bins + 1e-10)
    utilization_bonus = np.sqrt(utilization)  # Softer than linear
    
    # 5. Consolidation penalty (discourage very empty bins)
    consolidation_penalty = -0.3 * np.exp(-5 * norm_bins)
    
    # 6. Nearly-full bonus (exponential decay)
    nearly_full_threshold = 0.1 * (1 + norm_item)  # Dynamic threshold
    nearly_full_bonus = np.where(
        norm_remaining < nearly_full_threshold,
        2.0 * np.exp(-5 * norm_remaining),
        0
    )
    
    # Combine components with fixed weights (simpler than v0's dynamic weights)
    combined = (
        0.50 * best_fit +
        exact_fit_bonus +
        0.20 * size_adjustment +
        0.15 * utilization_bonus +
        0.10 * nearly_full_bonus +
        consolidation_penalty
    )
    
    # Final transformation with softmax-like scaling
    priorities[valid_mask] = np.sign(combined) * np.log1p(np.abs(combined))
    
    return priorities



# Function 6 - Score: -0.0396435688474268
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
    valid_mask = remaining >= -1e-10  # Small tolerance for floating point
    
    if not np.any(valid_mask):
        return priorities
    
    # Dynamic configuration based on problem state
    total_capacity = np.sum(bins)
    filled = total_capacity - np.sum(remaining[valid_mask])
    utilization = filled / total_capacity
    avg_bin_size = np.mean(bins)
    
    # Parameters that adapt based on global state
    exact_fit_boost = 100.0 * (1 + utilization)  # More important as we fill up
    near_fit_boost = 50.0 * (1 - utilization)    # More important early on
    log_scale = 10.0 * (item / avg_bin_size)     # Scale with relative item size
    size_weight = 1e-5 * (1 + utilization)       # Prefer smaller bins as we fill up
    frag_penalty_weight = 0.5 * utilization      # Fragmentation matters more when utilization is high
    
    # Get valid bins and their properties
    remaining_valid = remaining[valid_mask]
    bins_valid = bins[valid_mask]
    utilizations = (bins_valid - remaining_valid) / bins_valid
    
    # Detect exact fits (with floating point tolerance)
    exact_fit_mask = np.abs(remaining_valid) < 1e-10
    
    # Detect near fits (within 5% of bin size)
    near_fit_threshold = 0.05 * bins_valid
    near_fit_mask = (~exact_fit_mask) & (remaining_valid <= near_fit_threshold)
    
    # Calculate fragmentation penalty (discourage leaving small leftover spaces)
    frag_penalty = np.where(
        remaining_valid > 1e-10,
        np.exp(-remaining_valid / (0.1 * avg_bin_size)),
        0
    )
    
    # Main components
    log_priorities = np.log(1/(remaining_valid + 1e-10))  # Prefer bins with least remaining space
    
    # Utilization preference (smooth curve using sigmoid)
    util_priority = 1 / (1 + np.exp(-10 * (utilizations - 0.5)))
    
    # Combine all components
    priorities[valid_mask] = (
        (exact_fit_mask * exact_fit_boost) +            # Highest priority for exact fits
        (near_fit_mask * near_fit_boost) +              # High priority for near fits
        (log_priorities * log_scale) +                  # Logarithmic space preference
        (util_priority * log_scale * 0.3) +             # Utilization preference
        (bins_valid * size_weight) -                    # Small bin size preference
        (frag_penalty * frag_penalty_weight * log_scale) # Fragmentation penalty
    )
    
    return priorities



# Function 7 - Score: -0.039646182927871185
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
    
    if not np.any(valid_mask):
        return priorities
    
    # Parameters - carefully tuned defaults
    params = {
        'exact_fit_boost': 15.0,           # Very strong boost for perfect fits
        'near_fit_threshold': 0.03,        # Tighter threshold for near-exact fits
        'near_fit_boost': 8.0,             # Strong boost for near-exact fits
        'empty_bin_boost': 3.0,            # Increased boost for empty bins
        'base_scale': 1.2,                 # Slightly increased base scaling
        'penalty_start': 0.05,             # Sooner penalty start
        'penalty_max': 0.7,                # Lower maximum penalty threshold
        'global_fill_weight': 0.25,        # Balanced global consideration
        'nonlinear_exp': 0.6,              # More aggressive nonlinear scaling
        'small_item_threshold': 0.01,      # Threshold for considering item small
        'small_item_penalty': 0.5,         # Penalty for putting small items in large bins
    }
    
    remaining_valid = remaining[valid_mask]
    bins_valid = bins[valid_mask]
    fill_ratios = 1 - (remaining_valid / bins_valid)
    
    # Calculate global state metrics
    global_fill = np.mean(fill_ratios)
    adaptive_scale = 1 + params['global_fill_weight'] * global_fill
    
    # Normalize remaining space (handle zero case)
    max_remaining = remaining_valid.max()
    normalized_remaining = remaining_valid / (max_remaining + 1e-10)
    
    # Base score - nonlinear preference for least remaining space
    base_score = np.power(1/(normalized_remaining + 1e-10), params['nonlinear_exp'])
    
    # Exact and near-exact fits
    exact_fit_mask = remaining_valid == 0
    near_fit_mask = (remaining_valid / bins_valid) < params['near_fit_threshold']
    
    # Progressive penalty for nearly-full bins (smooth curve)
    remaining_ratio = remaining_valid / bins_valid
    penalty = np.where(
        remaining_ratio < params['penalty_max'],
        np.power(
            np.clip(
                (params['penalty_max'] - remaining_ratio) / 
                (params['penalty_max'] - params['penalty_start']),
                0, 1
            ),
            1.5  # Makes penalty curve smoother
        ),
        0
    )
    
    # Empty bin boost (helps reduce fragmentation)
    empty_bin_mask = (bins_valid == remaining_valid)
    empty_bin_boost = empty_bin_mask * params['empty_bin_boost']
    
    # Small item penalty - avoid putting tiny items in large bins
    small_item_penalty = np.where(
        (item < params['small_item_threshold']) & 
        (remaining_ratio > 0.5),
        params['small_item_penalty'],
        0
    )
    
    # Combine all components
    priorities[valid_mask] = (
        (exact_fit_mask * params['exact_fit_boost']) +      # Exact fits
        (near_fit_mask * params['near_fit_boost']) +        # Near-exact fits
        (base_score * params['base_scale']) -               # Base score
        penalty -                                           # Penalty
        small_item_penalty +                                # Small item penalty
        empty_bin_boost                                     # Empty bin preference
    ) * adaptive_scale                                      # Global adjustment
    
    return priorities



# Function 8 - Score: -0.0396472791514432
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    total_capacity = np.sum(bins)
    remaining_capacity = np.sum(np.maximum(0, bins - item))
    utilization = 1 - remaining_capacity / total_capacity if total_capacity > 0 else 0
    
    # Dynamic weights that change based on packing progress
    weights = {
        'exact_fit': 200.0 + 100.0 * item,  # Even higher bonus for exact fits
        'best_fit': 1.0,
        'nearly_full': 0.8 + 0.2 * utilization,  # More important as we fill up
        'early_bin': max(0.03, 0.15 - utilization * 0.12),  # Less important later
        'empty_penalty': 0.4 + 0.6 * utilization,  # More important to consolidate
        'variance_penalty': 0.3,  # Increased importance of balanced utilization
        'flexibility_bonus': 0.5 - 0.4 * utilization  # More important early
    }
    
    priorities = np.full_like(bins, -np.inf)  # Default to -inf (invalid)
    remaining_space = bins - item
    
    # Only consider bins that can fit the item
    valid_mask = remaining_space >= 0
    valid_remaining = remaining_space[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    valid_bins = bins[valid_mask]
    
    if valid_remaining.size == 0:
        return priorities
    
    # Base priority components
    base_priority = 1.0 / (valid_remaining + 1e-10)  # Best-fit component
    
    # Exact fit bonus (remaining_space = 0)
    exact_fit = np.isclose(valid_remaining, 0, atol=1e-10)
    exact_fit_bonus = np.where(exact_fit, weights['exact_fit'], 0.0)
    
    # Adaptive nearly-full bonus (threshold depends on item size and utilization)
    nearly_full_threshold = (0.05 + 0.15 * item) * (1 + utilization)
    nearly_full = valid_remaining <= nearly_full_threshold
    nearly_full_bonus = (nearly_full_threshold - valid_remaining) / nearly_full_threshold
    nearly_full_bonus = np.where(nearly_full, 
                                nearly_full_bonus * weights['nearly_full'], 
                                0.0)
    
    # Early bin preference (first-fit component)
    early_bin_bonus = (len(bins) - valid_indices) / len(bins) * weights['early_bin']
    
    # Penalty for leaving bins very empty (scaled by how empty)
    empty_ratio = valid_remaining / valid_bins
    empty_penalty = -weights['empty_penalty'] * np.sqrt(empty_ratio)
    
    # Variance penalty - prefer bins that would make utilization more balanced
    future_utilization = 1 - (valid_remaining / valid_bins)
    mean_util = np.mean(future_utilization)
    variance_penalty = -weights['variance_penalty'] * (future_utilization - mean_util)**2
    
    # Flexibility bonus - prefer bins that leave room for likely future items
    # Estimate future item size as average of current item and remaining capacity
    estimated_future_item = (item + np.mean(valid_remaining)) / 2
    flexibility = np.minimum(valid_remaining, estimated_future_item) / estimated_future_item
    flexibility_bonus = weights['flexibility_bonus'] * flexibility
    
    # Combine all components
    priorities[valid_mask] = (
        base_priority * weights['best_fit'] +
        exact_fit_bonus +
        nearly_full_bonus +
        early_bin_bonus +
        empty_penalty +
        variance_penalty +
        flexibility_bonus
    )
    
    # Special case for very small items - prefer to fill nearly-full bins
    if item < 0.05 * np.mean(bins):
        nearly_empty = valid_remaining > 0.9 * valid_bins
        priorities[valid_mask] = np.where(nearly_empty, 
                                        priorities[valid_mask] - 10.0, 
                                        priorities[valid_mask])
    
    return priorities



# Function 9 - Score: -0.03965510465959945
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    priorities = np.full_like(bins, -np.inf)
    remaining_space = bins - item
    
    # Only consider bins that can fit the item
    valid_mask = remaining_space >= 0
    if not np.any(valid_mask):
        return priorities
    
    valid_remaining = remaining_space[valid_mask]
    valid_bins = bins[valid_mask]
    
    # Robust normalization factors
    max_bin = np.max(bins)
    mean_bin = np.mean(bins[bins > 0]) if np.any(bins > 0) else max(1.0, item)
    scale_factor = max(max_bin, mean_bin, item, 1.0)
    
    # Normalized values (clipped for stability)
    norm_remaining = np.clip(valid_remaining / scale_factor, 1e-10, None)
    norm_bins = valid_bins / scale_factor
    norm_item = item / scale_factor
    
    # System state metrics
    total_remaining = np.sum(valid_remaining)
    total_capacity = np.sum(valid_bins)
    system_fill = 1 - (total_remaining / (total_capacity + 1e-10))
    item_relative_size = norm_item / (np.median(norm_bins) + 1e-10)
    
    # 1. Enhanced best-fit component
    best_fit = 1.0 / (norm_remaining + 0.01 * np.exp(-system_fill))
    
    # 2. Exact-fit detection with adaptive tolerance
    exact_fit_tol = max(1e-10, min(1e-5, norm_item * 1e-3))
    exact_fit_bonus = np.where(valid_remaining <= exact_fit_tol, 10000.0, 0.0)
    
    # 3. Utilization with adaptive non-linearity
    utilization = (valid_bins - valid_remaining) / valid_bins
    utilization_score = np.where(
        utilization > 0.9,
        3.0 * utilization,  # Strong reward for nearly full bins
        np.power(utilization, 0.7 - 0.2 * system_fill)  # Adaptive exponent
    )
    
    # 4. Load balancing (prefer less loaded bins)
    load_balance = 1.0 - (valid_remaining / (total_remaining + 1e-10))
    
    # 5. Size appropriateness (prefer bins of similar size)
    size_appropriateness = np.exp(-2 * np.abs(norm_bins - norm_item))
    
    # Dynamic weights based on system state
    w_best_fit = 0.5 - 0.2 * system_fill
    w_utilization = 0.3 + 0.3 * system_fill
    w_balance = 0.15
    w_size = 0.05 + 0.1 * (1 - item_relative_size)
    
    # Combine components
    combined = (
        w_best_fit * best_fit +
        exact_fit_bonus +
        w_utilization * utilization_score +
        w_balance * load_balance +
        w_size * size_appropriateness
    )
    
    # Final transformation
    priorities[valid_mask] = np.sign(combined) * np.log1p(10 * np.abs(combined))
    
    return priorities



# Function 10 - Score: -0.0396566579029438
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    total_capacity = np.sum(bins)
    remaining_capacity = np.sum(np.maximum(0, bins))
    utilization = 1 - (remaining_capacity / total_capacity) if total_capacity > 0 else 0
    
    # Dynamic weights that change based on packing phase (early vs late)
    phase = min(1.0, max(0.0, utilization / 0.8))  # 0-1 where 1 = late phase
    
    weights = {
        'exact_fit': 200.0 + 100.0 * item,  # Even higher bonus for exact fits
        'best_fit': 1.0 - 0.2 * phase,  # Less important later
        'nearly_full': 0.8 + 0.5 * phase,  # More important as we fill up
        'early_bin': max(0.03, 0.15 - phase * 0.12),  # Less important later
        'empty_penalty': 0.4 + 0.8 * phase,  # More important to consolidate later
        'variance_penalty': 0.3 + 0.2 * phase,  # More important to balance later
        'fragmentation_penalty': 0.5 * phase  # Penalize creating small gaps
    }
    
    priorities = np.full_like(bins, -np.inf)  # Default to -inf (invalid)
    remaining_space = bins - item
    
    # Only consider bins that can fit the item
    valid_mask = remaining_space >= 0
    valid_remaining = remaining_space[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    valid_bins = bins[valid_mask]
    
    if valid_remaining.size == 0:
        return priorities
    
    # Base priority components
    base_priority = 1.0 / (valid_remaining + 1e-10)  # Best-fit component
    
    # Enhanced exact fit detection with relative tolerance
    rel_tol = 1e-5 * item  # Scale tolerance with item size
    exact_fit = np.abs(valid_remaining) <= rel_tol
    exact_fit_bonus = np.where(exact_fit, weights['exact_fit'], 0.0)
    
    # Adaptive nearly-full bonus with dynamic threshold
    nearly_full_threshold = 0.15 + 0.15 * item - 0.1 * phase  # Adjust based on phase
    nearly_full = valid_remaining <= nearly_full_threshold
    nearly_full_bonus = (nearly_full_threshold - valid_remaining) / (nearly_full_threshold + 1e-10)
    nearly_full_bonus = np.where(nearly_full, 
                                nearly_full_bonus * weights['nearly_full'], 
                                0.0)
    
    # Early bin preference with phase-dependent decay
    early_bin_bonus = (len(bins) - valid_indices) / len(bins) * weights['early_bin']
    
    # Penalty for leaving bins very empty (scaled by how empty)
    empty_ratio = valid_remaining / valid_bins
    empty_penalty = -weights['empty_penalty'] * np.power(empty_ratio, 0.7)  # Softer penalty
    
    # Variance penalty - prefer bins that would make utilization more balanced
    future_utilization = 1 - (valid_remaining / valid_bins)
    mean_util = np.mean(future_utilization)
    variance_penalty = -weights['variance_penalty'] * (future_utilization - mean_util)**2
    
    # Fragmentation penalty - avoid creating small remaining spaces
    fragmentation_penalty = -weights['fragmentation_penalty'] * np.exp(-valid_remaining * 5)
    
    # Combine all components
    priorities[valid_mask] = (
        base_priority * weights['best_fit'] +
        exact_fit_bonus +
        nearly_full_bonus +
        early_bin_bonus +
        empty_penalty +
        variance_penalty +
        fragmentation_penalty
    )
    
    # Special case: if all valid bins are equally good (within tolerance), prefer earlier ones
    if len(valid_indices) > 1:
        max_priority = np.max(priorities[valid_mask])
        close_to_max = priorities[valid_mask] >= (max_priority - 0.01 * max_priority)
        if np.sum(close_to_max) > 1:
            priorities[valid_mask] = np.where(
                close_to_max,
                priorities[valid_mask] + (len(bins) - valid_indices) * 0.001,
                priorities[valid_mask]
            )
    
    return priorities



