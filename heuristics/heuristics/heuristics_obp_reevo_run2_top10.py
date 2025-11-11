# Top 10 functions for reevo run 2

# Function 1 - Score: -0.03838978794007594
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
    
    # System characteristics
    system_mean = np.mean(bins)
    system_max = np.max(bins)
    system_std = np.std(bins)
    item_ratio = item / system_mean
    
    # Dynamic strategy parameters (adaptive based on system state)
    fit_balance = np.clip(0.6 - 0.55 * item_ratio + 0.1 * (system_std/system_mean), 0.2, 0.9)
    size_sensitivity = 0.6 + 0.4 * np.tanh(2.5 - 5 * item_ratio)
    
    # Normalized metrics with stability guards
    normalized_space = remaining_space / (bins + 1e-8)
    utilization = 1 - normalized_space
    
    # 1. Adaptive hybrid fit scoring with dynamic coefficients
    best_fit = np.exp(-7 * normalized_space)
    worst_fit = np.tanh(remaining_space / (0.4 * system_mean + 0.1 * system_std))
    fit_score = fit_balance * best_fit + (1 - fit_balance) * worst_fit
    
    # 2. Multi-tier utilization rewards with adaptive thresholds
    utilization_thresholds = np.array([0.6, 0.8, 0.95])
    utilization_thresholds *= (1 + 0.1 * (system_std/system_mean))  # Adjust based on system diversity
    
    utilization_reward = np.where(
        utilization > utilization_thresholds[2],
        5 * utilization ** 7,  # Critical tier
        np.where(
            utilization > utilization_thresholds[1],
            2.5 * utilization ** 4,  # High tier
            np.where(
                utilization > utilization_thresholds[0],
                utilization ** 2,  # Medium tier
                utilization ** 0.7  # Baseline (sub-linear)
            )
        )
    )
    
    # 3. Size-aware bin weighting with non-linear sensitivity
    size_weights = (1.4 - 0.8 * (bins / system_max) ** size_sensitivity)
    
    # 4. Dynamic packing factor (item-size dependent)
    packing_factor = np.where(
        item_ratio > 0.4,
        1 + 0.25 * (1 - utilization),  # Large items favor tighter packing
        1 - 0.15 * utilization  # Small items favor spreading
    )
    
    # Combine components with system-awareness
    priorities = np.where(
        valid_mask,
        fit_score * utilization_reward * size_weights * packing_factor,
        -np.inf
    )
    
    # Special case handling with dynamic thresholds
    perfect_fit = remaining_space == 0
    priorities[perfect_fit] = np.inf
    
    near_perfect_threshold = 0.015 * system_mean + 0.005 * system_std
    near_perfect = (remaining_space > 0) & (remaining_space < near_perfect_threshold)
    priorities[near_perfect] *= 3.0
    
    # Micro-item distribution strategy
    micro_threshold = 0.025 * system_mean
    if item < micro_threshold:
        spread_factor = 1 + 0.5 * (1 - normalized_space[valid_mask])
        priorities[valid_mask] *= spread_factor
    
    # Controlled noise injection proportional to system diversity
    diversity_ratio = system_std / system_mean
    if diversity_ratio > 0.08:
        noise_scale = 0.03 * diversity_ratio * (np.max(priorities[valid_mask]) - np.min(priorities[valid_mask]))
        priorities[valid_mask] += np.random.normal(0, noise_scale, size=np.sum(valid_mask))
    
    return priorities



# Function 2 - Score: -0.0385669309618098
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
    
    # Enhanced system state analysis with sharper metrics
    active_bins = bins[valid_mask]
    total_capacity = np.sum(active_bins) if any(valid_mask) else np.sum(bins)
    system_utilization = 1 - np.sum(remaining_space[valid_mask]) / total_capacity if any(valid_mask) else 0.5
    median_capacity = np.median(active_bins) if any(valid_mask) else np.median(bins)
    capacity_variation = np.std(active_bins) / (median_capacity + 1e-8) if any(valid_mask) else 1.0
    
    # Ultra-sharp dynamic strategy parameters
    item_ratio = item / (median_capacity + 1e-8)
    fit_balance = 0.35 + 0.45 * np.tanh(5 * (system_utilization - 0.5))  # Very sharp transition
    capacity_sensitivity = 0.7 * (1 + np.exp(-4 * capacity_variation**0.8))  # Strong non-linear relationship
    
    # Multi-dimensional fit scoring with enhanced non-linearity
    normalized_space = remaining_space / (bins + 1e-8)
    
    # Adaptive best-fit with quadratic utilization dependence
    best_fit = np.exp(-(3 + 5 * system_utilization**1.8) * normalized_space**0.9)
    
    # Fragmentation-aware worst-fit with dynamic curvature
    worst_fit = 1 - normalized_space**(1.4 + 0.6 * system_utilization**2)
    
    # Hybrid fit score with power-law blending
    fit_score = (fit_balance**3 * best_fit**3 + (1 - fit_balance)**3 * worst_fit**3)**(1/3)
    
    # Ultra-sharp multi-tier utilization reward
    utilization = 1 - normalized_space
    utilization_reward = np.where(
        utilization > 0.96,
        6 * utilization**12,  # Extreme bonus
        np.where(
            utilization > 0.82,
            3 * utilization**6,  # High bonus
            np.where(
                utilization > 0.62,
                utilization**2.5,  # Medium bonus
                utilization**0.9  # Low utilization
            )
        )
    )
    
    # Capacity weighting with ultra-adaptive sensitivity
    capacity_weights = (bins / (median_capacity + 1e-8))**capacity_sensitivity
    capacity_weights = 2 / (1 + np.exp(-2 * capacity_weights)) - 1  # Sharper sigmoid
    
    # Combine components with dynamic exponents
    combined_score = (fit_score**1.2 * utilization_reward**1.1 * capacity_weights**0.9)**(1/1.1)
    priorities = np.where(valid_mask, combined_score, -np.inf)
    
    # Perfect fit detection with dynamic scaling bonuses
    perfect_fit_mask = np.abs(remaining_space) < 1e-8
    near_perfect_mask = (remaining_space > 0) & (normalized_space < 0.008)
    
    if any(valid_mask):
        max_score = np.max(priorities[valid_mask])
        priorities[perfect_fit_mask] = 4.0 * max_score**1.1 + 3.0
        priorities[near_perfect_mask] = 2.5 * max_score**1.05 + 2.0
    
    # Ultra-smart tie-breaking with system-adaptive noise
    if any(valid_mask):
        score_range = np.ptp(priorities[valid_mask])
        noise_scale = 0.0015 * score_range * (1 + 0.5 * capacity_variation**0.6)
        priorities[valid_mask] += np.random.normal(0, noise_scale, size=np.sum(valid_mask))
    
    # Small item adjustment with dynamic power-law spreading
    if item < 0.01 * median_capacity and any(valid_mask):
        spreading_factor = 1 + 0.8 * (1 - system_utilization)**2
        priorities[valid_mask] *= spreading_factor**0.8
    
    return priorities



# Function 3 - Score: -0.038607187474459896
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
    
    # Dynamic system characteristics
    system_mean = np.mean(bins)
    system_std = np.std(bins)
    diversity_ratio = system_std / (system_mean + 1e-8)
    item_ratio = item / (system_mean + 1e-8)
    
    # Adaptive strategy parameters
    fit_strategy = np.clip(0.7 - 0.6 * item_ratio + 0.2 * diversity_ratio, 0.1, 0.95)
    size_adaptivity = 0.5 + 0.5 * np.tanh(3 - 6 * item_ratio)
    
    # Core metrics
    normalized_space = remaining_space / (bins + 1e-8)
    utilization = 1 - normalized_space
    
    # 1. Hybrid fit strategy with smooth transitions
    best_fit = np.exp(-8 * normalized_space)
    worst_fit = np.tanh(remaining_space / (0.3 * system_mean + 0.2 * system_std))
    fit_score = fit_strategy * best_fit + (1 - fit_strategy) * worst_fit
    
    # 2. Dynamic tiered rewards with auto-adjusted thresholds
    tier_thresholds = 0.65 + 0.25 * np.array([0, 0.5, 0.9]) * diversity_ratio
    tier_rewards = np.where(
        utilization > tier_thresholds[2],
        6 * utilization**8,  # Critical tier
        np.where(
            utilization > tier_thresholds[1],
            3 * utilization**5,  # High tier
            np.where(
                utilization > tier_thresholds[0],
                utilization**3,  # Medium tier
                utilization**0.5  # Baseline (sub-linear)
            )
        )
    )
    
    # 3. Size-aware bin weighting with adaptive curve
    bin_size_factor = (1.5 - 0.9 * (bins / np.max(bins))**size_adaptivity)
    
    # 4. Dynamic packing strategy
    packing_factor = np.where(
        item_ratio > 0.35,
        1 + 0.3 * (1 - utilization)**2,  # Large items
        1 - 0.2 * utilization**1.5  # Small items
    )
    
    # Combine components
    priorities = np.where(
        valid_mask,
        fit_score * tier_rewards * bin_size_factor * packing_factor,
        -np.inf
    )
    
    # Special case handling
    perfect_fit = remaining_space == 0
    priorities[perfect_fit] = np.inf
    
    near_perfect = (remaining_space > 0) & (remaining_space < 0.02 * system_mean)
    priorities[near_perfect] *= 3.5
    
    # Micro-item distribution
    if item < 0.03 * system_mean:
        priorities[valid_mask] *= (1 + 0.7 * (1 - normalized_space[valid_mask]))
    
    # Diversity-driven exploration
    if diversity_ratio > 0.1:
        noise_magnitude = 0.04 * diversity_ratio * np.std(priorities[valid_mask])
        priorities[valid_mask] += np.random.normal(0, noise_magnitude, size=np.sum(valid_mask))
    
    return priorities



# Function 4 - Score: -0.03865322396486172
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
    
    # System state analysis with sharper metrics
    active_bins = bins[valid_mask]
    total_capacity = np.sum(active_bins) if any(valid_mask) else np.sum(bins)
    system_utilization = 1 - np.sum(remaining_space[valid_mask]) / total_capacity if any(valid_mask) else 0.5
    median_capacity = np.median(active_bins) if any(valid_mask) else np.median(bins)
    capacity_variation = np.std(active_bins) / (median_capacity + 1e-8) if any(valid_mask) else 1.0
    
    # Dynamic parameters with exponential transitions
    fit_balance = 0.3 + 0.5 * (1 - np.exp(-5 * (system_utilization - 0.5)))
    capacity_sensitivity = 0.5 * np.exp(-2 * capacity_variation)
    
    # Non-linear fit scoring with dynamic exponents
    normalized_space = remaining_space / (bins + 1e-8)
    best_fit = np.exp(-(3.0 * system_utilization**1.5) * normalized_space)
    worst_fit = 1 - normalized_space**(1.5 + system_utilization**2)
    fit_score = (fit_balance * best_fit + (1 - fit_balance) * worst_fit**2)**0.5
    
    # Adaptive utilization rewards with sharper thresholds
    utilization = 1 - normalized_space
    utilization_reward = np.where(
        utilization > 0.95,
        6 * utilization**8,
        np.where(
            utilization > 0.8,
            3 * utilization**4,
            np.where(
                utilization > 0.6,
                utilization**1.5,
                utilization**0.7
            )
        )
    )
    
    # Capacity weighting with power-law scaling
    capacity_weights = (bins / (median_capacity + 1e-8))**(1 - capacity_sensitivity)
    
    # Combined score with overflow protection
    combined_score = fit_score * utilization_reward * capacity_weights
    priorities = np.where(valid_mask, combined_score, -np.inf)
    
    # Perfect fit bonuses with dynamic scaling
    perfect_fit_mask = np.abs(remaining_space) < 1e-8
    if any(perfect_fit_mask):
        priorities[perfect_fit_mask] = 4.0 * np.max(priorities[valid_mask]) + 3.0
    
    # System-aware noise with adaptive magnitude
    if any(valid_mask) and capacity_variation > 0.1:
        noise_magnitude = 0.002 * np.ptp(priorities[valid_mask]) * capacity_variation**0.5
        priorities[valid_mask] += np.random.normal(0, noise_magnitude, size=np.sum(valid_mask))
    
    # Focused small-item handling
    if item < 0.01 * median_capacity and any(valid_mask):
        priorities[valid_mask] *= 1 + 0.5 * (1 - system_utilization)**2
    
    return priorities



# Function 5 - Score: -0.03870566132085258
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
    
    # System characteristics with adaptive normalization
    system_mean = np.mean(bins)
    system_std = np.std(bins)
    system_max = np.max(bins)
    system_diversity = system_std / (system_mean + 1e-8)
    item_ratio = item / system_mean
    
    # Dynamic strategy parameters with adaptive bounds
    fit_balance = 0.5 + 0.4 * np.tanh(2 - 4 * item_ratio)  # Tunes best-fit vs worst-fit
    size_sensitivity = 0.7 - 0.3 * np.exp(-3 * system_diversity)  # Bin size importance
    
    # Adaptive thresholds based on system state
    utilization_thresholds = np.array([0.6, 0.8, 0.95]) * (1 + 0.15 * system_diversity)
    perfect_fit_threshold = 0.02 * system_mean * (1 + system_diversity)
    
    # Core scoring components
    normalized_space = remaining_space / (bins + 1e-8)
    utilization = 1 - normalized_space
    
    # 1. Hybrid fit scoring with adaptive tightness
    best_fit = np.exp(-5 * (1 + item_ratio) * normalized_space)
    worst_fit = np.tanh(remaining_space / (0.3 * system_mean + 0.2 * system_std))
    fit_score = fit_balance * best_fit + (1 - fit_balance) * worst_fit
    
    # 2. Multi-tier utilization rewards with adaptive scaling
    utilization_reward = np.where(
        utilization > utilization_thresholds[2],
        4 * utilization ** 6,  # Critical tier
        np.where(
            utilization > utilization_thresholds[1],
            2 * utilization ** 3.5,  # High tier
            np.where(
                utilization > utilization_thresholds[0],
                utilization ** 1.5,  # Medium tier
                utilization ** 0.8  # Baseline
            )
        )
    )
    
    # 3. Size-aware bin weighting with diversity adaptation
    size_weights = (1.3 - 0.6 * (bins / system_max) ** size_sensitivity)
    
    # Combine components with validity mask
    priorities = np.where(
        valid_mask,
        fit_score * utilization_reward * size_weights,
        -np.inf
    )
    
    # Special case handling with adaptive thresholds
    perfect_fit = remaining_space == 0
    priorities[perfect_fit] = np.inf
    
    near_perfect = (remaining_space > 0) & (remaining_space < perfect_fit_threshold)
    priorities[near_perfect] *= 2.5
    
    # Small item distribution strategy
    if item_ratio < 0.1:
        spread_bonus = 1 + 0.6 * (1 - utilization[valid_mask])
        priorities[valid_mask] *= spread_bonus
    
    # Controlled noise injection proportional to system diversity
    if system_diversity > 0.1:
        score_range = np.max(priorities[valid_mask]) - np.min(priorities[valid_mask]) if any(valid_mask) else 1
        noise_scale = 0.02 * system_diversity * score_range
        priorities[valid_mask] += np.random.normal(0, noise_scale, size=np.sum(valid_mask))
    
    return priorities



# Function 6 - Score: -0.03878591114337961
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
    
    # Enhanced system state analysis
    active_bins = bins[valid_mask]
    total_capacity = np.sum(active_bins) if any(valid_mask) else np.sum(bins)
    system_utilization = 1 - np.sum(remaining_space[valid_mask]) / total_capacity if any(valid_mask) else 0.5
    median_capacity = np.median(active_bins) if any(valid_mask) else np.median(bins)
    capacity_variation = np.std(active_bins) / median_capacity if any(valid_mask) else 1.0
    
    # Dynamic strategy parameters with non-linear adaptation
    item_ratio = item / median_capacity if median_capacity > 0 else 1.0
    fit_balance = 0.4 + 0.4 * np.tanh(4 * (system_utilization - 0.55))  # Sharper transition
    capacity_sensitivity = 0.6 * (1 + np.exp(-3 * capacity_variation))  # Stronger inverse relationship
    
    # Multi-dimensional fit scoring
    normalized_space = remaining_space / (bins + 1e-8)
    
    # Adaptive best-fit with utilization-dependent decay (non-linear)
    best_fit = np.exp(-(2.5 + 4 * system_utilization**2) * normalized_space)
    
    # Fragmentation-aware worst-fit with dynamic curvature
    worst_fit = 1 - normalized_space**(1.3 + 0.7 * system_utilization**1.5)
    
    # Hybrid fit score with dynamic blending
    fit_score = (fit_balance * best_fit**2 + (1 - fit_balance) * worst_fit**2)**0.5
    
    # Multi-tier utilization reward with adaptive thresholds
    utilization = 1 - normalized_space
    utilization_reward = np.where(
        utilization > 0.95,
        5 * utilization**10,  # Critical bonus
        np.where(
            utilization > 0.8,
            2.5 * utilization**5,  # High utilization bonus
            np.where(
                utilization > 0.6,
                utilization**2,  # Medium bonus
                utilization**0.8  # Low utilization
            )
        )
    )
    
    # Capacity weighting with adaptive sensitivity and smooth normalization
    capacity_weights = (bins / (median_capacity + 1e-8))**capacity_sensitivity
    capacity_weights = 1.8 / (1 + np.exp(-1.5 * capacity_weights)) - 0.9  # Scaled sigmoid
    
    # Combine components with overflow handling
    combined_score = fit_score * utilization_reward * capacity_weights
    priorities = np.where(valid_mask, combined_score, -np.inf)
    
    # Enhanced perfect/near-perfect fit detection with dynamic bonuses
    perfect_fit_mask = np.abs(remaining_space) < 1e-8
    near_perfect_mask = (remaining_space > 0) & (normalized_space < 0.01)
    
    if any(valid_mask):
        max_score = np.max(priorities[valid_mask])
        priorities[perfect_fit_mask] = 3.5 * max_score + 2.5
        priorities[near_perfect_mask] = 2.2 * max_score + 1.5
    
    # Smart tie-breaking with system-aware adaptive noise
    if any(valid_mask):
        score_range = np.ptp(priorities[valid_mask])
        noise_scale = 0.002 * score_range * (1 + 0.4 * capacity_variation**0.7)
        priorities[valid_mask] += np.random.normal(0, noise_scale, size=np.sum(valid_mask))
    
    # Small item adjustment with dynamic spreading factor
    if item < 0.015 * median_capacity and any(valid_mask):
        spreading_factor = 1 + 0.6 * (1 - system_utilization)**1.5
        priorities[valid_mask] *= spreading_factor
    
    return priorities



# Function 7 - Score: -0.03881204857282224
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
    
    # System state analysis with enhanced stability
    active_bins = bins[valid_mask]
    total_capacity = np.sum(active_bins) if any(valid_mask) else np.sum(bins)
    system_utilization = 1 - np.sum(remaining_space[valid_mask]) / total_capacity if any(valid_mask) else 0.5
    median_capacity = np.median(active_bins) if any(valid_mask) else np.median(bins)
    capacity_variation = np.std(active_bins) / (median_capacity + 1e-8) if any(valid_mask) else 1.0
    
    # Dynamic strategy parameters with smooth adaptation
    item_ratio = item / (median_capacity + 1e-8)
    fit_balance = 0.5 + 0.4 * np.tanh(3 * (system_utilization - 0.6))  # Balanced transition
    capacity_sensitivity = 0.7 * (1 + np.exp(-4 * capacity_variation))  # Responsive to variation
    
    # Adaptive normalized space calculation
    normalized_space = remaining_space / (bins + 1e-8)
    utilization = 1 - normalized_space
    
    # Hybrid fit scoring with dynamic weighting
    best_fit = np.exp(-(2.0 + 2.5 * system_utilization) * normalized_space)
    worst_fit = 1 - normalized_space**(1.2 + 0.6 * system_utilization)
    fit_score = fit_balance * best_fit + (1 - fit_balance) * worst_fit
    
    # Multi-tier utilization rewards with adaptive thresholds
    utilization_reward = np.where(
        utilization > 0.95,
        4 * utilization**12,  # Critical tier
        np.where(
            utilization > 0.8,
            2 * utilization**6,  # High tier
            np.where(
                utilization > 0.6,
                utilization**2.5,  # Medium tier
                utilization  # Baseline
            )
        )
    )
    
    # Capacity-aware weighting with improved normalization
    capacity_weights = (bins / (median_capacity + 1e-8))**capacity_sensitivity
    capacity_weights = 1.6 / (1 + np.exp(-1.8 * capacity_weights)) - 0.8  # Optimized sigmoid
    
    # Combine components with stability
    combined_score = fit_score * utilization_reward * capacity_weights
    priorities = np.where(valid_mask, combined_score, -np.inf)
    
    # Progressive fit bonuses with dynamic scaling
    perfect_fit_mask = np.abs(remaining_space) < 1e-10
    near_perfect_mask = (remaining_space > 0) & (normalized_space < 0.01)
    
    if any(valid_mask):
        max_score = np.max(priorities[valid_mask])
        priorities[perfect_fit_mask] = 5.0 * max_score + 4.0  # Stronger perfect fit bonus
        priorities[near_perfect_mask] = 3.0 * max_score + 2.0  # Enhanced near-perfect bonus
    
    # System-aware tie-breaking with adaptive noise
    if any(valid_mask) and capacity_variation > 0.05:
        score_range = np.ptp(priorities[valid_mask])
        noise_scale = 0.0015 * score_range * (1 + 0.3 * capacity_variation)
        priorities[valid_mask] += np.random.normal(0, noise_scale, size=np.sum(valid_mask))
    
    # Dynamic item distribution strategies
    if item < 0.02 * median_capacity:  # Small items
        spreading_factor = 1 + 0.8 * (1 - system_utilization)**3  # More aggressive spreading
        priorities[valid_mask] *= spreading_factor
    elif item < 0.2 * median_capacity and system_utilization > 0.75:  # Medium items at high utilization
        priorities[valid_mask] *= 1 + 0.4 * (1 - normalized_space[valid_mask])**2  # Fragmentation prevention
    
    return priorities



# Function 8 - Score: -0.03883084046344201
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
    
    # Enhanced system state analysis
    active_bins = bins[valid_mask]
    total_capacity = np.sum(active_bins) if any(valid_mask) else np.sum(bins)
    system_utilization = 1 - np.sum(remaining_space[valid_mask]) / total_capacity if any(valid_mask) else 0.5
    median_capacity = np.median(active_bins) if any(valid_mask) else np.median(bins)
    capacity_variation = np.std(active_bins) / (median_capacity + 1e-8) if any(valid_mask) else 1.0
    
    # Dynamic strategy parameters with improved adaptation
    item_ratio = item / (median_capacity + 1e-8)
    fit_balance = 0.4 + 0.4 * np.tanh(4 * (system_utilization - 0.55))  # Smoother transition
    capacity_sensitivity = 0.6 * (1 + np.exp(-3 * capacity_variation))  # More responsive
    
    # Adaptive normalized space calculation
    normalized_space = remaining_space / (bins + 1e-8)
    
    # Enhanced hybrid fit scoring
    best_fit = np.exp(-(2.5 + 2 * system_utilization) * normalized_space)
    worst_fit = 1 - normalized_space**(1.3 + 0.5 * system_utilization)
    fit_score = fit_balance * best_fit + (1 - fit_balance) * worst_fit
    
    # Multi-stage utilization rewards with dynamic thresholds
    utilization = 1 - normalized_space
    utilization_reward = np.where(
        utilization > 0.95,
        5 * utilization**10,  # Critical bonus
        np.where(
            utilization > 0.8,
            2.5 * utilization**5,  # High utilization bonus
            np.where(
                utilization > 0.6,
                utilization**2,  # Medium bonus
                utilization**0.8  # Low utilization
            )
        )
    )
    
    # Improved capacity weighting with adaptive normalization
    capacity_weights = (bins / (median_capacity + 1e-8))**capacity_sensitivity
    capacity_weights = 1.8 / (1 + np.exp(-1.5 * capacity_weights)) - 0.9  # Better scaled sigmoid
    
    # Combine components with stability checks
    combined_score = fit_score * utilization_reward * capacity_weights
    priorities = np.where(valid_mask, combined_score, -np.inf)
    
    # Enhanced fit detection with progressive bonuses
    perfect_fit_mask = np.abs(remaining_space) < 1e-10
    near_perfect_mask = (remaining_space > 0) & (normalized_space < 0.01)
    
    if any(valid_mask):
        max_score = np.max(priorities[valid_mask])
        priorities[perfect_fit_mask] = 4.0 * max_score + 3.0  # Higher bonus for perfect fits
        priorities[near_perfect_mask] = 2.5 * max_score + 1.5  # Adjusted near-perfect bonus
    
    # Advanced tie-breaking with system-aware adaptive noise
    if any(valid_mask):
        score_range = np.ptp(priorities[valid_mask])
        noise_scale = 0.002 * score_range * (1 + 0.4 * capacity_variation)
        priorities[valid_mask] += np.random.normal(0, noise_scale, size=np.sum(valid_mask))
    
    # Dynamic small-item adjustment with utilization-based spreading
    if item < 0.03 * median_capacity:
        spreading_factor = 1 + 0.7 * (1 - system_utilization)**2
        priorities[valid_mask] *= spreading_factor
    
    # Fragmentation prevention for medium items
    elif item < 0.15 * median_capacity and system_utilization > 0.7:
        priorities[valid_mask] *= 1 + 0.3 * (1 - normalized_space[valid_mask])
    
    return priorities



# Function 9 - Score: -0.03883307725173573
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
    
    # System state analysis with enhanced metrics
    active_bins = bins[valid_mask]
    total_capacity = np.sum(active_bins) if any(valid_mask) else np.sum(bins)
    system_utilization = 1 - np.sum(remaining_space[valid_mask]) / total_capacity if any(valid_mask) else 0.5
    median_capacity = np.median(active_bins) if any(valid_mask) else np.median(bins)
    capacity_variation = np.std(active_bins) / median_capacity if any(valid_mask) else 1.0
    
    # Dynamic strategy parameters with enhanced adaptation
    item_ratio = item / median_capacity if median_capacity > 0 else 1.0
    fit_balance = 0.3 + 0.5 * np.tanh(3 * (system_utilization - 0.6))  # Sigmoid transition
    capacity_sensitivity = 0.5 * (1 + np.exp(-2 * capacity_variation))  # Inverse relationship
    
    # Multi-dimensional fit scoring with adaptive components
    normalized_space = remaining_space / (bins + 1e-8)
    
    # Adaptive best-fit with utilization-dependent decay
    best_fit = np.exp(-(2 + 3 * system_utilization) * normalized_space)
    
    # Fragmentation-aware worst-fit with dynamic exponent
    worst_fit = 1 - normalized_space**(1.2 + 0.8 * system_utilization)
    
    # Hybrid fit score with smooth blending
    fit_score = fit_balance * best_fit + (1 - fit_balance) * worst_fit
    
    # Multi-tier utilization reward with adaptive thresholds
    utilization = 1 - normalized_space
    utilization_reward = np.where(
        utilization > 0.97,
        4 * utilization**8,  # Critical bonus
        np.where(
            utilization > 0.85,
            2 * utilization**4,  # High utilization bonus
            np.where(
                utilization > 0.65,
                utilization**1.5,  # Medium bonus
                utilization**0.7  # Low utilization
            )
        )
    )
    
    # Capacity weighting with adaptive sensitivity and smooth normalization
    capacity_weights = (bins / median_capacity)**capacity_sensitivity
    capacity_weights = 2 / (1 + np.exp(-capacity_weights)) - 1  # Scaled sigmoid
    
    # Combine components with overflow handling
    combined_score = fit_score * utilization_reward * capacity_weights
    priorities = np.where(valid_mask, combined_score, -np.inf)
    
    # Enhanced perfect/near-perfect fit detection and bonuses
    perfect_fit_mask = np.abs(remaining_space) < 1e-8
    near_perfect_mask = (remaining_space > 0) & (normalized_space < 0.015)
    
    if any(valid_mask):
        max_score = np.max(priorities[valid_mask])
        priorities[perfect_fit_mask] = 3.0 * max_score + 2.0
        priorities[near_perfect_mask] = 2.0 * max_score + 1.0
    
    # Smart tie-breaking with system-aware adaptive noise
    if any(valid_mask):
        score_range = np.ptp(priorities[valid_mask])
        noise_scale = 0.003 * score_range * (1 + 0.3 * capacity_variation)
        priorities[valid_mask] += np.random.normal(0, noise_scale, size=np.sum(valid_mask))
    
    # Small item adjustment with dynamic spreading factor
    if item < 0.02 * median_capacity:
        spreading_factor = 1 + 0.5 * (1 - system_utilization)
        priorities[valid_mask] *= spreading_factor
    
    return priorities



# Function 10 - Score: -0.03885656087940398
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
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
    
    # System state analysis
    mean_cap = np.mean(bins)
    std_cap = np.std(bins)
    norm_item = item / mean_cap
    diversity = std_cap / (mean_cap + 1e-8)
    
    # Dynamic strategy parameters
    fit_balance = 0.65 - 0.5 * norm_item + 0.2 * np.tanh(diversity - 0.3)
    fit_balance = np.clip(fit_balance, 0.15, 0.85)
    
    # Core scoring components
    norm_space = remaining / (bins + 1e-8)
    util = 1 - norm_space
    
    # 1. Hybrid fit strategy with size adaptation
    best_fit = np.exp(-5 * norm_space)
    worst_fit = np.tanh(remaining / (0.35 * mean_cap * (1 + 0.3 * diversity)))
    fit_score = fit_balance * best_fit + (1 - fit_balance) * worst_fit
    
    # 2. Dynamic utilization rewards
    util_thresh = np.array([0.55, 0.78, 0.92]) * (1 + 0.15 * diversity)
    util_reward = np.piecewise(
        util,
        [util > util_thresh[2], util > util_thresh[1], util > util_thresh[0]],
        [
            lambda x: 4 * x**6,  # Critical
            lambda x: 2 * x**3,  # High
            lambda x: x**1.5,    # Medium
            lambda x: x**0.8     # Baseline
        ]
    )
    
    # 3. Bin size weighting with non-linearity
    size_weight = 1.3 - 0.7 * (bins / np.max(bins)) ** (0.5 + 0.5 * np.tanh(2 - 4 * norm_item))
    
    # 4. Perfect fit handling with tolerance
    priorities = np.where(valid, fit_score * util_reward * size_weight, -np.inf)
    perfect = remaining == 0
    priorities[perfect] = np.inf
    
    near_perfect = (remaining > 0) & (remaining < 0.02 * mean_cap * (1 + diversity))
    priorities[near_perfect] *= 2.5
    
    # Micro-item distribution
    if norm_item < 0.03:
        priorities[valid] *= 1 + 0.4 * (1 - norm_space[valid])
    
    # Diversity-proportional noise
    if diversity > 0.1:
        noise_mag = 0.04 * diversity * np.std(priorities[valid])
        priorities[valid] += np.random.normal(0, noise_mag, size=np.sum(valid))
    
    return priorities



