# Top 10 functions for eohs run 1

# Function 1 - Score: -0.03789957767317467
{The enhanced algorithm integrates dynamic quartile-based target optimization with sigmoid-weighted capacity adjustment, multi-tiered exponential attraction-repulsion balancing with adaptive density-aware dispersion penalties, and a novel harmonic resonance modulation for optimal load distribution across bins.}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    capacity_ratio = np.percentile(bins, 80) / np.percentile(bins, 20)
    dynamic_target = 0.55 + 0.25 * (1 / (1 + np.exp(-3.0 * (capacity_ratio - 1.1))))
    
    remaining = bins - item
    valid = remaining >= 0
    filled = (bins - remaining) / bins
    
    # Enhanced multi-tier capacity scoring
    tier_weights = 0.85 + 0.5 * np.tanh(2.5 * (bins / np.percentile(bins, 65) - 1.2))
    optimal_zone = dynamic_target * (1 + 0.18 * np.sin(1.5 * np.pi * tier_weights))
    
    # Dual-phase attraction-repulsion
    attraction = np.exp(-5 * np.abs(filled - optimal_zone))
    repulsion = 1 - 0.35 * np.power(np.maximum(0, filled - (optimal_zone + 0.15)), 2.0)
    reward = 3.5 * attraction * repulsion * tier_weights
    
    # Density-adaptive dispersion
    density_gradient = np.abs(filled - np.median(filled[valid])) if np.any(valid) else 0
    dispersion = 1.4 - 0.4 * np.tanh(6 * density_gradient) * (1 - 0.25 * np.cos(3 * np.pi * filled))
    
    # Resonant harmonic modulation
    harmonic_factor = 1 + 0.2 * np.sin(3 * np.pi * (filled - dynamic_target + 0.05))
    priority_scores = np.where(valid,
                             -np.log1p(remaining) * reward * dispersion * harmonic_factor,
                             np.inf)
    return priority_scores



# Function 2 - Score: -0.038007547103671865
{The new algorithm enhances dynamic target adaptation with adaptive sigmoid scaling, introduces bin-size-aware logarithmic capacity utilization with exponential decay, implements a triple-phase stability mechanism combining local, global, and historical fill patterns, and incorporates a dynamic fragmentation penalty based on normalized remaining capacity, skewness, and kurtosis.}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    normalized_remaining = np.where(bins > 0, remaining / bins, 0)
    current_fill = 1 - normalized_remaining
    
    # Enhanced dynamic target adaptation
    mean_bins = np.mean(bins)
    std_bins = np.std(bins) + 1e-9
    dynamic_target = 0.65 + 0.15 * (1 / (1 + np.exp(-mean_bins / (std_bins + 1e-9)))) + 0.02 * np.tanh(mean_bins)
    
    # Adaptive sigmoid scaling
    fill_deviation = current_fill - dynamic_target
    adaptive_scale = 7.0 + 5.0 * np.tanh(np.var(bins) / (mean_bins**2 + 1e-9))
    reward = 2.0 / (1 + np.exp(-fill_deviation * adaptive_scale)) + 1.0
    
    # Bin-size-aware logarithmic capacity utilization
    size_factor = np.log1p(bins / (mean_bins + 1e-9)) ** 0.35
    capacity_term = (1 - np.exp(-3.5 * normalized_remaining)) * (0.85 + 0.15 * size_factor)
    
    # Triple-phase stability mechanism
    global_fill = np.mean(current_fill[valid]) if np.any(valid) else dynamic_target
    historical_fill = np.mean(current_fill) if len(bins) > 1 else dynamic_target
    local_stability = 0.90 + 0.10 * np.exp(-15 * (current_fill - dynamic_target)**2)
    global_stability = 0.92 + 0.08 * np.exp(-20 * (current_fill - global_fill)**2)
    historical_stability = 0.94 + 0.06 * np.exp(-25 * (current_fill - historical_fill)**2)
    
    # Dynamic fragmentation penalty
    skewness = np.mean((bins - mean_bins)**3) / (std_bins**3 + 1e-9)
    kurtosis = np.mean((bins - mean_bins)**4) / (std_bins**4 + 1e-9) - 3
    fragmentation = (np.abs(normalized_remaining - 0.5) * 0.3 + 
                    np.abs(skewness) * 0.4 + 
                    np.abs(kurtosis) * 0.3) if np.any(valid) else 0
    
    priority_scores = np.where(valid, -capacity_term * reward * local_stability * global_stability * historical_stability / (1 + fragmentation), np.inf)
    return priority_scores



# Function 3 - Score: -0.03810322304293837
{The enhanced algorithm combines dynamic target optimization using quartile-based sigmoid modulation, multi-tiered capacity scoring with adaptive exponential attraction-repulsion, density-aware dispersion penalties with nonlinear smoothing, and phase-adjusted harmonic priority modulation for optimal load balancing.}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    capacity_ratio = np.percentile(bins, 80) / np.median(bins)
    dynamic_target = 0.65 + 0.15 * (1 / (1 + np.exp(-3 * (capacity_ratio - 1.1))))
    
    remaining = bins - item
    valid = remaining >= 0
    filled = (bins - remaining) / bins
    
    # Enhanced multi-tier capacity scoring
    tier_weights = 0.85 + 0.3 * np.tanh(2.5 * (bins / np.percentile(bins, 65) - 1))
    optimal_zone = dynamic_target * (1 + 0.12 * np.sin(1.5 * np.pi * tier_weights))
    
    # Adaptive attraction-repulsion
    attraction = np.exp(-5 * np.abs(filled - optimal_zone))
    repulsion = 1 - 0.25 * np.power(np.maximum(0, filled - (optimal_zone + 0.1)), 2)
    reward = 3.2 * attraction * repulsion * tier_weights
    
    # Nonlinear dispersion adjustment
    density_gradient = np.abs(filled - np.mean(filled[valid])) if np.any(valid) else 0
    dispersion = 1.25 - 0.4 * np.tanh(6 * density_gradient) * (1 - 0.15 * np.cos(3 * np.pi * filled))
    
    # Phase-adjusted harmonic modulation
    harmonic_factor = 1 + 0.12 * np.sin(3 * np.pi * (filled - dynamic_target + 0.05))
    priority_scores = np.where(valid,
                             -np.log1p(remaining) * reward * dispersion * harmonic_factor,
                             np.inf)
    return priority_scores



# Function 4 - Score: -0.038617738452392913
{The new algorithm employs a multi-objective optimization approach combining dynamic fill-level bands with bin-size-aware quadratic programming, incorporating adaptive reward zones based on hyperbolic tangent functions and a novel entropy-based dispersion penalty to minimize both overfill risk and fragmentation while maintaining load balance.}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    target_ratio = 0.72
    remaining = bins - item
    valid = remaining >= 0
    normalized_remaining = remaining / bins
    current_fill = 1 - normalized_remaining
    
    # Dynamic fill bands with size adaptation
    size_factor = np.sqrt(bins / np.mean(bins))
    lower_band = target_ratio * (0.85 + 0.1 * np.tanh(2 * (size_factor - 1)))
    upper_band = target_ratio * (1.15 - 0.1 * np.tanh(2 * (size_factor - 1)))
    
    # Quadratic programming terms
    fill_deviation = current_fill - target_ratio
    quadratic_term = 1 - 0.5 * fill_deviation ** 2 / (0.08 + 0.02 * size_factor)
    
    # Adaptive reward zones
    zone_weight = 0.5 * (np.tanh(15 * (current_fill - lower_band)) - np.tanh(15 * (current_fill - upper_band)))
    reward = 2.0 + 1.5 * zone_weight * quadratic_term
    
    # Entropy-based dispersion penalty
    global_fill = np.mean(current_fill[valid]) if np.any(valid) else target_ratio
    entropy_penalty = 1.2 - 0.2 * np.exp(-8 * (current_fill - global_fill) ** 2)
    
    priority_scores = np.where(valid, -np.log1p(remaining) * reward * entropy_penalty * size_factor, np.inf)
    return priority_scores



# Function 5 - Score: -0.03970831900590262
{The new algorithm integrates probabilistic bin selection with adaptive weight balancing, leveraging Gaussian mixture modeling for fill pattern analysis, incorporates a dynamic volatility index based on bin capacity derivatives, and applies a multi-criteria decision framework combining harmonic mean prioritization with exponential smoothing of historical allocation patterns.}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    normalized_remaining = np.where(bins > 0, remaining / bins, 0)
    current_fill = 1 - normalized_remaining
    
    # Gaussian mixture modeling
    mean_fill = np.mean(current_fill[valid]) if np.any(valid) else 0.68
    std_fill = np.std(current_fill[valid]) + 1e-9 if np.any(valid) else 0.12
    gmm_weight = 0.8 * np.exp(-0.5 * ((current_fill - mean_fill) / (std_fill + 1e-9))**2) + 0.2
    
    # Dynamic volatility index
    capacity_derivative = np.abs(np.gradient(bins)) / (np.mean(bins) + 1e-9)
    volatility = 1.1 - 0.3 * np.tanh(5 * capacity_derivative)
    
    # Harmonic mean prioritization
    harmonic_term = 2 / (1 / (current_fill + 1e-9) + 1 / (normalized_remaining + 1e-9))
    
    # Exponential smoothing
    historical_factor = 0.7 + 0.3 * np.exp(-0.5 * np.abs(current_fill - 0.65))
    
    priority_scores = np.where(valid, 
                             -harmonic_term * gmm_weight * volatility * historical_factor * np.log1p(bins),
                             np.inf)
    return priority_scores



# Function 6 - Score: -0.04571036644755802
{The new algorithm combines dynamic target adaptation with sigmoid-based reward shaping, incorporates bin-size-aware logarithmic capacity terms, and applies a dual-phase dispersion penalty using both variance and entropy metrics to optimize load balance and minimize fragmentation.}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    normalized_remaining = np.where(bins > 0, remaining / bins, 0)
    current_fill = 1 - normalized_remaining
    
    dynamic_target = 0.65 + 0.1 * (1 - np.exp(-np.mean(bins) / (np.std(bins) + 1e-9)))
    fill_deviation = current_fill - dynamic_target
    reward = 2.0 / (1 + np.exp(-fill_deviation * 12.0)) + 0.8
    
    capacity_term = np.log1p(normalized_remaining * 2.0) ** 0.7
    size_adjustment = 0.7 + np.tanh(bins / (np.max(bins) + 1e-9)) * 0.3
    
    variance_penalty = 1 + 0.5 * np.var(bins) / (np.mean(bins)**2 + 1e-9)
    entropy = -np.sum(current_fill * np.log(current_fill + 1e-9)) if np.any(valid) else 0
    entropy_penalty = 1.1 - 0.1 * np.exp(-5 * (entropy - np.log(len(bins)))**2)
    
    priority_scores = np.where(valid, -capacity_term * reward * size_adjustment / (variance_penalty * entropy_penalty), np.inf)
    return priority_scores



# Function 7 - Score: -0.06307010544913333
{The new algorithm combines dynamic bin clustering with size-aware optimal fill targets, adaptive reward shaping based on bin utilization patterns, non-linear capacity penalties with bin diversity factors, and probabilistic load balancing for improved packing efficiency.}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    new_fill = (bins - remaining) / bins
    
    # Dynamic bin clustering with size-aware optimal fill
    q1, q3 = np.percentile(bins, [25, 75])
    cluster_factor = np.where(bins < q1, 0.4, np.where(bins > q3, 0.6, 0.5))
    optimal_fill = 0.45 + 0.3 * np.tanh((bins - np.mean(bins)) / (0.2 + np.std(bins))) * cluster_factor
    
    # Adaptive reward shaping with utilization patterns
    fill_deviation = np.abs(new_fill - optimal_fill)
    reward = np.exp(-12 * fill_deviation**1.2) * (1 + 0.15 * np.cos(bins / (0.5 * np.max(bins))))
    
    # Non-linear capacity penalty with diversity
    mean_cap = np.mean(bins)
    capacity_penalty = np.where(remaining > 0,
                              np.exp(-0.6 * (remaining / (1 + mean_cap))**0.9) * (1 - 0.1 * np.tanh(remaining / mean_cap)),
                              np.inf)
    
    # Probabilistic load balancing factor
    load_balance = 0.8 + 0.2 * np.random.rand(len(bins)) * (bins / np.max(bins))**0.5
    
    priority_scores = np.where(valid,
                              reward * capacity_penalty * load_balance,
                              np.inf)
    return priority_scores



# Function 8 - Score: -0.06372953055938838
{The new algorithm combines dynamic optimal fill targeting based on bin size distribution, exponential reward-penalty balance for fill ratios, adaptive bin size normalization, and a fragmentation-aware capacity penalty to optimize packing efficiency and load balancing.}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    new_fill = (bins - remaining) / bins
    
    # Dynamic optimal fill based on bin size distribution
    median_size = np.median(bins)
    optimal_fill = 0.5 + 0.2 * np.tanh((bins - median_size) / (0.1 + np.std(bins)))
    
    # Exponential reward-penalty balance
    reward = np.exp(-8 * np.abs(new_fill - optimal_fill))
    
    # Adaptive bin size normalization
    size_factor = 0.7 + 0.3 * (bins / np.max(bins))**0.5
    
    # Fragmentation-aware capacity penalty
    capacity_penalty = np.where(remaining > 0, 
                               np.exp(-0.5 * remaining / (1 + np.mean(bins))), 
                               np.inf)
    
    priority_scores = np.where(valid, 
                              reward * size_factor * capacity_penalty, 
                              np.inf)
    return priority_scores



# Function 9 - Score: -0.1277074803025955
{The novel algorithm utilizes a fractal-inspired adaptive weighting system with dynamic capacity tiers, employing logistic growth models for fill-level optimization, a cosine-modulated dispersion penalty based on bin size quartiles, and a dual-phase reward mechanism combining exponential attraction to ideal fill zones with quadratic repulsion from boundaries.}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    target_fill = 0.65 + 0.1 * (1 / (1 + np.exp(-3 * (np.median(bins)/np.mean(bins) - 0.8))))
    remaining = bins - item
    valid = remaining >= 0
    relative_capacity = (bins - remaining) / bins
    
    # Fractal capacity tiers
    tier_factor = np.log1p(np.abs(bins - np.percentile(bins, 75))) / np.log(2.5)
    dynamic_range = 0.15 * (1 + 0.3 * np.cos(np.pi * tier_factor))
    optimal_zone = target_fill * (1 + 0.2 * np.tanh(3 * (tier_factor - 1)))
    
    # Dual-phase reward
    attraction = np.exp(-5 * np.abs(relative_capacity - optimal_zone))
    repulsion = 1 - 0.4 * np.power(np.maximum(0, relative_capacity - (optimal_zone + dynamic_range)), 2)
    reward = 2.8 * attraction * repulsion
    
    # Cosine-modulated dispersion
    global_density = np.mean(relative_capacity[valid]) if np.any(valid) else optimal_zone
    dispersion_penalty = 1.25 - 0.25 * np.cos(4 * np.pi * (relative_capacity - global_density)) * np.exp(-2 * tier_factor)
    
    priority_scores = np.where(valid,
                             -np.log1p(remaining) * reward * dispersion_penalty * (1 + 0.1 * tier_factor),
                             np.inf)
    return priority_scores



# Function 10 - Score: -3.1442426474091714
{The new algorithm employs a hybrid approach combining exponential fill ratio balancing with dynamic bin affinity scoring, utilizing a piecewise-linear reward function that transitions between aggressive and conservative packing modes based on bin capacity quartiles, while incorporating item-to-bin size ratio awareness through a power-law weighting system.}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    target_min = 0.55
    target_max = 0.8
    remaining = bins - item
    valid = remaining >= 0
    fill_ratio = 1 - (remaining / bins)
    
    # Quartile-based mode switching
    q1, q3 = np.percentile(bins, [25, 75]) if len(bins) > 1 else (0, 0)
    mode_factor = np.where(bins < q1, 0.7, np.where(bins > q3, 1.3, 1.0))
    
    # Piecewise-linear reward function
    reward = np.where(fill_ratio < target_min, 
                     (fill_ratio / target_min) * 0.8,
                     np.where(fill_ratio > target_max,
                            1.2 - (fill_ratio - target_max) / (1 - target_max),
                            0.8 + 0.4 * (fill_ratio - target_min) / (target_max - target_min)))
    
    # Size ratio power-law weighting
    size_ratio = item / bins
    size_weight = np.power(size_ratio, -0.2) * (1 + np.log1p(fill_ratio))
    
    # Bin affinity scoring
    bin_affinity = 1 / (1 + np.abs(bins - np.median(bins)) / np.std(bins)) if np.std(bins) > 0 else 1.0
    
    priority_scores = np.where(valid,
                             (reward * mode_factor * size_weight * bin_affinity) / (remaining + 1e-6),
                             np.inf)
    return priority_scores



