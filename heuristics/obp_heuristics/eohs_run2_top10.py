# Top 10 functions for eohs run 2

# Function 1 - Score: -0.03814451044190185
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    perfect_fit = (remaining_capacity == 0)
    tier1_fit = (remaining_capacity <= 0.02 * item) & (remaining_capacity > 0)
    tier2_fit = (remaining_capacity <= 0.07 * item) & (remaining_capacity > 0.02 * item)
    tier3_fit = (remaining_capacity <= 0.2 * item) & (remaining_capacity > 0.07 * item)
    
    # Adaptive exponential weighting with harmonic-geometric mean
    norm_capacity = np.where(valid_mask, remaining_capacity / bins, 0)
    utilization = np.where(valid_mask, 1 - norm_capacity, 0)
    weight = 1 / (1 + np.exp(-12 * (item / bins - 0.5)))
    harm_geo_mean = np.sqrt((norm_capacity * utilization) * (2 * norm_capacity * utilization) / (norm_capacity + utilization + 1e-6))
    
    # Quadratic penalty with dynamic scaling
    underutil_thresh = 0.7 + 0.2 * np.tanh(15 * (item / bins - 0.4))
    penalty = np.where(remaining_capacity > underutil_thresh * bins, -2 * (remaining_capacity / bins)**2, 0)
    
    # Multi-factor fit bonus with sigmoid transitions
    ratio = item / bins
    dynamic_factor = 0.7 * (1 + np.tanh(10 * (ratio - 0.35)))
    fit_bonus = dynamic_factor * np.exp(-8 * remaining_capacity / item) * valid_mask
    
    # Base score with utilization balancing
    base_scores = (harm_geo_mean + penalty + fit_bonus) * valid_mask
    balance_factor = 0.15 * np.abs(utilization - np.mean(utilization[valid_mask]))**1.5
    scores = base_scores - balance_factor
    
    # Dynamic tiered bonuses with smooth transitions
    scores[perfect_fit] = 6.0
    scores[tier1_fit] = 5.0 + 0.6 * np.exp(-12 * remaining_capacity[tier1_fit] / item)
    scores[tier2_fit] = 4.0 + 0.5 * np.exp(-9 * remaining_capacity[tier2_fit] / item)
    scores[tier3_fit] = 3.2 + 0.4 * np.exp(-6 * remaining_capacity[tier3_fit] / item)
    scores[~valid_mask] = -np.inf
    
    return scores



# Function 2 - Score: -0.03937659826072336
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    perfect_fit = (remaining_capacity == 0)
    
    # Dynamic entropy-based partitioning
    bin_entropy = np.log(np.var(bins) + 1) / (np.log(np.mean(bins) + 1) + 1e-8)
    resonance_factor = 0.5 * (1 + np.sin(2 * np.pi * bin_entropy))
    
    # Wavelet-inspired utilization scoring
    util = np.where(valid_mask, (bins - remaining_capacity) / bins, 0)
    wavelet_score = np.exp(-3 * np.abs(util - 0.7)**1.5) * (1 + 0.4 * np.cos(8 * np.pi * util))
    
    # Tempered Boltzmann clustering
    temp = 0.3 + 0.7 * resonance_factor
    cluster_center = np.median(bins) * (0.8 + 0.2 * np.tanh(5 * (item / np.mean(bins) - 0.6)))
    cluster_score = np.exp(-((remaining_capacity - cluster_center)**2) / (2 * temp**2)) / (temp * np.sqrt(2 * np.pi))
    
    # Multi-scale penalty function
    penalty_scale = 1.5 - 0.7 * resonance_factor
    penalty = np.where(
        remaining_capacity > 0.5 * bins,
        -penalty_scale * (remaining_capacity / bins)**(2 + 1.5 * resonance_factor),
        0
    )
    
    # Phase-dependent near-fit bonuses
    near_fit = (remaining_capacity <= 0.2 * item) & (remaining_capacity > 0)
    phase = 0.5 * (1 + np.cos(3 * np.pi * (item / np.mean(bins))))
    near_bonus = np.where(near_fit, 2.0 * np.exp(-4 * remaining_capacity / item) * (1 + 0.5 * phase), 0)
    
    # Combined scoring
    base_score = (wavelet_score + penalty) * valid_mask
    scores = base_score + 1.5 * near_bonus + 0.8 * cluster_score * (1 + 0.3 * resonance_factor)
    scores[perfect_fit] = 9.0
    scores[~valid_mask] = -np.inf
    
    return scores



# Function 3 - Score: -0.040223594612330486
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    perfect_fit = (remaining_capacity == 0)
    
    # Quantum phase detection with dynamic modulation
    bin_entropy = np.log(np.var(bins) + 1) / (np.mean(bins) + 1e-8)
    phase = 1.5 * np.tanh(3.2 * (bin_entropy - 0.4)) + 0.7
    size_ratio = item / (np.median(bins) + 1e-8)
    
    # Neural-adaptive dynamic weights
    w_cap = 0.8 * np.exp(-1.5 * phase) + 0.3 * np.log1p(size_ratio**0.6)
    w_util = 0.6 * np.exp(-1.1 * (1 - phase)) - 0.2 * size_ratio**0.3
    w_cluster = 0.5 * (1 - np.exp(-2.5 * phase)) + 0.15 * size_ratio**0.4
    
    # Multi-fractal capacity scoring
    norm_cap = np.where(valid_mask, remaining_capacity / (bins + 1e-8), 0)
    util = np.where(valid_mask, 1 - norm_cap, 0)
    fractal_score = (norm_cap**(w_cap/1.8)) * (util**(w_util*1.5)) * (1 + 0.4 * np.sin(2.5 * np.pi * util**1.7))
    
    # Hyperbolic near-fit with dynamic scaling
    near_fit = (remaining_capacity <= 0.12 * item) & (remaining_capacity > 0)
    near_bonus = np.where(near_fit, 2.0 / (1 + 1.8 * remaining_capacity/item)**1.2, 0)
    
    # Context-sensitive penalty function
    penalty_thresh = 0.35 + 0.4 * phase
    penalty = np.where(
        remaining_capacity > penalty_thresh * bins,
        -2.5 * ((remaining_capacity / (bins + 1e-8) - penalty_thresh) / (1 - penalty_thresh))**2.5,
        0
    )
    
    # Quantum-annealing cluster affinity
    cluster_size = np.maximum(0.35 * np.median(bins), item * (1 + 0.8 * phase - 0.3 * size_ratio**0.7))
    affinity = 1 - np.abs(remaining_capacity - cluster_size) / (cluster_size + 1e-6)
    cluster_score = 3.0 * np.exp(-7 * (1 - affinity)**2.2) * (1 + 0.6 * np.cos(5 * np.pi * affinity))
    
    # Multi-objective combination with phase modulation
    base_score = (fractal_score + penalty) * valid_mask
    scores = base_score + 1.5 * near_bonus + 0.9 * cluster_score * (1 + 0.6 * phase - 0.4 * size_ratio**0.5)
    scores[perfect_fit] = 10.0
    scores[~valid_mask] = -np.inf
    
    return scores



# Function 4 - Score: -0.0422393619495574
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    perfect_fit = (remaining_capacity == 0)
    
    # Dynamic cluster-aware sigmoid weighting with adaptive curvature
    cluster_center = np.median(bins) * (0.65 + 0.35 * np.tanh(6 * (item / np.mean(bins) - 0.55)))
    cluster_dist = np.abs(remaining_capacity - cluster_center) / (cluster_center + 1e-8)
    sig_curvature = 7 + 3 * np.tanh(4 * (item / np.mean(bins) - 0.5))
    sig_weight = 1 / (1 + np.exp(sig_curvature * (cluster_dist - 0.35)))
    
    # Graduated exponential fit bonus with smooth transitions
    rc_ratio = remaining_capacity / item
    fit_bonus = np.exp(-6 * rc_ratio) * (3.0 - 2.5 * np.tanh(20 * (rc_ratio - 0.02))) * \
                (1.2 - 0.8 * np.tanh(10 * (rc_ratio - 0.1)))
    
    # Phase-dependent polynomial penalty with adaptive threshold
    phase = 0.5 + 0.5 * np.tanh(12 * (item / np.mean(bins) - 0.5))
    penalty_thresh = 0.55 - 0.2 * phase
    penalty = np.where(
        remaining_capacity > penalty_thresh * bins,
        -2.5 * ((remaining_capacity / bins - penalty_thresh) / (1 - penalty_thresh))**2.5,
        0
    )
    
    # Harmonic size ratio adjustment
    ratio = item / bins
    ratio_factor = 0.9 * np.sin(np.pi * ratio)**1.5 * np.exp(-3 * (ratio - 0.45)**2)
    
    # Combined scoring with robustness
    base_score = (1.2 * sig_weight + 0.9 * ratio_factor + penalty) * valid_mask
    scores = base_score + fit_bonus
    scores[perfect_fit] = 6.0
    scores[~valid_mask] = -np.inf
    
    return scores



# Function 5 - Score: -0.06433025557185038
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    avg_bin_size = np.mean(bins)
    max_bin_size = np.max(bins)
    
    # Dynamic weight optimization with reinforcement
    ratio = item / avg_bin_size
    packing_state = 1 - np.mean(bins) / max_bin_size
    size_diversity = np.std(bins) / avg_bin_size
    temporal_decay = np.exp(-0.5 * (max_bin_size - bins) / avg_bin_size)
    
    fit_weight = 0.42 + 0.12 * np.exp(-2.0*ratio) + 0.08 * packing_state - 0.04 * size_diversity
    util_weight = 0.34 + 0.05 * ratio - 0.03 * packing_state + 0.04 * size_diversity + 0.03 * temporal_decay
    stability_weight = 0.24 - 0.02 * ratio + 0.07 * packing_state + 0.03 * size_diversity
    density_weight = 0.20 + 0.05 * ratio * (1 - packing_state) + 0.02 * temporal_decay
    
    # Enhanced harmonic fit with adaptive clustering
    multiples = np.floor_divide(remaining_capacity, item)
    fractional = remaining_capacity / item
    dist_to_multiple = np.minimum(remaining_capacity % item, item - (remaining_capacity % item))
    exact_fit = (remaining_capacity == 0) * 4.0
    near_fit = 1 / (1 + dist_to_multiple**2.8)
    fractional_bonus = 1 / (1 + 5*(fractional - np.round(fractional))**2)
    harmonic_fit = np.where(exact_fit > 0, exact_fit,
                           (near_fit + 0.6 * fractional_bonus) * (1 + 1.2 / (multiples + 1 + 0.15*ratio)))
    
    # Utilization with temporal-aware forecasting
    current_util = 1 - (remaining_capacity / bins)
    projected_util = 1 - (remaining_capacity / (bins + 0.7*item*temporal_decay))
    future_util = 1 - np.maximum(0, remaining_capacity - 0.9*item) / bins
    util_balance = 0.48 + 0.22 * (bins / avg_bin_size) - 0.10 * packing_state
    utilization_score = (0.42 * current_util + 0.38 * projected_util + 0.20 * future_util) * util_balance
    
    # Adaptive quantile-based stability scoring
    residuals = remaining_capacity % item
    q1, q3 = np.percentile(residuals[valid_mask], [20, 80])
    iqr = q3 - q1
    stability_score = np.exp(-0.7 * np.abs(residuals - np.median(residuals)) / (iqr + 1e-6))
    stability_score = np.where((residuals < 0.05*item) | (residuals > 0.95*item),
                              stability_score * 1.4, stability_score * (1 + 0.12*temporal_decay))
    
    # Cluster-aware multi-scale density analysis
    perfect_fit_thresh = 0.03 * bins + 0.06 * avg_bin_size * (1 - packing_state)
    cluster_size = np.maximum(0.25 * avg_bin_size * temporal_decay, 0.8 * item)
    cluster_density = 1 - np.abs(remaining_capacity - cluster_size) / (cluster_size + 0.08*item)
    global_density = 1 - remaining_capacity / (avg_bin_size + item*(0.95 - 0.12*packing_state))
    density_score = np.where(
        remaining_capacity <= perfect_fit_thresh,
        2.4 + (perfect_fit_thresh - remaining_capacity)/perfect_fit_thresh + 0.6*global_density,
        1.6 + 0.45*cluster_density + 0.35*global_density - 0.08*(1 - temporal_decay)
    )
    
    # Context-weighted combined score with reinforcement
    combined_score = (
        fit_weight * harmonic_fit +
        util_weight * utilization_score +
        stability_weight * stability_score +
        density_weight * density_score
    ) / (fit_weight + util_weight + stability_weight + density_weight)
    
    combined_score[~valid_mask] = -np.inf
    return combined_score



# Function 6 - Score: -0.06467615452464776
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    avg_bin_size = np.mean(bins)
    max_bin_size = np.max(bins)
    
    # Dynamic weight optimization with reinforcement feedback
    ratio = item / avg_bin_size
    packing_state = 1 - np.mean(bins) / max_bin_size
    size_diversity = np.std(bins) / avg_bin_size
    temporal_decay = np.exp(-0.5 * (max_bin_size - bins) / avg_bin_size)
    
    fit_weight = 0.38 + 0.15 * np.exp(-1.8*ratio) + 0.07 * packing_state - 0.05 * size_diversity
    util_weight = 0.32 + 0.06 * ratio - 0.04 * packing_state + 0.03 * size_diversity + 0.02 * temporal_decay
    stability_weight = 0.22 - 0.03 * ratio + 0.06 * packing_state + 0.02 * size_diversity
    density_weight = 0.18 + 0.04 * ratio * (1 - packing_state) + 0.01 * temporal_decay
    
    # Enhanced harmonic fit with adaptive bonuses
    multiples = np.floor_divide(remaining_capacity, item)
    fractional = remaining_capacity / item
    dist_to_multiple = np.minimum(remaining_capacity % item, item - (remaining_capacity % item))
    exact_fit = (remaining_capacity == 0) * 3.5
    near_fit = 1 / (1 + dist_to_multiple**2.5)
    fractional_bonus = 1 / (1 + 4*(fractional - np.round(fractional))**2)
    harmonic_fit = np.where(exact_fit > 0, exact_fit,
                           (near_fit + 0.5 * fractional_bonus) * (1 + 1.0 / (multiples + 1 + 0.2*ratio)))
    
    # Utilization with temporal-decay lookahead
    current_util = 1 - (remaining_capacity / bins)
    projected_util = 1 - (remaining_capacity / (bins + 0.6*item*temporal_decay))
    future_util = 1 - np.maximum(0, remaining_capacity - 0.8*item) / bins
    util_balance = 0.45 + 0.25 * (bins / avg_bin_size) - 0.12 * packing_state
    utilization_score = (0.4 * current_util + 0.35 * projected_util + 0.25 * future_util) * util_balance
    
    # Quantile-based stability scoring
    residuals = remaining_capacity % item
    q1, q3 = np.percentile(residuals[valid_mask], [25, 75])
    iqr = q3 - q1
    stability_score = np.exp(-0.5 * np.abs(residuals - np.median(residuals)) / (iqr + 1e-6))
    stability_score = np.where((residuals < 0.08*item) | (residuals > 0.92*item),
                              stability_score * 1.3, stability_score * (1 + 0.1*temporal_decay))
    
    # Adaptive multi-scale density analysis
    perfect_fit_thresh = 0.04 * bins + 0.05 * avg_bin_size * (1 - packing_state)
    cluster_size = np.maximum(0.2 * avg_bin_size * temporal_decay, 0.7 * item)
    cluster_density = 1 - np.abs(remaining_capacity - cluster_size) / (cluster_size + 0.1*item)
    global_density = 1 - remaining_capacity / (avg_bin_size + item*(0.9 - 0.1*packing_state))
    density_score = np.where(
        remaining_capacity <= perfect_fit_thresh,
        2.2 + (perfect_fit_thresh - remaining_capacity)/perfect_fit_thresh + 0.5*global_density,
        1.4 + 0.4*cluster_density + 0.3*global_density - 0.1*(1 - temporal_decay)
    )
    
    # Context-weighted combined score with reinforcement
    combined_score = (
        fit_weight * harmonic_fit +
        util_weight * utilization_score +
        stability_weight * stability_score +
        density_weight * density_score
    ) / (fit_weight + util_weight + stability_weight + density_weight)
    
    combined_score[~valid_mask] = -np.inf
    return combined_score



# Function 7 - Score: -0.0657617298403494
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    perfect_fit = (remaining_capacity == 0)
    
    # Adaptive capacity utilization with sigmoid smoothing
    capacity_ratio = remaining_capacity / (bins + 1e-8)
    sigmoid_ratio = 1 / (1 + np.exp(-8 * (capacity_ratio - 0.5)))
    adaptive_weight = 0.5 * sigmoid_ratio + 0.3 * (1 - capacity_ratio) + 0.2 * (item / (bins + 1e-8))
    
    # Multi-criteria scoring components
    utilization_score = np.power(1 - capacity_ratio, 0.6)
    future_potential = np.where(remaining_capacity >= item * 0.5,
                              np.power(item / (remaining_capacity + 1e-8), 0.25) * (1 + 0.2 * np.log1p(bins/(item + 1e-8))),
                              0)
    efficiency_score = np.tanh(np.floor(bins / (item + 1e-8)) * item / (bins + 1e-8))
    
    # Progressive penalty system
    penalty = np.where(remaining_capacity < item * 2.0,
                      np.power(remaining_capacity / (item + 1e-8), 2.0),
                      1.0 - 0.15 * np.power(capacity_ratio, 1.5))
    
    # Adaptive weighted harmonic mean integration
    combined_score = 1 / (0.4 / (utilization_score + 1e-8) + 
                         0.35 / (future_potential + 1e-8) + 
                         0.25 / (efficiency_score + 1e-8))
    scores = (adaptive_weight * combined_score + 
             (1 - adaptive_weight) * (0.55 * utilization_score + 0.45 * future_potential)) * valid_mask * penalty
    scores[perfect_fit] = 3.0
    scores[~valid_mask] = -np.inf
    return scores



# Function 8 - Score: -0.07301921006006429
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    perfect_fit = (remaining_capacity == 0)
    
    # Adaptive capacity utilization with exponential smoothing
    capacity_ratio = remaining_capacity / (bins + 1e-8)
    exp_ratio = np.exp(-4 * np.abs(capacity_ratio - 0.3))
    adaptive_weight = 0.6 * exp_ratio + 0.2 * (1 - capacity_ratio) + 0.2 * np.sqrt(item / (bins + 1e-8))
    
    # Enhanced multi-criteria scoring
    utilization_score = np.power(1 - capacity_ratio, 0.7) * (1 + 0.1 * np.log1p(bins))
    future_potential = np.where(remaining_capacity >= item * 0.3,
                              np.power(item / (remaining_capacity + 1e-8), 0.2) * (1 + 0.3 * np.log1p(bins/(2*item + 1e-8))),
                              0)
    efficiency_score = np.tanh(2 * np.floor(bins / (item + 1e-8)) * item / (bins + 1e-8)) * (1 + 0.1 * np.log1p(bins))
    
    # Progressive penalty-reward system
    penalty_reward = np.where(remaining_capacity < item * 1.5,
                             np.power(remaining_capacity / (item + 1e-8), 1.8),
                             1.0 + 0.2 * np.power(1 - capacity_ratio, 2.0))
    
    # Context-aware weighted geometric mean integration
    combined_score = np.power(utilization_score, 0.45) * np.power(future_potential + 1e-8, 0.35) * np.power(efficiency_score + 1e-8, 0.2)
    scores = (adaptive_weight * combined_score + 
             (1 - adaptive_weight) * (0.6 * utilization_score + 0.4 * future_potential)) * valid_mask * penalty_reward
    scores[perfect_fit] = 4.0
    scores[~valid_mask] = -np.inf
    return scores



# Function 9 - Score: -0.09460921997424315
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    perfect_fit = (remaining_capacity == 0)
    
    # Adaptive dynamic weights with stability factor
    capacity_ratio = remaining_capacity / bins
    dynamic_weight = 0.4 + 0.4 * (1 - np.power(capacity_ratio, 0.7))
    
    # Enhanced scoring components with density and proximity factors
    multiple_score = np.floor(bins / (item + 1e-6)) * item / bins
    remaining_score = np.power(1 - (remaining_capacity / bins), 0.5)
    lookahead_score = np.where(remaining_capacity >= item * 0.6, 
                              np.power(item / (remaining_capacity + 1e-6), 0.3), 0)
    density_score = np.power(item / (bins + 1e-6), 0.2)
    proximity_score = np.power(1 - np.abs(bins - item) / (bins + 1e-6), 1.5)
    
    # Non-linear decay with stability adjustment
    decay_factor = np.where(remaining_capacity < item, 
                           np.power(remaining_capacity / (item + 1e-6), 1.7), 
                           1.0 - 0.15 * np.power(capacity_ratio, 1.5))
    
    # Combined score with stability balance and density optimization
    scores = (dynamic_weight * (0.5 * multiple_score + 0.2 * lookahead_score + 0.2 * remaining_score + 0.1 * proximity_score) + 
             (1 - dynamic_weight) * (0.6 * remaining_score + 0.2 * lookahead_score + 0.1 * density_score + 0.1 * proximity_score)) * valid_mask * decay_factor
    scores[perfect_fit] = 2.0
    scores[~valid_mask] = -np.inf
    return scores



# Function 10 - Score: -3.145462372656542
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    perfect_fit = (remaining_capacity == 0)
    
    # Dynamic entropy modeling with adaptive scaling
    global_state = np.mean(bins) / (np.max(bins) + 1e-6)
    entropy_factor = 1.4 - 1.1 / (1 + np.exp(-15 * (global_state - 0.4)))
    
    # Hybrid scoring with adaptive weights
    linear_score = 1 - remaining_capacity / (bins + 1e-6)
    exp_decay = np.exp(-3.0 * remaining_capacity / (item + 1e-6))
    hybrid_score = (0.6 * linear_score + 0.4 * exp_decay) * entropy_factor
    
    # Localized bin state awareness with dynamic thresholds
    neighborhood = np.abs(bins - np.median(bins)) / (np.std(bins) + 1e-6)
    local_weight = 0.7 * np.exp(-0.5 * neighborhood**2.0)
    
    # Multi-phase optimization with reinforcement feedback
    phase = 0.7 * np.tanh(8 * (global_state - 0.5)) + 0.5
    capacity_ratio = remaining_capacity / (bins + 1e-6)
    phase_score = np.where(
        capacity_ratio < 0.2,
        1.5 * (1 - capacity_ratio / 0.2)**(phase + 0.2),
        0.8 * (capacity_ratio / 0.8)**(1.2 - phase)
    )
    
    # Dynamic ensemble scoring with feedback adjustment
    scores = (0.4 * hybrid_score + 0.3 * local_weight + 0.3 * phase_score) * valid_mask
    scores[perfect_fit] = 4.0
    scores[~valid_mask] = -np.inf
    
    return scores



