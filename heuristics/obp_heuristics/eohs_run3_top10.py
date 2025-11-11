# Top 10 functions for eohs run 3

# Function 1 - Score: -0.037181313515069446
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    closest_multiple = np.floor(remaining / (item + 1e-9) + 0.67) * item
    dist = np.abs(remaining - closest_multiple)
    dynamic_weight = 1.4 / (1 + np.exp(-3.5 * dist / (item + 1e-9)))
    exact_fit_bonus = (remaining == 0) * 6.0
    capacity_ratio = remaining / (item + 1e-9)
    density_factor = 0.9 / (1 + np.log1p(remaining * 1.8))
    fragmentation_penalty = 0.25 * (1 - np.exp(-capacity_ratio**1.8))
    adaptive_penalty = 0.7 * np.exp(-capacity_ratio * 0.6) - 0.3 * np.exp(capacity_ratio * 1.5) + 0.1 * capacity_ratio**0.8
    stability_term = 0.15 * np.tanh(2.5 - 0.5 * capacity_ratio)
    priority_scores = np.where(
        valid_mask,
        -dynamic_weight * dist + exact_fit_bonus - fragmentation_penalty + density_factor + adaptive_penalty + stability_term,
        -np.inf
    )
    return priority_scores



# Function 2 - Score: -0.03754491532646636
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    closest_multiple = np.floor(remaining / (item + 1e-9) + 0.7) * item
    dist = np.abs(remaining - closest_multiple)
    
    dynamic_weight = 1.5 / (1 + np.exp(-3.8 * dist / (item + 1e-9)))
    exact_fit_bonus = (remaining == 0) * 6.5
    capacity_ratio = remaining / (item + 1e-9)
    
    density_factor = 1.0 / (1 + np.log1p(remaining * 1.9))
    fragmentation_penalty = 0.28 * (1 - np.exp(-capacity_ratio**1.9))
    
    adaptive_penalty = (0.75 * np.exp(-capacity_ratio * 0.65) - 
                        0.35 * np.exp(capacity_ratio * 1.6) + 
                        0.12 * capacity_ratio**0.85 +
                        0.05 * capacity_ratio)
    
    stability_term = 0.18 * np.tanh(2.8 - 0.55 * capacity_ratio)
    future_potential = 0.22 * np.exp(-0.45 * (capacity_ratio - 1.6)**2) * (1 - 0.1 * dist)
    
    priority_scores = np.where(
        valid_mask,
        (-dynamic_weight * dist + exact_fit_bonus - fragmentation_penalty + 
         density_factor + adaptive_penalty + stability_term + future_potential),
        -np.inf
    )
    return priority_scores



# Function 3 - Score: -0.04872972649868597
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    epsilon = 1e-9
    
    # Dynamic proximity scoring
    closest_multiple = np.floor(remaining / (item + epsilon) + 0.67) * item
    dist = np.abs(remaining - closest_multiple)
    dynamic_weight = 1.6 / (1 + np.exp(-4.0 * dist / (item + epsilon)))
    
    # Enhanced exact-fit reward
    exact_fit_bonus = (remaining == 0) * 7.5
    
    # Quad-phase capacity penalty
    capacity_ratio = remaining / (item + epsilon)
    exp_penalty = 0.8 * np.exp(-capacity_ratio * 0.7)
    log_penalty = 0.4 * np.log1p(capacity_ratio * 2.2)
    poly_penalty = 0.15 * capacity_ratio**1.2
    trig_penalty = 0.1 * np.sin(capacity_ratio * np.pi * 1.5)
    
    # Adaptive density factor
    density_factor = 1.1 / (1 + np.log1p(remaining * 2.0)) * (1 + 0.3 * np.tanh(2.0 - capacity_ratio))
    
    # Fragmentation-aware adjustment
    fragmentation_penalty = 0.3 * (1 - np.exp(-capacity_ratio**2.0)) * (1 + 0.2 * np.cos(capacity_ratio * np.pi))
    
    # Stability term with future potential
    stability_term = 0.2 * np.tanh(3.0 - 0.6 * capacity_ratio) + 0.05 * np.sin(capacity_ratio * np.pi * 0.8)
    
    priority_scores = np.where(
        valid_mask,
        -dynamic_weight * dist + exact_fit_bonus - exp_penalty - log_penalty - poly_penalty - trig_penalty 
        - fragmentation_penalty + density_factor + stability_term,
        -np.inf
    )
    return priority_scores



# Function 4 - Score: -0.05495266771770675
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    closest_multiple = np.round(remaining / item) * item
    proximity_score = -np.abs(remaining - closest_multiple) / item
    capacity_penalty = -np.log1p(remaining)
    near_exact_bonus = 2.0 / (1 + np.exp(5 * (remaining - 0.5 * item)))
    fill_emphasis = 1.5 / (1 + np.exp(-2 * (bins - remaining - 0.8 * bins)))
    priority_scores = np.where(
        valid_mask,
        proximity_score + 0.4 * capacity_penalty + near_exact_bonus + fill_emphasis,
        -np.inf
    )
    return priority_scores



# Function 5 - Score: -0.056306624118229544
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    closest_multiple = np.floor(remaining / (item + 1e-9) + 0.68) * item
    dist = np.abs(remaining - closest_multiple)
    
    dynamic_weight = 1.45 / (1 + np.exp(-3.6 * dist / (item + 1e-9)))
    exact_fit_bonus = (remaining == 0) * 6.2 + (remaining > 0) * 0.5 / (1 + np.abs(remaining - item))
    
    capacity_ratio = remaining / (item + 1e-9)
    density_factor = 0.95 / (1 + np.log1p(remaining * 1.85))
    fragmentation_penalty = 0.26 * (1 - np.exp(-capacity_ratio**1.85))
    
    adaptive_penalty = (0.72 * np.exp(-capacity_ratio * 0.62) + 
                        0.15 * capacity_ratio**0.82)
    
    stability_term = 0.16 * np.tanh(2.6 - 0.52 * capacity_ratio)
    packing_potential = 0.2 * np.exp(-0.5 * (capacity_ratio - 1.7)**2) * (1 - 0.08 * dist)
    
    priority_scores = np.where(
        valid_mask,
        (-dynamic_weight * dist + exact_fit_bonus - fragmentation_penalty + 
         density_factor - adaptive_penalty + stability_term + packing_potential),
        -np.inf
    )
    return priority_scores



# Function 6 - Score: -0.05708817173925359
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    utilization = (bins - remaining) / (bins + 1e-9)
    
    # Dual-phase sigmoid penalty with dynamic curvature
    fill_ratio = utilization / (np.mean(utilization) + 1e-9)
    curvature = 2.0 + 3.0 * np.tanh(2.5 * (fill_ratio - 1.2))
    penalty = 1.0 / (1.0 + np.exp(-curvature * (utilization - 0.6)))
    
    # Adaptive weight based on system entropy
    entropy = -np.sum(utilization * np.log(utilization + 1e-9))
    weight = 0.5 + 0.5 * np.tanh(3.0 * (entropy - 0.8))
    
    priority_scores = np.where(
        valid_mask,
        np.exp(-0.5 * remaining) - weight * penalty,
        -np.inf
    )
    return priority_scores



# Function 7 - Score: -0.059962675635456444
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    harmonic_multiple = item / (1 + np.abs(remaining % item - item/2))
    dynamic_weight = 1 / (1 + np.exp(-2.5 * harmonic_multiple / item))
    exact_fit_bonus = (remaining == 0) * 5.0 * (1 + 0.2 * np.log1p(bins))
    capacity_ratio = remaining / bins
    density_factor = 2 / (1 + np.exp(3 * capacity_ratio)) - 0.5
    fragmentation_penalty = 0.2 * (1 - 1/(1 + np.exp(-5*(capacity_ratio-0.7))))
    priority_scores = np.where(
        valid_mask,
        -dynamic_weight * harmonic_multiple 
        - 0.6 * np.exp(capacity_ratio * 1.2) 
        + exact_fit_bonus 
        - fragmentation_penalty 
        + density_factor,
        -np.inf
    )
    return priority_scores



# Function 8 - Score: -0.0961335901577915
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    utilization = 1 - remaining / (bins + 1e-12)
    size_ratio = item / (bins + 1e-12)
    system_utilization = np.mean(utilization)
    diversity = np.std(bins) / (np.mean(bins) + 1e-12)
    
    adaptive_threshold = 0.4 + 0.4 * np.exp(-8 * size_ratio) + 0.2 * np.tanh(5 * system_utilization) - 0.1 * diversity
    load_factor = 0.6 + 0.6 * np.tanh(12 * (system_utilization - 0.55)) + 0.4 * np.exp(-4 * (1 - system_utilization)**3)
    
    penalty_phase1 = load_factor * (1 + np.tanh(25 * (utilization - adaptive_threshold)))
    penalty_phase2 = np.exp(8 * (utilization - adaptive_threshold))
    penalty_phase3 = np.exp(15 * (utilization - adaptive_threshold)**2)
    penalty_phase4 = np.exp(20 * (utilization - adaptive_threshold)**3)
    size_adjustment = 0.4 * np.log1p(6 * size_ratio) + 0.6 * (size_ratio ** 0.3) - 0.1 * size_ratio**2
    
    stability = 1.2 - 0.2 * np.tanh(10 * (system_utilization - 0.7))
    priority_scores = np.where(valid_mask,
                              stability * ((1/(remaining + 1e-12)) * (1 + size_adjustment) - 6.0 * penalty_phase1 * penalty_phase2 * penalty_phase3 * penalty_phase4 * (size_ratio**0.4)),
                              -np.inf)
    return priority_scores



# Function 9 - Score: -5.5223528734890674
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    utilization = (bins - remaining) / (bins + 1e-9)
    sigmoid_penalty = 1 / (1 + np.exp(-10 * (utilization - 0.5)))  # Smooth transition penalty
    dynamic_penalty = np.where(utilization < 0.5, 
                              (0.5 - utilization) ** 2, 
                              (utilization - 0.5) ** 3)  # Asymmetric penalty
    priority_scores = np.where(valid_mask, 
                             np.exp(-remaining) - sigmoid_penalty * dynamic_penalty, 
                             -np.inf)
    return priority_scores



# Function 10 - Score: -6.7032250919499
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    utilization = (bins - remaining) / (bins + 1e-9)
    
    # Multi-stage adaptive penalty system
    global_fill = np.mean(utilization)
    local_ratio = utilization / (global_fill + 1e-9)
    adaptive_curve = 1.8 + 4.2 * np.tanh(3.0 * (local_ratio - 1.1))
    penalty_stage1 = 1.0 / (1.0 + np.exp(-adaptive_curve * (utilization - 0.55)))
    penalty_stage2 = 0.7 * np.tanh(2.3 * (utilization - 0.4)) + 0.3
    combined_penalty = penalty_stage1 * penalty_stage2
    
    # Capacity-aware smoothing and entropy weighting
    capacity_factor = np.sqrt(bins / np.max(bins))
    entropy = -np.sum(utilization * np.log(utilization + 1e-9))
    dynamic_weight = 0.4 + 0.6 / (1.0 + np.exp(-2.5 * (entropy - 0.7)))
    
    priority_scores = np.where(
        valid_mask,
        np.exp(-0.4 * remaining * capacity_factor) - dynamic_weight * combined_penalty,
        -np.inf
    )
    return priority_scores



