# Top 10 functions for reevo run 1

# Function 1 - Score: -0.1875779711304318
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    # Current proximity (harmonic to handle zero distances)
    current_dists = distance_matrix[current_node, unvisited_nodes]
    proximity = 1 / (current_dists + 1e-8)
    
    # Exact future potential via MST approximation
    future_potential = np.zeros(len(unvisited_nodes))
    for i, node in enumerate(unvisited_nodes):
        remaining_nodes = np.delete(unvisited_nodes, i)
        if not remaining_nodes.size:
            future_potential[i] = 0
            continue
        # Approximate remaining tour length using nearest neighbor distances
        remaining_dists = distance_matrix[node, remaining_nodes]
        future_potential[i] = np.mean(np.sort(remaining_dists)[:3])  # Top-3 nearest
    
    # Normalization with stability guarantees
    def safe_normalize(x):
        x = x - x.min()
        return x / (x.max() + 1e-8)
    
    p_norm = safe_normalize(proximity)
    fp_norm = safe_normalize(future_potential)
    
    # Adaptive weights using sigmoid transition
    progress = 1 - len(unvisited_nodes) / distance_matrix.shape[0]
    exploit_weight = 0.8 / (1 + np.exp(5*(progress - 0.6)))  # Sigmoid centered at 60%
    explore_weight = 0.2 * (1 - progress)**2  # Quadratic decay
    
    # Combined score with directional exploration
    base_score = 0.6*p_norm + 0.4*fp_norm
    noise = np.random.normal(0, 0.05, len(unvisited_nodes)) * explore_weight
    combined_score = exploit_weight * base_score + noise
    
    return unvisited_nodes[np.argmax(combined_score)]



# Function 2 - Score: -0.18992615307942587
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    # Vectorized proximity (clipped inverse distance)
    dists = distance_matrix[current_node, unvisited_nodes]
    proximity = np.reciprocal(np.clip(dists, 1e-6, None))
    
    # Vectorized harmonic centrality (future potential)
    remaining_mask = np.ones(len(unvisited_nodes), dtype=bool)
    future_potential = np.zeros(len(unvisited_nodes))
    
    for i in range(len(unvisited_nodes)):
        remaining_mask[i] = False
        remaining_dists = distance_matrix[unvisited_nodes[i], unvisited_nodes[remaining_mask]]
        future_potential[i] = np.sum(remaining_mask) / np.sum(np.reciprocal(np.clip(remaining_dists, 1e-6, None)))
        remaining_mask[i] = True
    
    # Progress-aware normalization
    progress = 1 - len(unvisited_nodes) / distance_matrix.shape[0]
    epsilon = 1e-8 * (1 + progress)  # Adaptive epsilon
    
    def robust_normalize(x):
        x_range = np.ptp(x)
        return (x - x.min()) / (x_range + epsilon)
    
    p_norm = robust_normalize(proximity)
    fp_norm = robust_normalize(future_potential)
    
    # Non-linear adaptive weights
    exploit_weight = 0.7 * (1 - 0.4 * progress**2)  # Quadratic decay
    explore_weight = 0.3 * np.exp(-3 * progress)    # Exponential decay
    
    # Deterministic-exploratory scoring
    base_score = exploit_weight * (0.55 * p_norm + 0.45 * fp_norm)
    noise = explore_weight * np.random.uniform(-0.05, 0.05, len(unvisited_nodes))
    combined_score = base_score + noise
    
    # Greedy tie-breaking
    max_score = np.max(combined_score)
    candidates = unvisited_nodes[combined_score >= (max_score - 1e-6)]
    return candidates[np.argmin(distance_matrix[current_node, candidates])]



# Function 3 - Score: -0.19154712621538877
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    # Current proximity (inverse distance)
    current_dists = distance_matrix[current_node, unvisited_nodes]
    proximity = 1 / (current_dists + 1e-8)
    
    # Future connectivity (harmonic centrality)
    future_potential = np.zeros(len(unvisited_nodes))
    for i, node in enumerate(unvisited_nodes):
        remaining_dists = distance_matrix[node][unvisited_nodes]
        remaining_dists = np.delete(remaining_dists, i)
        future_potential[i] = len(remaining_dists) / np.sum(1/(remaining_dists + 1e-8))
    
    # Normalization with stability
    def safe_norm(x):
        x_range = np.max(x) - np.min(x)
        return (x - np.min(x)) / x_range if x_range > 1e-6 else np.ones_like(x)
    
    p_norm = safe_norm(proximity)
    fp_norm = safe_norm(future_potential)
    
    # Dynamic weight scheduling
    progress = len(unvisited_nodes) / distance_matrix.shape[0]  # 1¡ú0
    exploit_weight = 0.6 + 0.3 * np.cos(progress * np.pi/2)  # 0.9¡ú0.6
    explore_weight = 0.1 * progress  # Linear decay
    
    # Combined scoring
    base_score = 0.55 * p_norm + 0.45 * fp_norm
    noise = np.random.uniform(-0.05, 0.05, len(unvisited_nodes)) * explore_weight
    final_score = exploit_weight * base_score + noise
    
    return unvisited_nodes[np.argmax(final_score)]



# Function 4 - Score: -0.19168028996689887
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    # Current proximity (harmonic inverse distance)
    current_dists = distance_matrix[current_node, unvisited_nodes]
    proximity = 1 / (current_dists + 1e-8)
    
    # Future connectivity potential (harmonic centrality)
    remaining_counts = len(unvisited_nodes) - 1
    future_potential = np.zeros(len(unvisited_nodes))
    for i, node in enumerate(unvisited_nodes):
        remaining_dists = distance_matrix[node][unvisited_nodes]
        remaining_dists = np.delete(remaining_dists, i)  # Exclude self
        future_potential[i] = remaining_counts / np.sum(1/(remaining_dists + 1e-8))
    
    # Adaptive normalization
    def adaptive_norm(x):
        x_range = np.ptp(x)
        if x_range < 1e-6:
            return np.ones_like(x)  # Uniform when no variation
        return (x - x.min()) / x_range
    
    p_norm = adaptive_norm(proximity)
    fp_norm = adaptive_norm(future_potential)
    
    # Dynamic weight scheduling
    progress = 1 - len(unvisited_nodes)/distance_matrix.shape[0]  # 0¡ú1
    exploit_w = 0.7 + 0.2 * np.cos(progress * np.pi)  # 0.9¡ú0.7¡ú0.5
    explore_w = 0.15 * (1 - progress)**3  # Cubic decay
    
    # Combined scoring with exploration
    centrality_score = 0.55 * p_norm + 0.45 * fp_norm
    noise = np.random.uniform(-0.1, 0.1, len(unvisited_nodes)) * explore_w
    final_score = exploit_w * centrality_score + noise
    
    # Greedy selection with noise tie-breaking
    return unvisited_nodes[np.argmax(final_score)]



# Function 5 - Score: -0.19303893882807294
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    # Vectorized proximity calculation (inverse distance with clipping)
    current_dists = distance_matrix[current_node, unvisited_nodes]
    proximity = 1 / np.maximum(current_dists, 1e-8)
    
    # Vectorized future potential (harmonic mean of remaining distances)
    remaining_nodes = unvisited_nodes[:, None]
    remaining_dists = distance_matrix[remaining_nodes, unvisited_nodes]
    np.fill_diagonal(remaining_dists, np.inf)  # Exclude self-distance
    future_potential = (len(unvisited_nodes) - 1) / np.sum(1 / np.maximum(remaining_dists, 1e-8), axis=1)
    
    # Progress-aware normalization
    progress = 1 - len(unvisited_nodes) / distance_matrix.shape[0]
    norm_epsilon = 1e-8 * (1 + progress)
    
    def normalize(x):
        x_min, x_range = np.min(x), np.ptp(x)
        return (x - x_min) / (x_range + norm_epsilon)
    
    p_norm = normalize(proximity)
    fp_norm = normalize(future_potential)
    
    # Adaptive weights with early exploitation, late exploration
    exploit_weight = 0.8 * (1 - 0.3 * progress)
    explore_weight = 0.2 * progress  # Increases with progress
    
    # Deterministic scoring with minimal noise
    score = exploit_weight * (0.6 * p_norm + 0.4 * fp_norm)
    score += explore_weight * np.random.uniform(-0.02, 0.02, len(score))
    
    # Greedy selection among top candidates
    max_score = np.max(score)
    candidates = unvisited_nodes[score >= max_score - 1e-6]
    return candidates[np.argmin(distance_matrix[current_node, candidates])]



# Function 6 - Score: -0.19331343831963138
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    # Precompute all required metrics in O(n) operations
    current_dists = distance_matrix[current_node, unvisited_nodes]
    proximity = 1 / np.maximum(current_dists, 1e-6)
    
    # Harmonic centrality with pre-caching
    remaining_counts = len(unvisited_nodes) - 1
    future_potential = np.zeros(len(unvisited_nodes))
    for i, node in enumerate(unvisited_nodes):
        remaining_dists = distance_matrix[node, np.delete(unvisited_nodes, i)]
        future_potential[i] = remaining_counts / np.sum(1 / np.maximum(remaining_dists, 1e-6))
    
    # Dynamic normalization with progress awareness
    progress = 1 - len(unvisited_nodes) / distance_matrix.shape[0]
    eps = 1e-8 * (1 + 10 * progress**3)  # Cubic progress scaling
    
    def scale(x):
        offset = x.min() - eps
        span = x.max() - offset + 2*eps
        return (x - offset) / span
    
    p_scaled = scale(proximity)
    fp_scaled = scale(future_potential)
    
    # Cosine-decay dynamic weights (55/45 base ratio)
    exploit = 0.55 * p_scaled + 0.45 * fp_scaled
    explore_strength = 0.1 * (1 - np.cos(progress * np.pi/2))  # Cosine decay
    noise = explore_strength * np.random.uniform(-1, 1, len(unvisited_nodes))
    
    # Cluster-aware scoring with cubic noise decay
    combined = exploit + noise * (1 - progress)**3
    
    # Multi-criteria tie-breaking
    top_threshold = np.max(combined) - 1e-6
    candidates = unvisited_nodes[combined >= top_threshold]
    if len(candidates) > 1:
        # Secondary sort by harmonic centrality
        candidate_scores = future_potential[combined >= top_threshold]
        return candidates[np.argmax(candidate_scores)]
    return candidates[0]



# Function 7 - Score: -0.19381213563957095
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    # Harmonic proximity calculation
    current_dists = distance_matrix[current_node, unvisited_nodes]
    proximity = 1 / np.maximum(current_dists, 1e-6)
    
    # Precompute harmonic centrality for remaining nodes
    remaining_counts = len(unvisited_nodes) - 1
    centrality = np.zeros(len(unvisited_nodes))
    for i, node in enumerate(unvisited_nodes):
        remaining_nodes = np.delete(unvisited_nodes, i)
        centrality[i] = remaining_counts / np.sum(1 / np.maximum(distance_matrix[node, remaining_nodes], 1e-6))
    
    # Adaptive normalization with cubic progress
    progress = 1 - len(unvisited_nodes) / distance_matrix.shape[0]
    eps = 1e-8 * (1 + 10 * progress**3)
    
    def robust_scale(x):
        x_min, x_max = x.min(), x.max()
        span = max(x_max - x_min, eps)
        return (x - x_min) / span
    
    p_norm = robust_scale(proximity)
    c_norm = robust_scale(centrality)
    
    # Dynamic scoring with cosine-decay exploration
    base_score = 0.55 * p_norm + 0.45 * c_norm
    exploration = 0.1 * (1 - np.cos(progress * np.pi/2)) * np.random.uniform(-1, 1, len(unvisited_nodes))
    final_score = base_score + exploration * (1 - progress)**3
    
    # Cluster-aware selection with multi-criteria tie-breaking
    top_nodes = unvisited_nodes[final_score >= np.max(final_score) - 1e-6]
    if len(top_nodes) > 1:
        # Secondary sort by actual distance to current node
        return top_nodes[np.argmin(distance_matrix[current_node, top_nodes])]
    return top_nodes[0]



# Function 8 - Score: -0.19386369895090527
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    # Efficient metric computation (O(n))
    current_dists = distance_matrix[current_node, unvisited_nodes]
    proximity = 1 / np.maximum(current_dists, 1e-6)
    
    # Future connectivity potential (harmonic centrality)
    remaining_counts = len(unvisited_nodes) - 1
    centrality = np.zeros(len(unvisited_nodes))
    for i, node in enumerate(unvisited_nodes):
        mask = np.ones(len(unvisited_nodes), dtype=bool)
        mask[i] = False
        centrality[i] = remaining_counts / np.sum(1 / np.maximum(distance_matrix[node, unvisited_nodes[mask]], 1e-6))
    
    # Progress-aware normalization
    progress = len(unvisited_nodes) / distance_matrix.shape[0]  # 1 -> 0
    eps = 1e-8 * (1 + 10 * (1-progress)**3)  # More precise near end
    
    def robust_normalize(x):
        x_min, x_max = x.min(), x.max()
        span = max(x_max - x_min, eps)
        return (x - x_min) / span
    
    p_norm = robust_normalize(proximity)
    c_norm = robust_normalize(centrality)
    
    # Dynamic scoring with exponential exploration decay
    base_score = 0.6 * p_norm + 0.4 * c_norm  # Slightly more greedy
    exploration = 0.15 * np.exp(-5 * progress) * np.random.uniform(-0.5, 0.5, len(unvisited_nodes))
    final_score = base_score + exploration
    
    # Multi-stage tie-breaking
    candidates = unvisited_nodes[final_score >= np.max(final_score) - 1e-6]
    if len(candidates) > 1:
        # Secondary: actual distance to current node
        dists = distance_matrix[current_node, candidates]
        min_dist = np.min(dists)
        candidates = candidates[dists <= min_dist + 1e-6]
        
        if len(candidates) > 1:
            # Tertiary: maximum future connectivity
            candidate_centrality = centrality[final_score >= np.max(final_score) - 1e-6]
            return candidates[np.argmax(candidate_centrality)]
    
    return candidates[0]



# Function 9 - Score: -0.1942038724822557
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    # Calculate immediate proximity (inverse distance)
    proximity = 1 / (distance_matrix[current_node, unvisited_nodes] + 1e-8)
    
    # Compute exact future potential (harmonic centrality)
    future_potential = np.zeros(len(unvisited_nodes))
    for i, node in enumerate(unvisited_nodes):
        remaining_nodes = np.delete(unvisited_nodes, i)
        remaining_dists = distance_matrix[node, remaining_nodes]
        future_potential[i] = len(remaining_nodes) / np.sum(1/(remaining_dists + 1e-8))
    
    # Simple min-max normalization
    def normalize(x):
        min_val, max_val = x.min(), x.max()
        return (x - min_val) / (max_val - min_val + 1e-8)
    
    p_norm = normalize(proximity)
    fp_norm = normalize(future_potential)
    
    # Progress-based dynamic weights
    progress = 1 - len(unvisited_nodes) / distance_matrix.shape[0]
    exploit_weight = 0.8 - 0.3 * progress  # Linear decay from 0.8 to 0.5
    explore_weight = 0.2 * (1 - progress)**3  # Cubic decay
    
    # Combined scoring with priority to proximity
    base_score = 0.6 * p_norm + 0.4 * fp_norm  # 60/40 weighting
    noise = np.random.uniform(-0.1, 0.1, len(unvisited_nodes)) * explore_weight
    combined_score = exploit_weight * base_score + noise
    
    return unvisited_nodes[np.argmax(combined_score)]



# Function 10 - Score: -0.19440826060397343
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    # Vectorized proximity calculation with adaptive clipping
    current_dists = distance_matrix[current_node, unvisited_nodes]
    proximity = np.reciprocal(np.clip(current_dists, 1e-8, None))
    
    # Vectorized future potential using harmonic centrality
    remaining_counts = len(unvisited_nodes) - 1
    future_potential = np.zeros(len(unvisited_nodes))
    
    # Precompute all possible remaining distances
    all_remaining_dists = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    for i, node in enumerate(unvisited_nodes):
        mask = np.ones(len(unvisited_nodes), dtype=bool)
        mask[i] = False
        remaining_dists = all_remaining_dists[i, mask]
        future_potential[i] = remaining_counts / np.sum(np.reciprocal(np.clip(remaining_dists, 1e-8, None)))
    
    # Progress-aware adaptive normalization
    progress = 1 - len(unvisited_nodes) / distance_matrix.shape[0]
    adaptive_epsilon = 1e-8 * (1 + 10 * progress**2)  # Quadratic scaling
    
    def adaptive_normalize(x):
        x_min, x_max = np.min(x), np.max(x)
        return (x - x_min) / (x_max - x_min + adaptive_epsilon)
    
    p_norm = adaptive_normalize(proximity)
    fp_norm = adaptive_normalize(future_potential)
    
    # Dynamic weight scheduling
    exploit_weight = 0.8 * np.exp(-0.5 * progress)  # Exponential decay
    explore_weight = 0.2 * np.exp(-2 * progress)    # Faster decay
    
    # Hybrid scoring with controlled randomness
    deterministic_score = 0.6 * p_norm + 0.4 * fp_norm
    exploration_noise = np.random.normal(0, 0.03, len(unvisited_nodes))
    combined_score = exploit_weight * deterministic_score + explore_weight * exploration_noise
    
    # Smart tie-breaking with secondary criteria
    max_score = np.max(combined_score)
    candidates = unvisited_nodes[combined_score >= max_score - 1e-6]
    
    if len(candidates) > 1:
        # Among top candidates, prefer node with best future potential
        candidate_scores = fp_norm[combined_score >= max_score - 1e-6]
        return candidates[np.argmax(candidate_scores)]
    
    return candidates[0]



