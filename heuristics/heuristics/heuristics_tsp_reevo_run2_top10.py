# Top 10 functions for reevo run 2

# Function 1 - Score: -0.13695937232767924
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
    
    # Calculate exploration phase (0=start, 1=end)
    phase = 1 - len(unvisited_nodes) / len(distance_matrix)
    
    # Core metrics (vectorized)
    dist_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    cluster_density = np.mean(distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)], axis=1)
    
    # Normalized metrics (0-1 range)
    proximity = dist_current / np.max(dist_current)
    progress = (dist_current - dist_to_dest) / (np.max(np.abs(dist_current - dist_to_dest)) + 1e-6)
    exploration = 1 / (cluster_density + 1e-6)
    exploration = exploration / np.max(exploration)
    
    # Dynamic weights (non-linear transitions)
    w_proximity = 0.8 * np.exp(-2 * phase)  # Strong early, exponential decay
    w_progress = 0.1 + 0.7 * phase**2      # Quadratic growth
    w_exploration = 0.6 * (1 - phase)**1.5  # Strong early, power decay
    
    # Combined score with non-linear transformations
    score = (
        w_proximity * proximity +
        w_progress * np.tanh(3 * progress) +  # Emphasize progress direction
        w_exploration * exploration
    )
    
    return unvisited_nodes[np.argmin(score)]



# Function 2 - Score: -0.13820021189368392
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
    
    n_total = len(distance_matrix)
    n_remaining = len(unvisited_nodes)
    phase = 1 - n_remaining / n_total  # Normalized phase [0,1]
    
    # Core distance metrics
    to_node_dist = distance_matrix[current_node, unvisited_nodes]
    progress = to_node_dist - distance_matrix[unvisited_nodes, destination_node]
    
    # Enhanced clustering-aware exploration
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    local_density = np.sum(np.exp(-submatrix / np.median(submatrix)), axis=1)
    centrality = np.mean(submatrix, axis=1)
    isolation = np.max(submatrix, axis=1)
    
    exploration = (
        0.5 * local_density + 
        0.3 / (centrality + 1e-8) + 
        0.2 * isolation
    )
    
    # Three-level lookahead with phase-adaptive depth
    lookahead_depth = min(3, n_remaining - 1)
    if lookahead_depth > 1:
        future_cost = np.zeros(len(unvisited_nodes))
        for i, node in enumerate(unvisited_nodes):
            remaining = unvisited_nodes[unvisited_nodes != node]
            if len(remaining) >= lookahead_depth:
                closest = np.argmin(distance_matrix[node, remaining])
                future_cost[i] = distance_matrix[node, remaining[closest]] * (0.7 ** phase)
        progress -= future_cost * (0.4 + 0.3 * phase)
    
    # Dynamic normalization with phase-aware percentiles
    p_low = 20 + 60 * phase  # Adaptive percentile range
    p_high = 80 + 15 * (1 - phase)
    
    to_node_norm = to_node_dist / (np.percentile(to_node_dist, p_high) + 1e-8)
    progress_norm = progress / (np.percentile(np.abs(progress), p_high) + 1e-8)
    exploration_norm = exploration / (np.percentile(exploration, p_high) + 1e-8)
    
    # Phase-adaptive weights with sigmoid transitions
    proximity_weight = 0.7 / (1 + np.exp(5 * (phase - 0.4)))
    progress_weight = 0.6 / (1 + np.exp(-8 * (phase - 0.3)))
    exploration_weight = 0.5 * (1 - phase)**2
    
    # Non-linear scoring with adaptive exponents
    combined_score = (
        proximity_weight * (to_node_norm ** (1.2 - 0.4*phase)) +
        progress_weight * np.tanh(progress_norm * (2 - phase)) +
        exploration_weight * (exploration_norm ** (0.3 + 0.4*phase))
    )
    
    return unvisited_nodes[np.argmin(combined_score)]



# Function 3 - Score: -0.1385894612651023
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
    
    phase = 1 - len(unvisited_nodes) / len(distance_matrix)  # 0 (start) ¡ú 1 (end)
    
    # Core metrics
    to_node = distance_matrix[current_node, unvisited_nodes]
    progress = to_node - distance_matrix[unvisited_nodes, destination_node]
    exploration = 1 / (np.mean(distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)], axis=1) + 1e-6)
    
    # Normalization
    to_node_norm = to_node / np.max(to_node)
    progress_norm = progress / (np.std(progress) + 1e-6)
    exploration_norm = exploration / np.max(exploration)
    
    # Phase-dependent scoring
    proximity_score = to_node_norm * (1.1 - 0.6 * phase)
    progress_score = np.tanh(progress_norm) * (0.3 + 0.7 * phase)
    exploration_score = exploration_norm * (1.3 - 0.8 * phase)
    
    # Dynamic weights
    weights = np.array([
        0.6 - 0.3 * phase,  # proximity
        0.2 + 0.6 * phase,  # progress
        0.4 - 0.3 * phase   # exploration
    ])
    
    combined = (
        weights[0] * proximity_score +
        weights[1] * progress_score +
        weights[2] * exploration_score
    )
    
    return unvisited_nodes[np.argmin(combined)]



# Function 4 - Score: -0.13913887926771953
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
    
    phase = 1 - len(unvisited_nodes) / len(distance_matrix)  # Normalized [0,1]
    
    # Core metrics (vectorized)
    dist_current = distance_matrix[current_node, unvisited_nodes]
    dist_dest = distance_matrix[unvisited_nodes, destination_node]
    cluster_density = np.mean(distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)], axis=1)
    
    # Feature engineering
    proximity = dist_current / (np.max(dist_current) + 1e-8)
    progress = (dist_current - dist_dest) / (np.max(np.abs(dist_current - dist_dest)) + 1e-8)
    exploration = 1 / (cluster_density + 1e-8)  # Inverse density
    
    # Adaptive non-linear weights
    w_proximity = 0.8 * np.exp(-4 * phase**1.2)  # Fast exponential decay
    w_progress = 0.2 + 0.7 * (1 - np.exp(-3 * phase))  # Sigmoid-like growth
    w_exploration = 0.6 * (1 - phase)**3  # Cubic decay
    
    # Normalized scoring with non-linear features
    score = (
        w_proximity * proximity +
        w_progress * np.tanh(3 * progress) +  # Strong saturation
        w_exploration * (exploration / np.max(exploration))
    )
    
    return unvisited_nodes[np.argmin(score)]



# Function 5 - Score: -0.13919888535393293
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
    
    phase = 1 - len(unvisited_nodes) / len(distance_matrix)  # Exploration (0) ¡ú Exploitation (1)
    
    # Core metrics
    dist_current = distance_matrix[current_node, unvisited_nodes]
    progress = dist_current - distance_matrix[unvisited_nodes, destination_node]
    cluster_density = 1 / (np.mean(distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)], axis=1) + 1e-6)
    
    # Normalized metrics
    dist_norm = dist_current / (np.max(dist_current) + 1e-6)
    progress_norm = progress / (np.std(progress) + 1e-6)
    density_norm = cluster_density / (np.max(cluster_density) + 1e-6)
    
    # Phase-dependent scoring
    proximity_score = dist_norm * (1.2 - 0.7 * phase)
    progress_score = np.tanh(progress_norm) * (0.4 + 0.8 * phase)
    density_score = density_norm * (1.4 - phase)
    
    # Dynamic weight matrix
    weights = np.array([
        0.7 - 0.4 * phase,  # proximity
        0.1 + 0.7 * phase,  # progress
        0.5 - 0.4 * phase   # density
    ])
    
    combined_score = (
        weights[0] * proximity_score +
        weights[1] * progress_score +
        weights[2] * density_score
    )
    
    return unvisited_nodes[np.argmin(combined_score)]



# Function 6 - Score: -0.14027740205165978
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
    
    # Phase parameter (0=start, 1=end of tour)
    phase = 1 - len(unvisited_nodes) / len(distance_matrix)
    
    # Core metrics (vectorized)
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    avg_cluster_dist = np.mean(distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)], axis=1)
    
    # Normalized features (0-1 range)
    proximity = dist_from_current / np.max(dist_from_current)
    progress = (dist_from_current - dist_to_dest) / (np.max(np.abs(dist_from_current - dist_to_dest)) + 1e-8)
    exploration = 1 / (avg_cluster_dist + 1e-8)  # Inverse density measure
    
    # Phase-dependent weights (smooth transitions)
    w_proximity = 0.7 * np.exp(-3 * phase)  # Strong early, fast decay
    w_progress = 0.1 + 0.6 * phase**1.5     # Gradually increasing
    w_exploration = 0.5 * (1 - phase)**2    # Strong early, quadratic decay
    
    # Combined score with emphasis on directionality
    score = (
        w_proximity * proximity +
        w_progress * np.tanh(2 * progress) +  # Saturate extreme values
        w_exploration * (exploration / np.max(exploration))  # Normalized
    )
    
    return unvisited_nodes[np.argmin(score)]



# Function 7 - Score: -0.14055630202633157
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
    
    # Calculate adaptive phase (sigmoid transition)
    remaining_ratio = len(unvisited_nodes) / len(distance_matrix)
    phase = 1 / (1 + np.exp(8 * (remaining_ratio - 0.3)))  # Sharp transition at 30% remaining
    
    # Core metrics (vectorized)
    dist_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    cluster_sparsity = np.mean(distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)], axis=1)
    
    # Normalized metrics with non-linear scaling
    proximity = np.log1p(dist_current) / np.max(np.log1p(dist_current))
    progress = np.arctan(dist_current - dist_to_dest)  # Natural direction sensitivity
    progress = progress / (np.max(np.abs(progress)) + 1e-6)
    exploration = 1 / (cluster_sparsity + 1e-6)
    exploration = np.tanh(exploration / np.max(exploration))  # Soft capped exploration
    
    # Dynamic weights with adaptive transitions
    w_proximity = 0.7 * np.exp(-3 * phase**0.8)  # Fast early decay
    w_progress = 0.1 + 0.8 * phase**3           # Cubic growth for late phase
    w_exploration = 0.5 * (1 - phase)**2        # Quadratic decay
    
    # Combined score with emphasis on critical transitions
    score = (
        w_proximity * proximity +
        w_progress * np.tanh(4 * progress) +    # Strong progress emphasis
        w_exploration * exploration
    )
    
    return unvisited_nodes[np.argmin(score)]



# Function 8 - Score: -0.14074378652939046
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
    
    phase = 1 - len(unvisited_nodes) / len(distance_matrix)  # 0 (start) ¡ú 1 (end)
    
    # Core metrics (vectorized)
    dist_current = distance_matrix[current_node, unvisited_nodes]
    dist_dest = distance_matrix[unvisited_nodes, destination_node]
    cluster_density = 1 / (np.mean(distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)], axis=1) + 1e-8)
    
    # Normalized features
    proximity = dist_current / (np.mean(dist_current) + 1e-8)
    progress = (dist_current - dist_dest) / (np.std(dist_current - dist_dest) + 1e-8)
    exploration = cluster_density / (np.mean(cluster_density) + 1e-8)
    
    # Phase-dependent weights (smooth transitions)
    w_proximity = 0.7 * np.exp(-2 * phase)  # Strong early, exponential decay
    w_progress = 0.1 + 0.7 * phase**1.2    # Gradually increasing
    w_exploration = 0.6 * (1 - phase)**1.5  # Strong early, power decay
    
    # Combined score with non-linear transformations
    score = (
        w_proximity * proximity +
        w_progress * np.tanh(progress) +    # Bounded progress influence
        w_exploration * exploration
    )
    
    return unvisited_nodes[np.argmin(score)]



# Function 9 - Score: -0.1411242480127259
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
    
    # Adaptive phase (0=start, 1=end) with non-linear progression
    phase = 1 - (len(unvisited_nodes) / len(distance_matrix))**0.7
    
    # Core metrics
    dist_to_current = distance_matrix[current_node, unvisited_nodes]
    progress = dist_to_current - distance_matrix[unvisited_nodes, destination_node]
    cluster_density = 1 / (np.mean(distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)], axis=1) + 1e-6)
    
    # Normalized metrics with robust scaling
    proximity = dist_to_current / (np.max(dist_to_current) + 1e-6)
    progress_norm = progress / (np.max(np.abs(progress)) + 1e-6)
    density_norm = cluster_density / (np.max(cluster_density) + 1e-6)
    
    # Phase-dependent weights with smooth transitions
    w_proximity = 0.8 * np.exp(-2 * phase)  # Strong early, exponential decay
    w_progress = 0.1 + 0.7 * phase**1.5    # Non-linear growth
    w_exploration = 0.6 * (1 - phase)**2    # Quadratic decay
    
    # Combined score with non-linear transformations
    score = (
        w_proximity * proximity +
        w_progress * np.tanh(3 * progress_norm) +  # Emphasize progress direction
        w_exploration * density_norm
    )
    
    return unvisited_nodes[np.argmin(score)]



# Function 10 - Score: -0.14130366779886583
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
    
    phase = 1 - len(unvisited_nodes) / len(distance_matrix)  # Linear phase from 0 to 1
    
    # Core metrics
    dist_to_node = distance_matrix[current_node, unvisited_nodes]
    progress = dist_to_node - distance_matrix[unvisited_nodes, destination_node]
    cluster_cohesion = np.mean(distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)], axis=1)
    
    # Normalized metrics (0 to 1 range)
    proximity = dist_to_node / np.max(dist_to_node)
    progress_norm = progress / (np.max(np.abs(progress)) + 1e-6)
    exploration = 1 / (cluster_cohesion + 1e-6)
    exploration_norm = exploration / np.max(exploration)
    
    # Dynamic weights based on phase
    w_proximity = 0.7 * (1 - phase)  # Strong early, fades
    w_progress = 0.1 + 0.6 * phase   # Grows with phase
    w_exploration = 0.5 * (1 - phase)  # Strong early
    
    # Combined score
    score = (
        w_proximity * proximity +
        w_progress * np.tanh(progress_norm) +
        w_exploration * exploration_norm
    )
    
    return unvisited_nodes[np.argmin(score)]



