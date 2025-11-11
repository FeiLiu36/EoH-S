# Top 10 functions for reevo run 2

# Function 1 - Score: -0.24655408735926804
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Optimized capacity threshold with smoother sigmoid transition
    max_demand = np.max(demands[unvisited_nodes])
    capacity_ratio = rest_capacity / (max_demand + 1e-10)
    threshold = 0.75 + 0.22 / (1 + np.exp(-25 * (capacity_ratio - 0.73)))  # Balanced transition
    
    # Strategic candidate selection with relaxed filtering
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    critical_mask = demands[unvisited_nodes] > (rest_capacity * threshold)
    candidates = unvisited_nodes[feasible_mask | critical_mask]
    
    if len(candidates) == 0:
        return depot
    
    # Vectorized feature computation
    dist_current = distance_matrix[current_node, candidates]
    dist_depot = distance_matrix[candidates, depot]
    demand_ratio = demands[candidates] / (rest_capacity + 1e-10)
    
    # Enhanced adaptive scaling with consistent normalization
    def balanced_scale(x):
        x_range = np.max(x) - np.min(x) + 1e-10
        x_norm = (x - np.min(x)) / x_range
        return np.log1p(7.5 * x_norm) / np.log(3.8)  # Balanced scaling
    
    proximity = balanced_scale(1 / (dist_current + 1e-8))
    urgency = balanced_scale(demand_ratio * np.exp(4.2 * (1 - capacity_ratio)))
    detour_penalty = balanced_scale(1 + np.maximum(0, dist_depot - dist_current) / (dist_current + 1e-8))
    
    # Dynamic weight adaptation with smoother curves
    capacity_tension = 1 - capacity_ratio
    w_proximity = 0.87 - 0.79 / (1 + np.exp(-24 * (capacity_tension - 0.27)))  # Smoother transition
    w_urgency = 0.62 / (1 + np.exp(-20 * (capacity_tension - 0.18)))  # Balanced urgency
    w_detour = 0.19 * (1 - np.exp(-6.0 * capacity_tension))  # Optimized penalty
    
    # Adaptive exploration with early-stage emphasis
    progress = 1 - len(candidates) / len(unvisited_nodes)
    exploration = 0.012 * np.exp(-18 * progress) * np.random.randn(len(candidates))
    
    # Optimized scoring with balanced components
    scores = (
        w_proximity * proximity +
        w_urgency * urgency -
        w_detour * detour_penalty +
        exploration
    )
    
    # Balanced critical boost
    is_critical = critical_mask[feasible_mask | critical_mask]
    critical_boost = 1.74 + 0.90 * (1 - np.exp(-5.1 * capacity_tension))
    scores[is_critical] *= critical_boost
    
    return candidates[np.argmax(scores)]



# Function 2 - Score: -0.2470094547011063
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Dynamic capacity threshold with adaptive sigmoid
    capacity_ratio = rest_capacity / (np.max(demands[unvisited_nodes]) + 1e-10)
    threshold = 0.68 + 0.31 / (1 + np.exp(-25 * (capacity_ratio - 0.72)))  # Smoother transition
    
    # Intelligent candidate filtering with tiered urgency
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    critical_mask = demands[unvisited_nodes] > (rest_capacity * threshold)
    candidates = unvisited_nodes[feasible_mask | critical_mask]
    
    if len(candidates) == 0:
        return depot
    
    # Vectorized multi-feature extraction
    dist_current = distance_matrix[current_node, candidates]
    dist_depot = distance_matrix[candidates, depot]
    demand_ratio = demands[candidates] / (rest_capacity + 1e-10)
    
    # Optimized scaling functions with adaptive curvature
    def dynamic_scale(x):
        x_range = np.max(x) - np.min(x) + 1e-10
        x_norm = (x - np.min(x)) / x_range
        return np.log1p(7.5 * x_norm) / np.log(3.8)  # Balanced scaling
    
    proximity = dynamic_scale(1 / (dist_current + 1e-8))
    urgency = dynamic_scale(demand_ratio * np.exp(4.2 * (1 - capacity_ratio)))
    detour_cost = dynamic_scale(1 + np.maximum(0, dist_depot - dist_current) / (dist_current + 1e-8))
    
    # Self-adjusting weights with capacity awareness
    capacity_tension = 1 - capacity_ratio
    w_proximity = 0.85 - 0.78 / (1 + np.exp(-24 * (capacity_tension - 0.28)))
    w_urgency = 0.67 / (1 + np.exp(-19 * (capacity_tension - 0.19)))
    w_detour = 0.22 * (1 - np.exp(-5.8 * capacity_tension))
    
    # Adaptive exploration with route progression
    route_progress = 1 - len(candidates) / len(unvisited_nodes)
    exploration = 0.009 * np.exp(-18 * route_progress) * np.random.randn(len(candidates))
    
    # Multi-objective scoring with critical boost
    scores = (
        w_proximity * proximity +
        w_urgency * urgency -
        w_detour * detour_cost +
        exploration
    )
    
    # Tiered critical node enhancement
    is_critical = critical_mask[feasible_mask | critical_mask]
    critical_boost = 1.82 + 0.88 * (1 - np.exp(-5.1 * capacity_tension))
    scores[is_critical] *= critical_boost
    
    return candidates[np.argmax(scores)]



# Function 3 - Score: -0.24729956859418326
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    if not unvisited_nodes.size:
        return depot
    
    # Adaptive capacity threshold with enhanced smoothness
    max_demand = demands[unvisited_nodes].max()
    capacity_ratio = rest_capacity / (max_demand + 1e-10)
    threshold = 0.72 + 0.27 / (1 + np.exp(-20 * (capacity_ratio - 0.70)))
    
    # Intelligent candidate filtering
    feasible = demands[unvisited_nodes] <= rest_capacity
    critical = demands[unvisited_nodes] > (rest_capacity * threshold)
    candidates = unvisited_nodes[feasible | critical]
    
    if not candidates.size:
        return depot
    
    # Efficient feature computation
    dist_current = distance_matrix[current_node, candidates]
    dist_depot = distance_matrix[candidates, depot]
    demand_ratio = demands[candidates] / (rest_capacity + 1e-10)
    
    # Optimized scaling with adaptive range
    def adaptive_scale(x):
        x_min = x.min()
        x_range = x.max() - x_min + 1e-10
        x_norm = (x - x_min) / x_range
        return np.log1p(6.8 * x_norm) / np.log(3.5)
    
    proximity = adaptive_scale(1 / (dist_current + 1e-8))
    urgency = adaptive_scale(demand_ratio * np.exp(3.8 * (1 - capacity_ratio)))
    detour_cost = adaptive_scale(1 + np.maximum(0, dist_depot - dist_current) / (dist_current + 1e-8))
    
    # Dynamically adjusted weights
    capacity_tension = 1 - capacity_ratio
    w_proximity = 0.84 - 0.72 / (1 + np.exp(-20 * (capacity_tension - 0.22)))
    w_urgency = 0.58 / (1 + np.exp(-16 * (capacity_tension - 0.12)))
    w_detour = 0.19 * (1 - np.exp(-5.0 * capacity_tension))
    
    # Progress-sensitive exploration
    progress = 1 - candidates.size / unvisited_nodes.size
    exploration = 0.008 * np.exp(-20 * progress) * np.random.randn(candidates.size)
    
    # Balanced scoring with adaptive components
    scores = (
        w_proximity * proximity +
        w_urgency * urgency -
        w_detour * detour_cost +
        exploration
    )
    
    # Smart critical node handling
    is_critical = critical[feasible | critical]
    critical_boost = 1.65 + 0.75 * (1 - np.exp(-4.2 * capacity_tension))
    scores[is_critical] *= critical_boost
    
    return candidates[np.argmax(scores)]



# Function 4 - Score: -0.2474566564763805
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Precision-tuned capacity threshold with optimized sigmoid
    max_demand = np.max(demands[unvisited_nodes])
    capacity_ratio = rest_capacity / (max_demand + 1e-10)
    threshold = 0.76 + 0.23 / (1 + np.exp(-28 * (capacity_ratio - 0.74)))  # Balanced transition
    
    # Strategic candidate selection
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    critical_mask = demands[unvisited_nodes] > (rest_capacity * threshold)
    candidates = unvisited_nodes[feasible_mask | critical_mask]
    
    if len(candidates) == 0:
        return depot
    
    # Vectorized feature computation
    dist_current = distance_matrix[current_node, candidates]
    dist_depot = distance_matrix[candidates, depot]
    demand_ratio = demands[candidates] / (rest_capacity + 1e-10)
    
    # Enhanced adaptive scaling with optimal curvature
    def adaptive_scale(x):
        x_range = np.max(x) - np.min(x) + 1e-10
        x_norm = (x - np.min(x)) / x_range
        return np.log1p(8.0 * x_norm) / np.log(4.0)  # Consistent scaling
    
    proximity = adaptive_scale(1 / (dist_current + 1e-8))
    urgency = adaptive_scale(demand_ratio * np.exp(4.4 * (1 - capacity_ratio)))
    detour_penalty = adaptive_scale(1 + np.maximum(0, dist_depot - dist_current) / (dist_current + 1e-8))
    
    # Dynamic weight adaptation with precision tuning
    capacity_tension = 1 - capacity_ratio
    w_proximity = 0.89 - 0.81 / (1 + np.exp(-26 * (capacity_tension - 0.26)))  # Smoother curve
    w_urgency = 0.64 / (1 + np.exp(-21 * (capacity_tension - 0.17)))  # Responsive urgency
    w_detour = 0.18 * (1 - np.exp(-6.2 * capacity_tension))  # Optimized penalty
    
    # Controlled exploration with adaptive decay
    progress = 1 - len(candidates) / len(unvisited_nodes)
    exploration = 0.011 * np.exp(-20 * progress) * np.random.randn(len(candidates))
    
    # Optimized scoring with balanced components
    scores = (
        w_proximity * proximity +
        w_urgency * urgency -
        w_detour * detour_penalty +
        exploration
    )
    
    # Precision-tuned critical boost
    is_critical = critical_mask[feasible_mask | critical_mask]
    critical_boost = 1.76 + 0.92 * (1 - np.exp(-5.3 * capacity_tension))
    scores[is_critical] *= critical_boost
    
    return candidates[np.argmax(scores)]



# Function 5 - Score: -0.2476017202279563
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Enhanced capacity threshold with optimized sigmoid parameters
    max_demand = np.max(demands[unvisited_nodes])
    capacity_ratio = rest_capacity / (max_demand + 1e-10)
    threshold = 0.78 + 0.21 / (1 + np.exp(-30 * (capacity_ratio - 0.76)))  # Smoother transition
    
    # Strategic candidate selection with adaptive filtering
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    critical_mask = demands[unvisited_nodes] > (rest_capacity * threshold)
    candidates = unvisited_nodes[feasible_mask | critical_mask]
    
    if len(candidates) == 0:
        return depot
    
    # Optimized feature computation
    dist_current = distance_matrix[current_node, candidates]
    dist_depot = distance_matrix[candidates, depot]
    demand_ratio = demands[candidates] / (rest_capacity + 1e-10)
    
    # Enhanced scaling function with optimal parameters
    def optimal_scale(x):
        x_range = np.max(x) - np.min(x) + 1e-10
        x_norm = (x - np.min(x)) / x_range
        return np.log1p(8.5 * x_norm) / np.log(4.2)  # Improved scaling curve
    
    proximity = optimal_scale(1 / (dist_current + 1e-8))
    urgency = optimal_scale(demand_ratio * np.exp(4.5 * (1 - capacity_ratio)))
    detour_cost = optimal_scale(1 + np.maximum(0, dist_depot - dist_current) / (dist_current + 1e-8))
    
    # Dynamic weight adaptation with refined curves
    capacity_tension = 1 - capacity_ratio
    w_proximity = 0.90 - 0.82 / (1 + np.exp(-28 * (capacity_tension - 0.27)))  # Smoother adaptation
    w_urgency = 0.66 / (1 + np.exp(-22 * (capacity_tension - 0.18)))  # More responsive curve
    w_detour = 0.19 * (1 - np.exp(-6.5 * capacity_tension))  # Balanced penalty
    
    # Adaptive exploration with progress-aware decay
    progress = 1 - len(candidates) / len(unvisited_nodes)
    exploration = 0.012 * np.exp(-22 * progress) * np.random.randn(len(candidates))
    
    # Optimized scoring with component balancing
    scores = (
        w_proximity * proximity +
        w_urgency * urgency -
        w_detour * detour_cost +
        exploration
    )
    
    # Enhanced critical boost with capacity awareness
    is_critical = critical_mask[feasible_mask | critical_mask]
    critical_boost = 1.78 + 0.95 * (1 - np.exp(-5.5 * capacity_tension))
    scores[is_critical] *= critical_boost
    
    return candidates[np.argmax(scores)]



# Function 6 - Score: -0.24769690551677911
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    if not unvisited_nodes.size:
        return depot
    
    # Enhanced capacity threshold with smoother transition
    max_demand = demands[unvisited_nodes].max()
    capacity_ratio = rest_capacity / (max_demand + 1e-10)
    threshold = 0.74 + 0.25 / (1 + np.exp(-24 * (capacity_ratio - 0.71)))
    
    # Smart candidate selection
    feasible = demands[unvisited_nodes] <= rest_capacity
    critical = demands[unvisited_nodes] > (rest_capacity * threshold)
    candidates = unvisited_nodes[feasible | critical]
    
    if not candidates.size:
        return depot
    
    # Vectorized feature computation
    dist_current = distance_matrix[current_node, candidates]
    dist_depot = distance_matrix[candidates, depot]
    demand_ratio = demands[candidates] / (rest_capacity + 1e-10)
    
    # Precision-tuned scaling function
    def optimal_scale(x):
        x_min = x.min()
        x_range = x.max() - x_min + 1e-10
        x_norm = (x - x_min) / x_range
        return np.log1p(7.3 * x_norm) / np.log(3.7)
    
    proximity = optimal_scale(1 / (dist_current + 1e-8))
    urgency = optimal_scale(demand_ratio * np.exp(4.1 * (1 - capacity_ratio)))
    detour_cost = optimal_scale(1 + np.maximum(0, dist_depot - dist_current) / (dist_current + 1e-8))
    
    # Dynamically balanced weights
    capacity_tension = 1 - capacity_ratio
    w_proximity = 0.87 - 0.77 / (1 + np.exp(-23 * (capacity_tension - 0.24)))
    w_urgency = 0.62 / (1 + np.exp(-19 * (capacity_tension - 0.14)))
    w_detour = 0.18 * (1 - np.exp(-5.5 * capacity_tension))
    
    # Progress-aware exploration
    progress = 1 - candidates.size / unvisited_nodes.size
    exploration = 0.009 * np.exp(-23 * progress) * np.random.randn(candidates.size)
    
    # Optimized scoring function
    scores = (
        w_proximity * proximity +
        w_urgency * urgency -
        w_detour * detour_cost +
        exploration
    )
    
    # Enhanced critical node handling
    is_critical = critical[feasible | critical]
    critical_boost = 1.71 + 0.82 * (1 - np.exp(-4.8 * capacity_tension))
    scores[is_critical] *= critical_boost
    
    return candidates[np.argmax(scores)]



# Function 7 - Score: -0.24801695496435847
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Precision-tuned capacity threshold with optimized transition
    max_demand = np.max(demands[unvisited_nodes])
    capacity_ratio = rest_capacity / (max_demand + 1e-10)
    threshold = 0.78 + 0.21 / (1 + np.exp(-30 * (capacity_ratio - 0.72)))  # Sharper transition
    
    # Strategic candidate selection with refined filtering
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    critical_mask = demands[unvisited_nodes] > (rest_capacity * threshold)
    candidates = unvisited_nodes[feasible_mask | critical_mask]
    
    if len(candidates) == 0:
        return depot
    
    # Vectorized feature computation
    dist_current = distance_matrix[current_node, candidates]
    dist_depot = distance_matrix[candidates, depot]
    demand_ratio = demands[candidates] / (rest_capacity + 1e-10)
    
    # Enhanced adaptive scaling with optimal curvature
    def optimized_scale(x):
        x_range = np.max(x) - np.min(x) + 1e-10
        x_norm = (x - np.min(x)) / x_range
        return np.log1p(8.5 * x_norm) / np.log(4.2)  # Fine-tuned scaling
    
    proximity = optimized_scale(1 / (dist_current + 1e-8))
    urgency = optimized_scale(demand_ratio * np.exp(4.6 * (1 - capacity_ratio)))
    detour_penalty = optimized_scale(1 + np.maximum(0, dist_depot - dist_current) / (dist_current + 1e-8))
    
    # Dynamic weight adaptation with precision tuning
    capacity_tension = 1 - capacity_ratio
    w_proximity = 0.91 - 0.83 / (1 + np.exp(-28 * (capacity_tension - 0.24)))  # Optimized curve
    w_urgency = 0.66 / (1 + np.exp(-23 * (capacity_tension - 0.16)))  # Responsive urgency
    w_detour = 0.17 * (1 - np.exp(-6.5 * capacity_tension))  # Fine-tuned penalty
    
    # Controlled exploration with adaptive decay
    progress = 1 - len(candidates) / len(unvisited_nodes)
    exploration = 0.010 * np.exp(-22 * progress) * np.random.randn(len(candidates))
    
    # Optimized scoring with balanced components
    scores = (
        w_proximity * proximity +
        w_urgency * urgency -
        w_detour * detour_penalty +
        exploration
    )
    
    # Precision-tuned critical boost
    is_critical = critical_mask[feasible_mask | critical_mask]
    critical_boost = 1.78 + 0.94 * (1 - np.exp(-5.4 * capacity_tension))
    scores[is_critical] *= critical_boost
    
    return candidates[np.argmax(scores)]



# Function 8 - Score: -0.24809110729574696
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    if not unvisited_nodes.size:
        return depot
    
    # Precision-tuned capacity threshold
    max_demand = demands[unvisited_nodes].max()
    capacity_ratio = rest_capacity / (max_demand + 1e-10)
    threshold = 0.73 + 0.26 / (1 + np.exp(-28 * (capacity_ratio - 0.71)))
    
    # Efficient candidate selection with early termination
    feasible = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible):
        return depot
    
    critical = demands[unvisited_nodes] > (rest_capacity * threshold)
    candidates = unvisited_nodes[feasible | critical]
    
    # Precompute all required features
    dist_current = distance_matrix[current_node, candidates]
    dist_depot = distance_matrix[candidates, depot]
    demand_ratio = demands[candidates] / (rest_capacity + 1e-10)
    
    # Enhanced scaling function with adaptive curvature
    def optimal_scale(x):
        x_min = x.min()
        x_range = x.max() - x_min + 1e-10
        x_norm = (x - x_min) / x_range
        return np.log1p(8.0 * x_norm) / np.log(4.0)
    
    # Feature engineering with improved stability
    proximity = optimal_scale(1 / (dist_current + 1e-8))
    urgency = optimal_scale(demand_ratio * np.exp(4.3 * (1 - capacity_ratio)))
    detour_cost = optimal_scale(1 + np.maximum(0, dist_depot - dist_current) / (dist_current + 1e-8))
    
    # Dynamically adjusted weights with smooth transitions
    capacity_tension = 1 - capacity_ratio
    w_proximity = 0.90 - 0.80 / (1 + np.exp(-26 * (capacity_tension - 0.24)))
    w_urgency = 0.65 / (1 + np.exp(-22 * (capacity_tension - 0.14)))
    w_detour = 0.18 * (1 - np.exp(-6.0 * capacity_tension))
    
    # Smart exploration with adaptive decay
    progress = 1 - candidates.size / unvisited_nodes.size
    exploration = 0.008 * np.exp(-25 * progress) * np.random.randn(candidates.size)
    
    # Comprehensive scoring with precision components
    scores = (
        w_proximity * proximity +
        w_urgency * urgency -
        w_detour * detour_cost +
        exploration
    )
    
    # Critical node handling with adaptive boosting
    is_critical = critical[feasible | critical]
    critical_boost = 1.75 + 0.90 * (1 - np.exp(-5.2 * capacity_tension))
    scores[is_critical] *= critical_boost
    
    return candidates[np.argmax(scores)]



# Function 9 - Score: -0.2481161036203105
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Adaptive capacity threshold with smoother sigmoid transition
    capacity_ratio = rest_capacity / (np.max(demands[unvisited_nodes]) + 1e-10)
    threshold = 0.75 + 0.25 / (1 + np.exp(-25 * (capacity_ratio - 0.8)))  # More gradual transition
    
    # Smart candidate selection with dual feasibility checks
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    critical_mask = demands[unvisited_nodes] > (rest_capacity * threshold)
    candidates = unvisited_nodes[feasible_mask | critical_mask]
    
    if len(candidates) == 0:
        return depot
    
    # Efficient feature computation
    dist_current = distance_matrix[current_node, candidates]
    dist_depot = distance_matrix[candidates, depot]
    demand_ratio = demands[candidates] / (rest_capacity + 1e-10)
    
    # Optimized scaling function with adaptive range handling
    def adaptive_scale(x):
        x_range = np.max(x) - np.min(x) + 1e-10
        x_norm = (x - np.min(x)) / x_range
        return np.log1p(7.0 * x_norm) / np.log(3.8)  # Better curvature for typical ranges
    
    proximity = adaptive_scale(1 / (dist_current + 1e-8))
    urgency = adaptive_scale(demand_ratio * np.exp(4.0 * (1 - capacity_ratio)))
    detour_penalty = adaptive_scale(1 + np.maximum(0, dist_depot - dist_current) / (dist_current + 1e-8))
    
    # Capacity-aware dynamic weighting
    capacity_tension = 1 - capacity_ratio
    w_proximity = 0.85 - 0.75 / (1 + np.exp(-25 * (capacity_tension - 0.3)))  # More balanced
    w_urgency = 0.70 / (1 + np.exp(-20 * (capacity_tension - 0.2)))  # More gradual
    w_detour = 0.20 * (1 - np.exp(-5.0 * capacity_tension))  # Better balanced
    
    # Progress-adaptive exploration with noise scaling
    progress = 1 - len(candidates) / len(unvisited_nodes)
    exploration = 0.015 * np.exp(-20 * progress) * np.random.randn(len(candidates))
    
    # Final scoring with component balancing
    scores = (
        w_proximity * proximity +
        w_urgency * urgency -
        w_detour * detour_penalty +
        exploration
    )
    
    # Smart critical node boosting
    is_critical = critical_mask[feasible_mask | critical_mask]
    critical_boost = 1.7 + 0.9 * (1 - np.exp(-6.0 * capacity_tension))
    scores[is_critical] *= critical_boost
    
    return candidates[np.argmax(scores)]



# Function 10 - Score: -0.24822530536117568
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    if not unvisited_nodes.size:
        return depot
    
    # Adaptive capacity threshold with enhanced sigmoid
    max_demand = demands[unvisited_nodes].max()
    capacity_ratio = rest_capacity / (max_demand + 1e-10)
    threshold = 0.76 + 0.23 / (1 + np.exp(-26 * (capacity_ratio - 0.73)))
    
    # Intelligent candidate filtering
    feasible = demands[unvisited_nodes] <= rest_capacity
    critical = demands[unvisited_nodes] > (rest_capacity * threshold)
    candidates = unvisited_nodes[feasible | critical]
    
    if not candidates.size:
        return depot
    
    # Optimized feature computation
    dist_current = distance_matrix[current_node, candidates]
    dist_depot = distance_matrix[candidates, depot]
    demand_ratio = demands[candidates] / (rest_capacity + 1e-10)
    
    # Advanced scaling with precise coefficients
    def refined_scale(x):
        x_min = x.min()
        x_range = x.max() - x_min + 1e-10
        x_norm = (x - x_min) / x_range
        return np.log1p(7.8 * x_norm) / np.log(4.0)
    
    proximity = refined_scale(1 / (dist_current + 1e-8))
    urgency = refined_scale(demand_ratio * np.exp(4.3 * (1 - capacity_ratio)))
    detour_cost = refined_scale(1 + np.maximum(0, dist_depot - dist_current) / (dist_current + 1e-8))
    
    # Dynamic weight balancing with improved sigmoids
    capacity_tension = 1 - capacity_ratio
    w_proximity = 0.89 - 0.79 / (1 + np.exp(-25 * (capacity_tension - 0.26)))
    w_urgency = 0.64 / (1 + np.exp(-21 * (capacity_tension - 0.16)))
    w_detour = 0.19 * (1 - np.exp(-6.0 * capacity_tension))
    
    # Progress-adaptive exploration with noise tuning
    progress = 1 - candidates.size / unvisited_nodes.size
    exploration = 0.01 * np.exp(-25 * progress) * np.random.randn(candidates.size)
    
    # Final scoring with optimized critical boost
    scores = (
        w_proximity * proximity +
        w_urgency * urgency -
        w_detour * detour_cost +
        exploration
    )
    
    # Precision-tuned critical node handling
    is_critical = critical[feasible | critical]
    critical_boost = 1.78 + 0.88 * (1 - np.exp(-5.2 * capacity_tension))
    scores[is_critical] *= critical_boost
    
    return candidates[np.argmax(scores)]



