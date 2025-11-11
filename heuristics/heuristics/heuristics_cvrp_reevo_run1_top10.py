# Top 10 functions for reevo run 1

# Function 1 - Score: -0.25589456017416023
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
    
    # Adaptive capacity buffer with demand characteristics and route progress
    demand_stats = demands[unvisited_nodes]
    demand_cv = np.std(demand_stats) / (np.mean(demand_stats) + 1e-10)
    route_progress = 1 - len(unvisited_nodes) / len(demands)
    
    # Three-component adaptive buffer
    buffer = (0.02 + 
             0.06 * (1 - np.exp(-3 * (demand_cv - 0.3))) + 
             0.03 * route_progress * (1 + 0.5 * demand_cv))
    feasible_mask = demands[unvisited_nodes] <= rest_capacity * (1 + buffer)
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if not feasible_nodes.size:
        return depot
    
    # Robust distance metrics using percentile normalization
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    dist_p90 = np.percentile(distance_matrix, 90)
    
    # Normalized components with outlier protection
    norm_current = (current_dists - np.min(current_dists)) / (np.ptp(current_dists) + 1e-10)
    norm_depot = (depot_dists - np.min(depot_dists)) / (np.ptp(depot_dists) + 1e-10)
    
    # Route state analysis with multiple factors
    remaining_demand = np.sum(demands[unvisited_nodes])
    capacity_ratio = min(1.0, remaining_demand / (rest_capacity + 1e-10))
    urgency = (len(unvisited_nodes) / len(demands)) ** 0.7
    
    # Core scoring components with enhanced formulations
    proximity = (0.8 / (current_dists + 0.1*dist_p90) + 
                0.2 * (1 - norm_current) * (1 + 0.3 * urgency))
    utilization = np.power(demands[feasible_nodes], 0.8) / (rest_capacity + 1e-10)
    spatial_cohesion = np.exp(-3 * abs(norm_current - (1 - norm_depot)))
    
    # Dynamic weight adaptation using route state
    proximity_weight = 0.7 - 0.2 * (1 / (1 + np.exp(-12 * (capacity_ratio - 0.4))))
    utilization_weight = 0.6 / (1 + np.exp(-10 * (1.2 - capacity_ratio)))
    spatial_weight = 0.4 * (1 - np.exp(-3 * urgency))
    
    # Advanced spatial clustering analysis
    if len(feasible_nodes) > 1:
        centroid = np.mean(distance_matrix[feasible_nodes], axis=0)
        spatial_scores = np.linalg.norm(distance_matrix[feasible_nodes] - centroid, axis=1)
        spatial_scores = (spatial_scores - np.min(spatial_scores)) / (np.ptp(spatial_scores) + 1e-10)
    else:
        spatial_scores = np.zeros(len(feasible_nodes))
    
    # Adaptive critical demand detection
    demand_ratio = demands[feasible_nodes] / rest_capacity
    critical_threshold = (0.5 + 
                        0.3 * np.tanh(5 * (demand_cv - 0.4)) + 
                        0.15 * route_progress)
    critical_bonus = np.where(demand_ratio > critical_threshold,
                            (demand_ratio - critical_threshold) ** 1.5,
                            0)
    
    # Balanced composite scoring
    scores = (proximity_weight * proximity +
             utilization_weight * utilization +
             spatial_weight * spatial_cohesion +
             0.7 * spatial_scores +
             1.5 * critical_bonus)
    
    # Sophisticated tie-breaking mechanism
    best_idx = np.argmax(scores)
    if np.sum(np.isclose(scores, scores[best_idx], rtol=1e-8, atol=1e-8)) > 1:
        tied_nodes = feasible_nodes[np.isclose(scores, scores[best_idx])]
        # Priority: critical bonus -> spatial cohesion -> proximity -> utilization
        tie_breakers = np.column_stack([
            -critical_bonus[np.isclose(scores, scores[best_idx])],
            -spatial_cohesion[np.isclose(scores, scores[best_idx])],
            current_dists[np.isclose(scores, scores[best_idx])],
            -utilization[np.isclose(scores, scores[best_idx])]
        ])
        return tied_nodes[np.lexsort(tie_breakers.T)[0]]
    
    return feasible_nodes[best_idx]



# Function 2 - Score: -0.2584921835653716
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
    
    # Dynamic capacity buffer with demand volatility and route progress
    demand_cv = np.std(demands[unvisited_nodes]) / (np.mean(demands[unvisited_nodes]) + 1e-10)
    completion_ratio = 1 - len(unvisited_nodes) / len(demands)
    buffer = 0.04 + 0.08 * (1 - np.exp(-4 * (demand_cv - 0.2))) * (1 + 0.15 * completion_ratio)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity * (1 + buffer)
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if not feasible_nodes.size:
        return depot
    
    # Enhanced distance metrics with robust normalization
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    dist_iqr = np.percentile(distance_matrix, 75) - np.percentile(distance_matrix, 25)
    
    # Dual-phase normalization with outlier protection
    norm_current = (current_dists - np.min(current_dists)) / (np.ptp(current_dists) + 1e-10)
    norm_depot = (depot_dists - np.min(depot_dists)) / (np.ptp(depot_dists) + 1e-10)
    
    # Route state analysis with multiple indicators
    remaining_demand = np.sum(demands[unvisited_nodes])
    capacity_utilization = min(1.0, remaining_demand / (rest_capacity + 1e-10))
    urgency_factor = (len(unvisited_nodes) / len(demands)) ** 0.7
    
    # Core components with adaptive weights
    proximity = 0.6/(current_dists + 0.3*dist_iqr) + 0.4*(1 - norm_current)
    utilization = np.power(demands[feasible_nodes], 0.8) / (rest_capacity + 1e-10)
    spatial_balance = np.exp(-abs(norm_current - norm_depot))
    
    # Dynamic weight adaptation using sigmoid functions
    proximity_weight = 0.7 - 0.3 / (1 + np.exp(-8 * (capacity_utilization - 0.5)))
    utilization_weight = 0.5 / (1 + np.exp(-6 * (1 - capacity_utilization)))
    spatial_weight = 0.4 * urgency_factor
    
    # Advanced spatial clustering analysis
    if len(feasible_nodes) > 2:
        centroid = np.mean(distance_matrix[feasible_nodes], axis=0)
        spatial_scores = np.linalg.norm(distance_matrix[feasible_nodes] - centroid, axis=1)
        spatial_scores = (spatial_scores - np.min(spatial_scores)) / (np.ptp(spatial_scores) + 1e-10)
    else:
        spatial_scores = 0
    
    # Critical demand identification with adaptive threshold
    demand_percentile = demands[feasible_nodes] / rest_capacity
    critical_threshold = 0.6 + 0.2 * np.tanh(3 * (demand_cv - 0.4))
    critical_bonus = np.where(demand_percentile > critical_threshold, 
                            0.5 * (demand_percentile - critical_threshold), 
                            0)
    
    # Composite scoring with balanced components
    scores = (proximity_weight * proximity +
              utilization_weight * utilization +
              spatial_weight * spatial_balance +
              0.6 * spatial_scores +
              critical_bonus)
    
    # Enhanced tie-breaking with multiple fallback criteria
    best_idx = np.argmax(scores)
    if np.sum(np.isclose(scores, scores[best_idx], rtol=1e-8, atol=1e-8)) > 1:
        tied_nodes = feasible_nodes[np.isclose(scores, scores[best_idx])]
        # Priority: critical nodes -> spatial balance -> utilization -> proximity
        tie_breakers = np.column_stack([
            -critical_bonus[np.isclose(scores, scores[best_idx])],
            -spatial_balance[np.isclose(scores, scores[best_idx])],
            -utilization[np.isclose(scores, scores[best_idx])],
            current_dists[np.isclose(scores, scores[best_idx])]
        ])
        return tied_nodes[np.lexsort(tie_breakers.T)[0]]
    
    return feasible_nodes[best_idx]



# Function 3 - Score: -0.2592973271059549
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
    
    # Simplified capacity buffer calculation
    demand_cv = np.std(demands[unvisited_nodes]) / (np.mean(demands[unvisited_nodes]) + 1e-10)
    completion_ratio = len(unvisited_nodes) / len(demands)
    
    # Optimized buffer with smoother transitions
    buffer = (0.06 + 0.12 * (1 - np.exp(-5 * (demand_cv - 0.1)))) * (1 + 0.2 * np.exp(-8 * completion_ratio))
    feasible_mask = demands[unvisited_nodes] <= rest_capacity * (1 + buffer)
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if not feasible_nodes.size:
        return depot
    
    # Efficient distance metrics
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    dist_iqr = np.percentile(distance_matrix, 75) - np.percentile(distance_matrix, 25) + 1e-10
    
    # Balanced distance normalization
    norm_current = 1 - np.exp(-current_dists / (0.5 * dist_iqr))
    norm_depot = 1 - np.exp(-depot_dists / (0.5 * dist_iqr))
    
    # Route state analysis with tuned parameters
    capacity_ratio = min(1.0, np.sum(demands[unvisited_nodes]) / (rest_capacity + 1e-10))
    urgency_factor = np.exp(-3 * completion_ratio)
    
    # Core scoring components
    proximity = 0.7 / (current_dists + 0.3 * dist_iqr) + 0.3 * (1 - norm_current)
    utilization = np.power(demands[feasible_nodes], 0.7) / (rest_capacity + 1e-10)
    spatial_balance = 1.2 - abs(norm_current - (1 - norm_depot))
    
    # Optimized dynamic weights
    proximity_weight = 0.65 - 0.25 / (1 + np.exp(-10 * (capacity_ratio - 0.3)))
    utilization_weight = 0.4 / (1 + np.exp(-8 * (1 - capacity_ratio)))
    spatial_weight = 0.3 * urgency_factor
    
    # Efficient spatial clustering
    if len(feasible_nodes) > 1:
        centroid = np.mean(distance_matrix[feasible_nodes], axis=0)
        spatial_scores = np.linalg.norm(distance_matrix[feasible_nodes] - centroid, axis=1)
        spatial_scores = (spatial_scores - np.min(spatial_scores)) / (np.ptp(spatial_scores) + 1e-10)
    else:
        spatial_scores = 0
    
    # Dynamic critical demand detection
    demand_ratio = demands[feasible_nodes] / rest_capacity
    critical_threshold = 0.5 + 0.3 * (1 - np.exp(-6 * (demand_cv - 0.15)))
    critical_bonus = np.where(demand_ratio > critical_threshold,
                            0.8 * np.power(demand_ratio - critical_threshold, 1.0) * urgency_factor,
                            0)
    
    # Final scoring with balanced weights
    scores = (proximity_weight * proximity +
              utilization_weight * utilization +
              spatial_weight * spatial_balance +
              0.5 * spatial_scores +
              critical_bonus)
    
    # Simplified tie-breaking
    best_idx = np.argmax(scores)
    if np.sum(np.isclose(scores, scores[best_idx], rtol=1e-8, atol=1e-8)) > 1:
        tied_nodes = feasible_nodes[np.isclose(scores, scores[best_idx])]
        return tied_nodes[np.argmin(distance_matrix[current_node, tied_nodes])]
    
    return feasible_nodes[best_idx]



# Function 4 - Score: -0.2604516005015519
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
    
    # Adaptive capacity buffer with demand volatility and route progress awareness
    demand_stats = demands[unvisited_nodes]
    demand_mean = np.mean(demand_stats)
    demand_cv = np.std(demand_stats) / (demand_mean + 1e-10)
    completion_ratio = 1 - len(unvisited_nodes) / len(demands)
    
    # Three-level buffer adjustment: base + volatility + progress
    buffer = (0.03 + 
              0.05 * (1 - np.exp(-5 * (demand_cv - 0.25))) + 
              0.02 * completion_ratio * (1 + demand_cv))
    feasible_mask = demands[unvisited_nodes] <= rest_capacity * (1 + buffer)
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if not feasible_nodes.size:
        return depot
    
    # Robust distance metrics using IQR and min-max scaling
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    dist_iqr = np.percentile(distance_matrix, 75) - np.percentile(distance_matrix, 25)
    
    # Dual normalization with outlier protection
    norm_current = (current_dists - np.min(current_dists)) / (np.ptp(current_dists) + 1e-10)
    norm_depot = (depot_dists - np.min(depot_dists)) / (np.ptp(depot_dists) + 1e-10)
    
    # Route state analysis with multiple dimensions
    remaining_demand = np.sum(demands[unvisited_nodes])
    capacity_utilization = min(1.0, remaining_demand / (rest_capacity + 1e-10))
    urgency_factor = (len(unvisited_nodes) / len(demands)) ** 0.8
    
    # Core scoring components with enhanced formulations
    proximity = (0.7 / (current_dists + 0.2*dist_iqr) + 
                0.3 * (1 - norm_current) * (1 + 0.2 * urgency_factor))
    utilization = np.power(demands[feasible_nodes], 0.9) / (rest_capacity + 1e-10)
    spatial_balance = np.exp(-2 * abs(norm_current - (1 - norm_depot)))
    
    # Dynamic weight adaptation using advanced sigmoid functions
    proximity_weight = 0.75 - 0.25 / (1 + np.exp(-10 * (capacity_utilization - 0.45)))
    utilization_weight = 0.55 / (1 + np.exp(-8 * (1.1 - capacity_utilization)))
    spatial_weight = 0.35 * (urgency_factor + 0.2 * demand_cv)
    
    # Spatial clustering analysis with density awareness
    if len(feasible_nodes) > 2:
        centroid = np.mean(distance_matrix[feasible_nodes], axis=0)
        spatial_scores = np.linalg.norm(distance_matrix[feasible_nodes] - centroid, axis=1)
        spatial_scores = (spatial_scores - np.min(spatial_scores)) / (np.ptp(spatial_scores) + 1e-10)
    else:
        spatial_scores = np.zeros(len(feasible_nodes))
    
    # Critical demand detection with adaptive threshold and gradient bonus
    demand_ratio = demands[feasible_nodes] / rest_capacity
    critical_threshold = (0.55 + 
                        0.25 * np.tanh(4 * (demand_cv - 0.35)) + 
                        0.1 * completion_ratio)
    critical_bonus = np.where(demand_ratio > critical_threshold,
                            (demand_ratio - critical_threshold) ** 1.2,
                            0)
    
    # Composite scoring with balanced components
    scores = (proximity_weight * proximity +
             utilization_weight * utilization +
             spatial_weight * spatial_balance +
             0.65 * spatial_scores +
             1.2 * critical_bonus)
    
    # Enhanced tie-breaking with prioritized criteria
    best_idx = np.argmax(scores)
    if np.sum(np.isclose(scores, scores[best_idx], rtol=1e-8, atol=1e-8)) > 1:
        tied_nodes = feasible_nodes[np.isclose(scores, scores[best_idx])]
        # Priority: critical bonus -> spatial balance -> proximity -> utilization
        tie_breakers = np.column_stack([
            -critical_bonus[np.isclose(scores, scores[best_idx])],
            -spatial_balance[np.isclose(scores, scores[best_idx])],
            current_dists[np.isclose(scores, scores[best_idx])],
            -utilization[np.isclose(scores, scores[best_idx])]
        ])
        return tied_nodes[np.lexsort(tie_breakers.T)[0]]
    
    return feasible_nodes[best_idx]



# Function 5 - Score: -0.26085203928989625
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
    
    # Enhanced capacity buffer with dynamic adjustment
    demand_stats = demands[unvisited_nodes]
    demand_mean = np.mean(demand_stats)
    demand_cv = np.std(demand_stats) / (demand_mean + 1e-10)
    completion_ratio = 1 - len(unvisited_nodes) / len(demands)
    
    # Three-component adaptive buffer with improved coefficients
    buffer = (0.03 + 
             0.05 * (1 - np.exp(-8 * (demand_cv - 0.15))) + 
             0.04 * completion_ratio * (1 + 0.4*demand_cv))
    feasible_mask = demands[unvisited_nodes] <= rest_capacity * (1 + buffer)
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if not feasible_nodes.size:
        return depot
    
    # Robust distance metrics with adaptive normalization
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    
    # Median-based normalization with outlier protection
    def robust_normalize(x):
        median = np.median(x)
        mad = np.median(np.abs(x - median)) + 1e-10
        return (x - median) / mad
    
    norm_current = robust_normalize(current_dists)
    norm_depot = robust_normalize(depot_dists)
    
    # Advanced route state analysis
    remaining_demand = np.sum(demands[unvisited_nodes])
    capacity_ratio = min(1.0, remaining_demand / (rest_capacity + 1e-10))
    urgency = (len(unvisited_nodes) / len(demands)) ** 0.7
    
    # Optimized scoring components
    proximity = (0.8 / (current_dists + 0.1*np.percentile(current_dists, 80)) + 
                0.2 * (1 - norm_current) * (1 + 0.1 * urgency))
    utilization = np.power(demands[feasible_nodes], 0.9) / (rest_capacity + 1e-10)
    spatial_balance = np.exp(-3.0 * abs(norm_current - (1 - norm_depot)))
    
    # Dynamic weight adaptation with smooth transitions
    proximity_weight = 0.7 - 0.25 / (1 + np.exp(-10 * (capacity_ratio - 0.45)))
    utilization_weight = 0.65 / (1 + np.exp(-8 * (1.1 - capacity_ratio)))
    spatial_weight = 0.35 * (1 - np.exp(-5 * urgency))
    
    # Density-aware spatial clustering
    if len(feasible_nodes) > 3:
        centroid = np.median(distance_matrix[feasible_nodes], axis=0)
        spatial_scores = np.linalg.norm(distance_matrix[feasible_nodes] - centroid, axis=1)
        spatial_scores = robust_normalize(spatial_scores)
    else:
        spatial_scores = np.zeros(len(feasible_nodes))
    
    # Adaptive critical demand detection with progressive bonus
    demand_ratio = demands[feasible_nodes] / rest_capacity
    critical_threshold = (0.55 + 
                        0.25 * np.tanh(7 * (demand_cv - 0.25)) + 
                        0.1 * completion_ratio)
    critical_bonus = np.where(demand_ratio > critical_threshold,
                            (demand_ratio - critical_threshold) ** 1.4,
                            0)
    
    # Balanced scoring with optimized component weights
    scores = (proximity_weight * proximity +
             utilization_weight * utilization +
             spatial_weight * (spatial_balance + 0.5 * spatial_scores) +
             1.4 * critical_bonus)
    
    # Enhanced hierarchical tie-breaking with clear priorities
    best_idx = np.argmax(scores)
    if np.sum(np.isclose(scores, scores[best_idx], rtol=1e-8, atol=1e-8)) > 1:
        tied_nodes = feasible_nodes[np.isclose(scores, scores[best_idx])]
        tie_breakers = np.column_stack([
            -critical_bonus[np.isclose(scores, scores[best_idx])],
            -spatial_balance[np.isclose(scores, scores[best_idx])],
            current_dists[np.isclose(scores, scores[best_idx])],
            -utilization[np.isclose(scores, scores[best_idx])]
        ])
        return tied_nodes[np.lexsort(tie_breakers.T)[0]]
    
    return feasible_nodes[best_idx]



# Function 6 - Score: -0.26153339625262184
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
    
    # Multi-stage capacity buffer with demand statistics and route progress
    demand_stats = demands[unvisited_nodes]
    demand_mean = np.mean(demand_stats)
    demand_cv = np.std(demand_stats) / (demand_mean + 1e-10)
    completion_ratio = 1 - len(unvisited_nodes) / len(demands)
    
    # Dynamic buffer with three components: base, volatility, and progress
    buffer = (0.02 + 
              0.06 * (1 - np.exp(-6 * (demand_cv - 0.2))) + 
              0.03 * completion_ratio * (1 + 0.5*demand_cv))
    feasible_mask = demands[unvisited_nodes] <= rest_capacity * (1 + buffer)
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if not feasible_nodes.size:
        return depot
    
    # Enhanced distance metrics with robust normalization
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    dist_iqr = np.percentile(distance_matrix, 75) - np.percentile(distance_matrix, 25)
    
    # Adaptive normalization with outlier protection
    norm_current = (current_dists - np.min(current_dists)) / (np.ptp(current_dists) + 1e-10)
    norm_depot = (depot_dists - np.min(depot_dists)) / (np.ptp(depot_dists) + 1e-10)
    
    # Route state analysis with multiple dimensions
    remaining_demand = np.sum(demands[unvisited_nodes])
    capacity_utilization = min(1.0, remaining_demand / (rest_capacity + 1e-10))
    urgency_factor = (len(unvisited_nodes) / len(demands)) ** 0.75
    
    # Core scoring components with improved formulations
    proximity = (0.75 / (current_dists + 0.15*dist_iqr) + 
                0.25 * (1 - norm_current) * (1 + 0.15 * urgency_factor))
    utilization = np.power(demands[feasible_nodes], 0.95) / (rest_capacity + 1e-10)
    spatial_balance = np.exp(-2.5 * abs(norm_current - (1 - norm_depot)))
    
    # Dynamic weight adaptation using advanced sigmoid functions
    proximity_weight = 0.7 - 0.3 / (1 + np.exp(-12 * (capacity_utilization - 0.4)))
    utilization_weight = 0.6 / (1 + np.exp(-10 * (1.05 - capacity_utilization)))
    spatial_weight = 0.4 * (urgency_factor + 0.25 * demand_cv)
    
    # Spatial clustering with density awareness and outlier protection
    if len(feasible_nodes) > 2:
        centroid = np.mean(distance_matrix[feasible_nodes], axis=0)
        spatial_scores = np.linalg.norm(distance_matrix[feasible_nodes] - centroid, axis=1)
        spatial_scores = (spatial_scores - np.min(spatial_scores)) / (np.ptp(spatial_scores) + 1e-10)
    else:
        spatial_scores = np.zeros(len(feasible_nodes))
    
    # Critical demand detection with adaptive threshold and progressive bonus
    demand_ratio = demands[feasible_nodes] / rest_capacity
    critical_threshold = (0.5 + 
                        0.3 * np.tanh(5 * (demand_cv - 0.3)) + 
                        0.15 * completion_ratio)
    critical_bonus = np.where(demand_ratio > critical_threshold,
                            (demand_ratio - critical_threshold) ** 1.5,
                            0)
    
    # Optimized composite scoring with balanced components
    scores = (proximity_weight * proximity +
             utilization_weight * utilization +
             spatial_weight * spatial_balance +
             0.7 * spatial_scores +
             1.3 * critical_bonus)
    
    # Enhanced hierarchical tie-breaking mechanism
    best_idx = np.argmax(scores)
    if np.sum(np.isclose(scores, scores[best_idx], rtol=1e-8, atol=1e-8)) > 1:
        tied_nodes = feasible_nodes[np.isclose(scores, scores[best_idx])]
        # Priority: critical bonus -> spatial balance -> proximity -> utilization
        tie_breakers = np.column_stack([
            -critical_bonus[np.isclose(scores, scores[best_idx])],
            -spatial_balance[np.isclose(scores, scores[best_idx])],
            current_dists[np.isclose(scores, scores[best_idx])],
            -utilization[np.isclose(scores, scores[best_idx])]
        ])
        return tied_nodes[np.lexsort(tie_breakers.T)[0]]
    
    return feasible_nodes[best_idx]



# Function 7 - Score: -0.26319604666305646
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
    
    # Phase-adaptive capacity buffer
    demand_stats = demands[unvisited_nodes]
    demand_cv = np.std(demand_stats) / (np.mean(demand_stats) + 1e-10)
    completion_ratio = len(unvisited_nodes) / len(demands)
    
    # Dynamic buffer with smooth phase transitions
    base_buffer = 0.05 + 0.15 * (1 - np.exp(-6 * (demand_cv - 0.15)))
    phase_factor = 0.25 * (1 - np.exp(-10 * (1 - completion_ratio)))
    buffer = base_buffer * (1 + phase_factor)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity * (1 + buffer)
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if not feasible_nodes.size:
        return depot
    
    # IQR-based distance normalization
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    dist_q1, dist_q3 = np.percentile(distance_matrix, [25, 75])
    dist_iqr = dist_q3 - dist_q1 + 1e-10
    
    norm_current = 1 - np.exp(-current_dists / (0.6 * dist_iqr))
    norm_depot = 1 - np.exp(-depot_dists / (0.6 * dist_iqr))
    
    # Route state analysis
    remaining_demand = np.sum(demands[unvisited_nodes])
    capacity_ratio = min(1.0, remaining_demand / (rest_capacity + 1e-10))
    urgency_factor = np.exp(-4 * completion_ratio)
    
    # Core scoring components
    proximity = 0.8 / (current_dists + 0.2 * dist_iqr) + 0.2 * (1 - norm_current)
    utilization = np.power(demands[feasible_nodes], 0.8) / (rest_capacity + 1e-10)
    spatial_balance = 1.3 - abs(norm_current - (1 - norm_depot))**1.5
    
    # Dynamic weights with smooth transitions
    proximity_weight = 0.7 - 0.3 / (1 + np.exp(-12 * (capacity_ratio - 0.4)))
    utilization_weight = 0.5 / (1 + np.exp(-10 * (1 - capacity_ratio)))
    spatial_weight = 0.35 * urgency_factor
    
    # Simplified spatial clustering
    if len(feasible_nodes) > 1:
        centroid = np.mean(distance_matrix[feasible_nodes], axis=0)
        spatial_scores = np.linalg.norm(distance_matrix[feasible_nodes] - centroid, axis=1)
        spatial_scores = (spatial_scores - np.min(spatial_scores)) / (np.ptp(spatial_scores) + 1e-10)
    else:
        spatial_scores = 0
    
    # Dynamic critical demand detection
    demand_ratio = demands[feasible_nodes] / rest_capacity
    critical_threshold = 0.6 + 0.3 * (1 - np.exp(-7 * (demand_cv - 0.2)))
    critical_bonus = np.where(demand_ratio > critical_threshold,
                            0.9 * np.power(demand_ratio - critical_threshold, 1.2) * urgency_factor,
                            0)
    
    # Optimized composite scoring
    scores = (proximity_weight * proximity +
              utilization_weight * utilization +
              spatial_weight * spatial_balance +
              0.6 * spatial_scores +
              critical_bonus)
    
    # Enhanced tie-breaking
    best_idx = np.argmax(scores)
    if np.sum(np.isclose(scores, scores[best_idx], rtol=1e-8, atol=1e-8)) > 1:
        tied_nodes = feasible_nodes[np.isclose(scores, scores[best_idx])]
        tie_breakers = np.column_stack([
            -critical_bonus[np.isclose(scores, scores[best_idx])],
            -spatial_balance[np.isclose(scores, scores[best_idx])],
            -utilization[np.isclose(scores, scores[best_idx])],
            current_dists[np.isclose(scores, scores[best_idx])]
        ])
        return tied_nodes[np.lexsort(tie_breakers.T)[0]]
    
    return feasible_nodes[best_idx]



# Function 8 - Score: -0.2637319347025865
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
    
    # Phase-aware capacity buffer with demand volatility
    demand_stats = demands[unvisited_nodes]
    demand_mean = np.mean(demand_stats)
    demand_cv = np.std(demand_stats) / (demand_mean + 1e-10)
    completion_ratio = 1 - len(unvisited_nodes) / len(demands)
    
    # Dynamic buffer with phase adaptation
    base_buffer = 0.05 + 0.1 * (1 - np.exp(-5 * (demand_cv - 0.2)))
    phase_factor = 0.2 * (1 + np.tanh(8 * (completion_ratio - 0.6)))
    buffer = base_buffer * (1 + phase_factor)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity * (1 + buffer)
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if not feasible_nodes.size:
        return depot
    
    # Robust distance metrics with phase-dependent scaling
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    dist_q1, dist_q3 = np.percentile(distance_matrix, [25, 75])
    dist_iqr = dist_q3 - dist_q1
    
    # Dual adaptive normalization
    norm_current = 1 - np.exp(-current_dists / (0.7 * dist_iqr + 1e-10))
    norm_depot = 1 - np.exp(-depot_dists / (0.7 * dist_iqr + 1e-10))
    
    # Route state analysis with multiple factors
    remaining_demand = np.sum(demands[unvisited_nodes])
    capacity_ratio = min(1.0, remaining_demand / (rest_capacity + 1e-10))
    urgency_factor = np.power(len(unvisited_nodes) / len(demands), 0.8)
    
    # Core scoring components with phase adaptation
    proximity = 0.85/(current_dists + 0.25*dist_iqr) + 0.15*(1 - norm_current)
    utilization = np.power(demands[feasible_nodes], 0.85) / (rest_capacity + 1e-10)
    spatial_balance = 1.2 - abs(norm_current - (1 - norm_depot))**2.0
    
    # Dynamic weights with smooth phase transitions
    proximity_weight = 0.65 - 0.25 / (1 + np.exp(-15 * (capacity_ratio - 0.3)))
    utilization_weight = 0.55 / (1 + np.exp(-12 * (1 - capacity_ratio)))
    spatial_weight = 0.4 * (1 - np.exp(-5 * urgency_factor))
    
    # Density-aware spatial clustering
    if len(feasible_nodes) > 1:
        centroid = np.mean(distance_matrix[feasible_nodes], axis=0)
        spatial_scores = np.linalg.norm(distance_matrix[feasible_nodes] - centroid, axis=1)
        spatial_scores = (spatial_scores - np.min(spatial_scores)) / (np.ptp(spatial_scores) + 1e-10)
    else:
        spatial_scores = 0
    
    # Phase-sensitive critical demand detection
    demand_ratio = demands[feasible_nodes] / rest_capacity
    critical_threshold = 0.55 + 0.35 * (1 - np.exp(-8 * (demand_cv - 0.25)))
    critical_bonus = np.where(demand_ratio > critical_threshold,
                            0.8 * np.power(demand_ratio - critical_threshold, 1.3),
                            0)
    
    # Optimized composite scoring with phase emphasis
    scores = (proximity_weight * proximity +
              utilization_weight * utilization +
              spatial_weight * spatial_balance +
              0.7 * spatial_scores +
              critical_bonus)
    
    # Enhanced hierarchical tie-breaking
    best_idx = np.argmax(scores)
    if np.sum(np.isclose(scores, scores[best_idx], rtol=1e-8, atol=1e-8)) > 1:
        tied_nodes = feasible_nodes[np.isclose(scores, scores[best_idx])]
        # Priority: critical bonus -> spatial balance -> utilization -> proximity
        tie_breakers = np.column_stack([
            -critical_bonus[np.isclose(scores, scores[best_idx])],
            -spatial_balance[np.isclose(scores, scores[best_idx])],
            -utilization[np.isclose(scores, scores[best_idx])],
            current_dists[np.isclose(scores, scores[best_idx])]
        ])
        return tied_nodes[np.lexsort(tie_breakers.T)[0]]
    
    return feasible_nodes[best_idx]



# Function 9 - Score: -0.26377207721943624
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
    
    # Adaptive feasibility buffer with three components
    demand_mean = np.mean(demands[unvisited_nodes])
    demand_cv = np.std(demands[unvisited_nodes]) / (demand_mean + 1e-10)
    progress = 1 - len(unvisited_nodes) / len(demands)
    
    buffer = (0.04 + 
             0.08 * (1 - np.exp(-4 * (demand_cv - 0.25))) + 
             0.04 * progress * (1 + 0.3 * demand_cv))
    feasible_mask = demands[unvisited_nodes] <= rest_capacity * (1 + buffer)
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if not feasible_nodes.size:
        return depot
    
    # Robust distance metrics using interquartile range normalization
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    
    def iqr_normalize(x):
        q75, q25 = np.percentile(x, [75, 25])
        return (x - np.median(x)) / (q75 - q25 + 1e-10)
    
    norm_current = iqr_normalize(current_dists)
    norm_depot = iqr_normalize(depot_dists)
    
    # Route state analysis
    remaining_demand = np.sum(demands[unvisited_nodes])
    capacity_ratio = min(1.0, remaining_demand / (rest_capacity + 1e-10))
    urgency = (len(unvisited_nodes) / len(demands)) ** 0.8
    
    # Core scoring components
    proximity = (0.75 / (current_dists + 0.15*np.percentile(current_dists, 75)) + 
                0.25 * (1 - norm_current) * (1 + 0.2 * urgency))
    utilization = np.power(demands[feasible_nodes], 0.85) / (rest_capacity + 1e-10)
    spatial_balance = np.exp(-2.5 * abs(norm_current - (1 - norm_depot)))
    
    # Dynamic weight adaptation
    proximity_weight = 0.65 - 0.25 / (1 + np.exp(-15 * (capacity_ratio - 0.5)))
    utilization_weight = 0.55 / (1 + np.exp(-12 * (1.1 - capacity_ratio)))
    spatial_weight = 0.35 * (1 - np.exp(-4 * urgency))
    
    # Efficient spatial analysis using centroid
    if len(feasible_nodes) > 1:
        centroid = np.mean(distance_matrix[feasible_nodes], axis=0)
        spatial_scores = np.linalg.norm(distance_matrix[feasible_nodes] - centroid, axis=1)
        spatial_scores = iqr_normalize(spatial_scores)
    else:
        spatial_scores = np.zeros(len(feasible_nodes))
    
    # Adaptive critical demand detection
    demand_ratio = demands[feasible_nodes] / rest_capacity
    critical_threshold = (0.55 + 
                        0.25 * np.tanh(6 * (demand_cv - 0.35)) + 
                        0.1 * progress)
    critical_bonus = np.where(demand_ratio > critical_threshold,
                            (demand_ratio - critical_threshold) ** 1.3,
                            0)
    
    # Balanced composite scoring
    scores = (proximity_weight * proximity +
             utilization_weight * utilization +
             spatial_weight * (spatial_balance + 0.6 * spatial_scores) +
             1.3 * critical_bonus)
    
    # Efficient tie-breaking with clear priorities
    best_idx = np.argmax(scores)
    if np.sum(np.isclose(scores, scores[best_idx], rtol=1e-8, atol=1e-8)) > 1:
        tied_nodes = feasible_nodes[np.isclose(scores, scores[best_idx])]
        tie_breakers = np.column_stack([
            -critical_bonus[np.isclose(scores, scores[best_idx])],
            -spatial_balance[np.isclose(scores, scores[best_idx])],
            current_dists[np.isclose(scores, scores[best_idx])],
            -utilization[np.isclose(scores, scores[best_idx])]
        ])
        return tied_nodes[np.lexsort(tie_breakers.T)[0]]
    
    return feasible_nodes[best_idx]



# Function 10 - Score: -0.2641063639462515
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
    
    # Phase-adaptive capacity buffer with demand volatility
    demand_stats = demands[unvisited_nodes]
    demand_mean = np.mean(demand_stats)
    demand_cv = np.std(demand_stats) / (demand_mean + 1e-10)
    completion_ratio = 1 - len(unvisited_nodes) / len(demands)
    
    # Dynamic buffer with smooth phase transition
    base_buffer = 0.04 + 0.12 * (1 - np.exp(-6 * (demand_cv - 0.15)))
    phase_factor = 0.25 * (1 + np.tanh(10 * (completion_ratio - 0.55)))
    buffer = base_buffer * (1 + phase_factor)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity * (1 + buffer)
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if not feasible_nodes.size:
        return depot
    
    # Robust distance metrics with IQR scaling
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    dist_q1, dist_q3 = np.percentile(distance_matrix, [25, 75])
    dist_iqr = dist_q3 - dist_q1
    
    # Enhanced exponential normalization
    norm_current = 1 - np.exp(-current_dists / (0.6 * dist_iqr + 1e-10))
    norm_depot = 1 - np.exp(-depot_dists / (0.6 * dist_iqr + 1e-10))
    
    # Route state analysis with smooth phase transitions
    remaining_demand = np.sum(demands[unvisited_nodes])
    capacity_ratio = min(1.0, remaining_demand / (rest_capacity + 1e-10))
    urgency_factor = np.power(len(unvisited_nodes) / len(demands), 0.9)
    
    # Core components with phase adaptation
    proximity = 0.9/(current_dists + 0.3*dist_iqr) + 0.1*(1 - norm_current)
    utilization = np.power(demands[feasible_nodes], 0.9) / (rest_capacity + 1e-10)
    spatial_balance = 1.25 - abs(norm_current - (1 - norm_depot))**2.2
    
    # Dynamic weights with enhanced phase sensitivity
    proximity_weight = 0.7 - 0.3 / (1 + np.exp(-18 * (capacity_ratio - 0.25)))
    utilization_weight = 0.6 / (1 + np.exp(-15 * (1 - capacity_ratio)))
    spatial_weight = 0.45 * (1 - np.exp(-6 * urgency_factor))
    
    # Density-aware spatial clustering
    if len(feasible_nodes) > 1:
        centroid = np.mean(distance_matrix[feasible_nodes], axis=0)
        spatial_scores = np.linalg.norm(distance_matrix[feasible_nodes] - centroid, axis=1)
        spatial_scores = (spatial_scores - np.min(spatial_scores)) / (np.ptp(spatial_scores) + 1e-10)
    else:
        spatial_scores = 0
    
    # Phase-critical demand detection
    demand_ratio = demands[feasible_nodes] / rest_capacity
    critical_threshold = 0.5 + 0.4 * (1 - np.exp(-10 * (demand_cv - 0.2)))
    critical_bonus = np.where(demand_ratio > critical_threshold,
                            0.9 * np.power(demand_ratio - critical_threshold, 1.4),
                            0)
    
    # Optimized scoring with enhanced phase emphasis
    scores = (proximity_weight * proximity +
              utilization_weight * utilization +
              spatial_weight * spatial_balance +
              0.75 * spatial_scores +
              critical_bonus)
    
    # Hierarchical tie-breaking with critical node priority
    best_idx = np.argmax(scores)
    if np.sum(np.isclose(scores, scores[best_idx], rtol=1e-8, atol=1e-8)) > 1:
        tied_nodes = feasible_nodes[np.isclose(scores, scores[best_idx])]
        tie_breakers = np.column_stack([
            -critical_bonus[np.isclose(scores, scores[best_idx])],
            -spatial_balance[np.isclose(scores, scores[best_idx])],
            -utilization[np.isclose(scores, scores[best_idx])],
            current_dists[np.isclose(scores, scores[best_idx])]
        ])
        return tied_nodes[np.lexsort(tie_breakers.T)[0]]
    
    return feasible_nodes[best_idx]



