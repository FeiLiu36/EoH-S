# Top 10 functions for eohs run 2

# Function 1 - Score: -0.26095603282372815
{The new algorithm selects the next node by considering feasibility, a dynamic weighted score combining proximity, demand utilization, remaining capacity ratio with adaptive penalties, a temporal urgency factor based on time window constraints (simulated via distance from depot), and a novel spatial dispersion score that prioritizes nodes in less dense regions while considering route continuity, with a fallback to the depot if no feasible node exists.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    utilizations = demands[feasible_nodes] / rest_capacity
    remaining_capacity_ratio = (rest_capacity - demands[feasible_nodes]) / rest_capacity
    proximity_score = 1 - distances / np.max(distances)
    capacity_penalty = np.where(remaining_capacity_ratio < 0.3, 0.6, np.where(remaining_capacity_ratio < 0.5, 0.8, 1.0))
    urgency_score = distance_matrix[depot, feasible_nodes] / np.max(distance_matrix[depot, :])
    
    spatial_dispersion = np.array([np.median(distance_matrix[n, unvisited_nodes]) for n in feasible_nodes])
    dispersion_score = spatial_dispersion / np.max(spatial_dispersion)
    
    route_continuity = np.array([distance_matrix[current_node, n] / np.max(distances) for n in feasible_nodes])
    continuity_score = 1 - route_continuity
    
    scores = (0.35 * proximity_score + 
              0.2 * utilizations + 
              0.15 * remaining_capacity_ratio * capacity_penalty + 
              0.15 * urgency_score + 
              0.1 * dispersion_score + 
              0.05 * continuity_score)
    
    return feasible_nodes[np.argmax(scores)]



# Function 2 - Score: -0.2627846007214498
{The novel algorithm selects the next node by considering feasibility, a dynamic weighted score combining proximity, demand utilization, remaining capacity ratio with adaptive penalties, a time-window-like urgency score based on distance from the depot, a novel hybrid exploration factor combining cluster density and spatial dispersion, and a route progress-based adaptive weight adjustment with a non-linear transition from exploration to exploitation, with a fallback to the depot if no feasible node exists.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    utilizations = demands[feasible_nodes] / rest_capacity
    remaining_capacity_ratio = (rest_capacity - demands[feasible_nodes]) / rest_capacity
    proximity_score = 1 - distances / np.max(distances)
    capacity_penalty = np.where(remaining_capacity_ratio < 0.2, 0.5, np.where(remaining_capacity_ratio < 0.5, 0.8, 1.0))
    urgency_score = distance_matrix[depot, feasible_nodes] / np.max(distance_matrix[depot, :])
    
    cluster_density = np.array([np.mean(distance_matrix[n, unvisited_nodes]) for n in feasible_nodes])
    dispersion_score = np.array([np.std(distance_matrix[n, unvisited_nodes]) for n in feasible_nodes])
    hybrid_exploration = 0.6 * (cluster_density / np.max(cluster_density)) + 0.4 * (dispersion_score / np.max(dispersion_score))
    
    route_progress = len(unvisited_nodes) / len(distance_matrix)
    exploration_weight = 0.5 * (1 - route_progress**2)
    exploitation_weight = 1 - exploration_weight
    
    scores = (0.4 * exploitation_weight * proximity_score + 
              0.2 * exploitation_weight * utilizations + 
              0.15 * exploitation_weight * remaining_capacity_ratio * capacity_penalty + 
              0.1 * exploitation_weight * urgency_score + 
              0.15 * exploration_weight * hybrid_exploration)
    
    return feasible_nodes[np.argmax(scores)]



# Function 3 - Score: -0.2629758540981269
{The novel algorithm selects the next node by considering feasibility, a dynamic weighted score combining proximity, demand utilization, remaining capacity ratio with adaptive penalties, a time-window-like urgency score based on distance from the depot, a novel exploration factor combining cluster density and route diversity with adaptive weights, and a route progress-based adaptive weight adjustment with a non-linear transition from exploration to exploitation, with a fallback to the depot if no feasible node exists.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    utilizations = demands[feasible_nodes] / rest_capacity
    remaining_capacity_ratio = (rest_capacity - demands[feasible_nodes]) / rest_capacity
    proximity_score = 1 - distances / np.max(distances)
    capacity_penalty = np.where(remaining_capacity_ratio < 0.2, 0.4, np.where(remaining_capacity_ratio < 0.5, 0.7, 1.0))
    urgency_score = distance_matrix[depot, feasible_nodes] / np.max(distance_matrix[depot, :])
    
    cluster_density = np.array([np.mean(distance_matrix[n, unvisited_nodes]) for n in feasible_nodes])
    diversity_score = np.array([np.std(distance_matrix[n, unvisited_nodes]) / np.max(distance_matrix) for n in feasible_nodes])
    exploration_factor = 0.5 * (cluster_density / np.max(cluster_density)) + 0.5 * diversity_score
    
    route_progress = len(unvisited_nodes) / len(distance_matrix)
    exploration_weight = 0.6 * (1 - route_progress**1.5)
    exploitation_weight = 1 - exploration_weight
    
    scores = (0.3 * exploitation_weight * proximity_score + 
              0.25 * exploitation_weight * utilizations + 
              0.2 * exploitation_weight * remaining_capacity_ratio * capacity_penalty + 
              0.1 * exploitation_weight * urgency_score + 
              0.15 * exploration_weight * exploration_factor)
    
    return feasible_nodes[np.argmax(scores)]



# Function 4 - Score: -0.26336192210290693
{The novel algorithm selects the next node by combining a hierarchical decision-making approach with adaptive feature prioritization, incorporating spatial-temporal proximity, demand-capacity balance with dynamic thresholds, route diversity preservation through node entropy, and a novel hybrid scoring mechanism that blends greedy exploitation with probabilistic exploration, while maintaining fallback to depot when necessary.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    utilizations = demands[feasible_nodes] / rest_capacity
    remaining_capacity = rest_capacity - demands[feasible_nodes]
    remaining_ratio = remaining_capacity / rest_capacity
    
    proximity_score = np.exp(-distances / np.mean(distances))
    capacity_score = np.where(remaining_ratio < 0.15, 0.1,
                            np.where(remaining_ratio < 0.3, 0.4,
                                   np.where(remaining_ratio < 0.5, 0.7, 1.0)))
    
    entropy = -np.sum((distance_matrix[feasible_nodes][:, unvisited_nodes] / np.sum(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1, keepdims=True)) * 
                     np.log1p(distance_matrix[feasible_nodes][:, unvisited_nodes] / np.sum(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1, keepdims=True)), axis=1)
    diversity_score = 1 - entropy / np.max(entropy)
    
    future_cost = np.min(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1) / np.max(distances)
    
    cluster_density = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / np.mean(distance_matrix)
    
    phase = 1 - len(feasible_nodes) / len(unvisited_nodes)
    w_proximity = 0.25 * (1 - phase**2)
    w_utilization = 0.15 + 0.05 * phase
    w_capacity = 0.12 * (1 + phase)
    w_diversity = 0.08 * (1 - phase**0.5)
    w_future = 0.18 * phase
    w_cluster = 0.12 * (1 - phase**1.5)
    w_random = 0.1 * phase
    
    scores = (w_proximity * proximity_score +
             w_utilization * utilizations +
             w_capacity * remaining_ratio * capacity_score +
             w_diversity * diversity_score +
             w_future * future_cost +
             w_cluster * cluster_density)
    
    if np.random.random() < 0.1 * phase:
        return np.random.choice(feasible_nodes)
    
    return feasible_nodes[np.argmax(scores)]



# Function 5 - Score: -0.26563982044126505
{The improved algorithm enhances node selection by incorporating dynamic multi-objective optimization with adaptive weights, integrating spatial clustering, temporal constraints, demand forecasting, vehicle load balancing, and a reinforcement learning-inspired reward shaping mechanism, while maintaining all previous features with optimized parameter tuning.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    utilizations = demands[feasible_nodes] / rest_capacity
    remaining_capacity = rest_capacity - demands[feasible_nodes]
    remaining_ratio = remaining_capacity / rest_capacity
    
    proximity_score = 1 - distances / np.max(distances)
    capacity_penalty = np.where(remaining_ratio < 0.1, 0.2,
                              np.where(remaining_ratio < 0.3, 0.5,
                                      np.where(remaining_ratio < 0.6, 0.8, 1.0)))
    
    visit_counts = np.bincount(feasible_nodes, minlength=len(demands))[feasible_nodes]
    diversification = 1 / (1 + np.log1p(visit_counts))
    
    look_ahead = np.min(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / np.max(distances)
    
    regret = (np.min(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1) - distances) / np.max(distances)
    
    compactness = np.sum(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1) / (len(unvisited_nodes) * np.max(distances))
    
    criticality = (demands[feasible_nodes] / (distances + 1e-6)) / np.max(demands[feasible_nodes] / (distances + 1e-6))
    
    cluster_score = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / np.max(distances)
    
    temporal_score = np.exp(-np.abs(remaining_ratio - 0.5))
    
    route_progress = 1 - len(feasible_nodes) / len(unvisited_nodes)
    w_proximity = 0.2 * (1 - route_progress**1.5) + 0.1 * route_progress
    w_utilization = 0.15 + 0.1 * route_progress
    w_capacity = 0.12 + 0.08 * route_progress
    w_diversification = 0.08 * (1 - route_progress**0.7)
    w_lookahead = 0.12 * (1 - route_progress)
    w_regret = 0.18 * route_progress**0.8
    w_compactness = 0.04 * (1 - route_progress)
    w_criticality = 0.12 * route_progress
    w_cluster = 0.05 * (1 - route_progress**0.3)
    w_temporal = 0.08 * route_progress**0.5
    
    scores = (w_proximity * proximity_score + 
              w_utilization * utilizations + 
              w_capacity * remaining_ratio * capacity_penalty + 
              w_diversification * diversification + 
              w_lookahead * look_ahead + 
              w_regret * regret + 
              w_compactness * compactness + 
              w_criticality * criticality + 
              w_cluster * cluster_score + 
              w_temporal * temporal_score)
    
    return feasible_nodes[np.argmax(scores)]



# Function 6 - Score: -0.26584731797895805
{The novel algorithm selects the next node by considering feasibility, a dynamic weighted score combining proximity, demand utilization, remaining capacity ratio with adaptive penalties, temporal efficiency factor, route continuity bonus, and a novel clustering-based regional preference factor, with a fallback to the depot if no feasible node exists.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    distances = distance_matrix[current_node, feasible_nodes]
    utilizations = demands[feasible_nodes] / rest_capacity
    remaining_capacity_ratio = (rest_capacity - demands[feasible_nodes]) / rest_capacity
    proximity_score = 1 - distances / np.max(distances)
    capacity_penalty = np.where(remaining_capacity_ratio < 0.2, 0.5, 1.0)
    temporal_factor = np.array([np.mean(distance_matrix[n, feasible_nodes]) / np.max(distance_matrix) for n in feasible_nodes])
    direction_bonus = np.array([distance_matrix[current_node, n] / np.max(distances) for n in feasible_nodes])
    cluster_factor = np.array([np.mean(distance_matrix[n, feasible_nodes]) / (np.mean(distance_matrix[n, unvisited_nodes]) + 1e-6) for n in feasible_nodes])
    scores = 0.3 * proximity_score + 0.2 * utilizations + 0.15 * remaining_capacity_ratio * capacity_penalty + 0.15 * temporal_factor + 0.1 * direction_bonus + 0.1 * cluster_factor
    return feasible_nodes[np.argmax(scores)]



# Function 7 - Score: -0.27387401351779755
{The novel algorithm selects the next node by considering feasibility, a dynamic weighted score combining proximity, demand utilization, remaining capacity ratio with adaptive penalties, a temporal urgency factor based on distance from depot, a novel regional exploration score favoring nodes in less explored regions, and a route progress-based adaptive weight adjustment, with a fallback to the depot if no feasible node exists.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    utilizations = demands[feasible_nodes] / rest_capacity
    remaining_capacity_ratio = (rest_capacity - demands[feasible_nodes]) / rest_capacity
    proximity_score = 1 - distances / np.max(distances)
    capacity_penalty = np.where(remaining_capacity_ratio < 0.25, 0.4, np.where(remaining_capacity_ratio < 0.5, 0.7, 1.0))
    urgency_score = distance_matrix[depot, feasible_nodes] / np.max(distance_matrix[depot, :])
    
    regional_exploration = np.array([np.mean(distance_matrix[n, unvisited_nodes]) for n in feasible_nodes])
    exploration_score = regional_exploration / np.max(regional_exploration)
    
    route_progress = len(unvisited_nodes) / len(distance_matrix)
    exploration_weight = 0.5 * route_progress
    exploitation_weight = 1 - exploration_weight
    
    scores = (0.3 * exploitation_weight * proximity_score + 
              0.25 * exploitation_weight * utilizations + 
              0.2 * exploitation_weight * remaining_capacity_ratio * capacity_penalty + 
              0.15 * exploitation_weight * urgency_score + 
              0.1 * exploration_weight * exploration_score)
    
    return feasible_nodes[np.argmax(scores)]



# Function 8 - Score: -0.2785370843001844
{The improved algorithm selects the next node by considering feasibility, a dynamic weighted score combining proximity, demand utilization, remaining capacity ratio with adaptive penalties, temporal efficiency factor, route continuity bonus, a novel clustering-based regional preference factor, and an added exploration factor for diversity, with a fallback to the depot if no feasible node exists.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    distances = distance_matrix[current_node, feasible_nodes]
    utilizations = demands[feasible_nodes] / rest_capacity
    remaining_capacity_ratio = (rest_capacity - demands[feasible_nodes]) / rest_capacity
    proximity_score = 1 - distances / np.max(distances)
    capacity_penalty = np.where(remaining_capacity_ratio < 0.2, 0.5, 1.0)
    temporal_factor = np.array([np.mean(distance_matrix[n, feasible_nodes]) / np.max(distance_matrix) for n in feasible_nodes])
    direction_bonus = np.array([distance_matrix[current_node, n] / np.max(distances) for n in feasible_nodes])
    cluster_factor = np.array([np.mean(distance_matrix[n, feasible_nodes]) / (np.mean(distance_matrix[n, unvisited_nodes]) + 1e-6) for n in feasible_nodes])
    exploration_factor = np.random.rand(len(feasible_nodes)) * 0.1
    scores = 0.25 * proximity_score + 0.2 * utilizations + 0.15 * remaining_capacity_ratio * capacity_penalty + 0.15 * temporal_factor + 0.1 * direction_bonus + 0.1 * cluster_factor + 0.05 * exploration_factor
    return feasible_nodes[np.argmax(scores)]



# Function 9 - Score: -0.2809179036377353
{The novel algorithm selects the next node by considering feasibility, a dynamic weighted score combining proximity, demand utilization, remaining capacity ratio with adaptive penalties, a time-window-like urgency score based on distance from the depot, a novel exploration factor combining spatial dispersion and route diversity with adaptive weights, and a route progress-based adaptive weight adjustment with a sigmoid transition from exploration to exploitation, with a fallback to the depot if no feasible node exists.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    utilizations = demands[feasible_nodes] / rest_capacity
    remaining_capacity_ratio = (rest_capacity - demands[feasible_nodes]) / rest_capacity
    proximity_score = 1 - distances / np.max(distances)
    capacity_penalty = np.where(remaining_capacity_ratio < 0.2, 0.3, np.where(remaining_capacity_ratio < 0.5, 0.6, 1.0))
    urgency_score = distance_matrix[depot, feasible_nodes] / np.max(distance_matrix[depot, :])
    
    dispersion_score = np.array([np.std(distance_matrix[n, unvisited_nodes]) for n in feasible_nodes])
    diversity_score = np.array([np.mean(distance_matrix[n, unvisited_nodes]) / np.max(distance_matrix) for n in feasible_nodes])
    exploration_factor = 0.4 * (dispersion_score / np.max(dispersion_score)) + 0.6 * diversity_score
    
    route_progress = len(unvisited_nodes) / len(distance_matrix)
    exploration_weight = 0.7 / (1 + np.exp(10 * (route_progress - 0.5)))
    exploitation_weight = 1 - exploration_weight
    
    scores = (0.35 * exploitation_weight * proximity_score + 
              0.25 * exploitation_weight * utilizations + 
              0.15 * exploitation_weight * remaining_capacity_ratio * capacity_penalty + 
              0.1 * exploitation_weight * urgency_score + 
              0.15 * exploration_weight * exploration_factor)
    
    return feasible_nodes[np.argmax(scores)]



# Function 10 - Score: -0.28687145442010853
{The novel algorithm selects the next node by considering feasibility, a dynamic weighted score combining proximity, demand utilization, remaining capacity ratio with adaptive penalties, a novel directional consistency factor favoring nodes aligned with the current route direction, a temporal-spatial balance factor, and a demand-density-based exploration bonus, with a fallback to the depot if no feasible node exists.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    utilizations = demands[feasible_nodes] / rest_capacity
    remaining_capacity_ratio = (rest_capacity - demands[feasible_nodes]) / rest_capacity
    proximity_score = 1 - distances / np.max(distances)
    capacity_penalty = np.where(remaining_capacity_ratio < 0.3, 0.4, np.where(remaining_capacity_ratio < 0.6, 0.7, 1.0))
    
    direction_vector = distance_matrix[current_node, :] - distance_matrix[depot, :]
    direction_score = np.array([1 - np.abs(direction_vector[n] - direction_vector[current_node]) / (np.max(np.abs(direction_vector)) + 1e-6) for n in feasible_nodes])
    
    temporal_spatial = np.array([(distance_matrix[depot, n] + distance_matrix[current_node, n]) / (2 * np.max(distance_matrix)) for n in feasible_nodes])
    
    demand_density = np.array([np.sum(demands[unvisited_nodes]) / (np.mean(distance_matrix[n, unvisited_nodes]) + 1e-6) for n in feasible_nodes])
    exploration_bonus = demand_density / np.max(demand_density)
    
    scores = (0.3 * proximity_score + 
              0.2 * utilizations + 
              0.15 * remaining_capacity_ratio * capacity_penalty + 
              0.15 * direction_score + 
              0.1 * temporal_spatial + 
              0.1 * exploration_bonus)
    
    return feasible_nodes[np.argmax(scores)]



