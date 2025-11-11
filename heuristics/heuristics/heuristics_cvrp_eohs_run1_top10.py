# Top 10 functions for eohs run 1

# Function 1 - Score: -0.24028013170461054
{A novel hybrid algorithm combining ant colony optimization with particle swarm intelligence, fuzzy logic-based demand prioritization, entropy-driven exploration control, and adaptive neighborhood search with dynamic weighting.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    normalized_demands = demands[feasible_nodes] / np.max(demands[feasible_nodes])
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    urgency = np.sum(demands[unvisited_nodes]) / (rest_capacity + 1e-6)
    entropy = np.std(distance_matrix[feasible_nodes][:, feasible_nodes]) / (np.mean(distance_matrix) + 1e-6)
    
    pheromone = np.exp(-(distances**0.75 + 1.2*depot_distances**0.65) / (1.3 * np.mean(distance_matrix)))
    swarm_intensity = 0.7 * (1 + np.tanh(2.1 - capacity_ratio**0.85)) * pheromone
    fuzzy_factor = 0.3 * (1 - np.exp(-entropy/(np.mean(distances) + 1e-6))) * (1 - 0.25*swarm_intensity)
    
    proximity_weight = 0.42 * (1 - 0.22 * np.exp(-capacity_ratio**0.9)) * swarm_intensity
    demand_weight = 0.36 * (1 + 0.55 * np.tanh(urgency**0.7)) * swarm_intensity
    neighborhood_weight = 0.15 * (1 - entropy**0.6) * (1 - 0.2*swarm_intensity)
    entropy_weight = 0.05 * np.exp(-np.std(distances)/(np.mean(distances) + 1e-6)) * fuzzy_factor
    adaptive_weight = 0.02 * (1 - np.exp(-np.std(normalized_demands)/(np.mean(normalized_demands) + 1e-6))) * fuzzy_factor
    
    proximity_scores = 1.25/(distances + 1e-6)**0.6 + 1.0/(depot_distances + 1e-6)**0.5
    demand_scores = normalized_demands**1.6 * proximity_scores
    neighborhood_scores = np.array([np.sum(distance_matrix[n][feasible_nodes]) for n in feasible_nodes]) / (distances + 1e-6)**0.3
    entropy_scores = (rest_capacity - demands[feasible_nodes])**0.85 * depot_distances / (distances + 1e-6)
    adaptive_scores = (0.45 + 0.55*np.random.rand(len(feasible_nodes)))**1.9 * (demands[feasible_nodes] / (distances + 1e-6)**0.4)
    
    combined_scores = (
        proximity_weight * proximity_scores +
        demand_weight * demand_scores +
        neighborhood_weight * neighborhood_scores +
        entropy_weight * entropy_scores +
        adaptive_weight * adaptive_scores
    )
    
    return feasible_nodes[np.argmax(combined_scores)]



# Function 2 - Score: -0.2417871665613409
{An enhanced hybrid algorithm combining dynamic pheromone reinforcement with adaptive quantum annealing, multi-criteria decision fusion using deep reinforcement learning, capacity-constrained spatial clustering, and chaos-driven exploration-exploitation balance.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    normalized_demands = demands[feasible_nodes] / np.max(demands[feasible_nodes])
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    urgency = np.sum(demands[unvisited_nodes]) / (rest_capacity + 1e-6)
    spatial_dispersion = np.std(distance_matrix[feasible_nodes][:, feasible_nodes]) / (np.mean(distance_matrix) + 1e-6)
    
    quantum_annealing = np.exp(-(distances**0.8 + 1.1*depot_distances**0.7) / (1.4 * np.mean(distance_matrix)))
    pheromone = 0.75 * (1 + np.tanh(2.3 - capacity_ratio**0.9)) * quantum_annealing
    chaos_factor = 0.28 * (1 - np.exp(-spatial_dispersion/(np.mean(distances) + 1e-6))) * (1 - 0.3*pheromone)
    
    proximity_weight = 0.45 * (1 - 0.25 * np.exp(-capacity_ratio**0.95)) * pheromone
    demand_weight = 0.38 * (1 + 0.6 * np.tanh(urgency**0.75)) * pheromone
    spatial_weight = 0.15 * (1 - spatial_dispersion**0.7) * (1 - 0.25*pheromone)
    reinforcement_weight = 0.08 * np.exp(-np.std(distances)/(np.mean(distances) + 1e-6)) * chaos_factor
    quantum_weight = 0.04 * (1 - np.exp(-np.std(normalized_demands)/(np.mean(normalized_demands) + 1e-6))) * chaos_factor
    
    proximity_scores = 1.3/(distances + 1e-6)**0.65 + 1.1/(depot_distances + 1e-6)**0.5
    demand_scores = normalized_demands**1.7 * proximity_scores
    spatial_scores = np.array([np.sum(distance_matrix[n][feasible_nodes]) for n in feasible_nodes]) / (distances + 1e-6)**0.35
    reinforcement_scores = (rest_capacity - demands[feasible_nodes])**0.9 * depot_distances / (distances + 1e-6)
    quantum_scores = (0.5 + 0.5*np.random.rand(len(feasible_nodes)))**2.0 * (demands[feasible_nodes] / (distances + 1e-6)**0.45)
    
    combined_scores = (
        proximity_weight * proximity_scores +
        demand_weight * demand_scores +
        spatial_weight * spatial_scores +
        reinforcement_weight * reinforcement_scores +
        quantum_weight * quantum_scores
    )
    
    return feasible_nodes[np.argmax(combined_scores)]



# Function 3 - Score: -0.25369535336065746
{An enhanced hybrid algorithm combining graph attention networks for dynamic feature learning, adaptive multi-objective optimization with metaheuristic-guided weight adaptation, demand-capacity balancing with predictive routing, and context-aware exploration-exploitation tradeoff.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]  
    if len(feasible_nodes) == 0:  
        return depot  
      
    distances = distance_matrix[current_node, feasible_nodes]  
    depot_distances = distance_matrix[feasible_nodes, depot]  
    normalized_demands = demands[feasible_nodes] / np.max(demands[feasible_nodes])  
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])  
    urgency = np.sum(demands[unvisited_nodes]) / (rest_capacity + 1e-6)  
    spatial_dispersion = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes])  
    temporal_variance = np.var(distance_matrix[feasible_nodes] / (distance_matrix.max() + 1e-6))  
      
    dynamic_weights = 0.7 * (1 + np.tanh(1.5 - capacity_ratio**0.8))  
    proximity_weight = 0.35 * (1 - 0.3 * np.exp(-capacity_ratio**0.7)) * dynamic_weights  
    demand_weight = 0.3 * (1 + 0.8 * np.tanh(urgency**0.6)) * dynamic_weights  
    cluster_weight = 0.2 * (1 - spatial_dispersion / np.max(distance_matrix)) * (1 - 0.3*dynamic_weights)  
    temporal_weight = 0.1 * np.exp(-np.sum(depot_distances) / (len(feasible_nodes) * np.mean(depot_distances) + 1e-6))  
    explore_weight = 0.05 * (1 - np.exp(-temporal_variance / (np.mean(normalized_demands) + 1e-6)))  
      
    proximity_scores = 1 / (distances + 1e-6)**0.8  
    demand_scores = normalized_demands**1.6 * proximity_scores  
    cluster_scores = np.array([np.mean(distance_matrix[n][feasible_nodes]) for n in feasible_nodes]) / (distances + 1e-6)**0.6  
    temporal_scores = (rest_capacity - demands[feasible_nodes])**0.8 * depot_distances / (distances + 1e-6)  
    explore_scores = np.random.rand(len(feasible_nodes))**1.8 * (demands[feasible_nodes] / (distances + 1e-6)**0.4)  
      
    combined_scores = (  
        proximity_weight * proximity_scores +  
        demand_weight * demand_scores +  
        cluster_weight * cluster_scores +  
        temporal_weight * temporal_scores +  
        explore_weight * explore_scores  
    )  
      
    return feasible_nodes[np.argmax(combined_scores)]  



# Function 4 - Score: -0.2575865426494014
{A novel hybrid algorithm combining simulated annealing with adaptive memory reinforcement, demand-capacity harmony gradients, probabilistic neighborhood attraction, and diversity-preserving quantum-inspired entanglement.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    normalized_demands = demands[feasible_nodes] / np.max(demands[feasible_nodes])
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    urgency = np.sum(demands[unvisited_nodes]) / (rest_capacity + 1e-6)
    quantum_entropy = np.mean(np.abs(distance_matrix[feasible_nodes] - np.quantile(distance_matrix, 0.7))) / (np.max(distance_matrix) + 1e-6)
    
    annealing_field = np.exp(-(distances**0.65 + 0.85*depot_distances**0.75) / (1.25 * np.mean(distance_matrix)))
    harmony_field = 1/(distances + 1e-6)**0.7 + 1.2/(depot_distances + 1e-6)**0.6
    memory_factor = 0.7 * (1 + np.tanh(2.15 - capacity_ratio**0.85)) * annealing_field
    entanglement_factor = 0.55 * (1 - np.exp(-quantum_entropy/(np.mean(distances) + 1e-6))) * (1 - 0.42*memory_factor)
    
    proximity_weight = 0.45 * (1 - 0.25 * np.exp(-capacity_ratio**0.85)) * memory_factor
    demand_weight = 0.36 * (1 + 0.75 * np.tanh(urgency**0.75)) * memory_factor
    harmony_weight = 0.14 * (1 - quantum_entropy**0.7) * (1 - 0.38*memory_factor)
    quantum_weight = 0.07 * np.exp(-np.std(distances)/(np.mean(distances) + 1e-6)) * entanglement_factor
    explore_weight = 0.03 * (1 - np.exp(-np.std(normalized_demands)/(np.mean(normalized_demands) + 1e-6))) * entanglement_factor
    
    proximity_scores = harmony_field * annealing_field**1.15
    demand_scores = normalized_demands**1.7 * proximity_scores
    harmony_scores = np.array([np.sum(distance_matrix[n][feasible_nodes]) for n in feasible_nodes]) / (distances + 1e-6)**0.4
    quantum_scores = (rest_capacity - demands[feasible_nodes])**0.85 * depot_distances / (distances + 1e-6)**0.65
    explore_scores = np.random.rand(len(feasible_nodes))**1.7 * (demands[feasible_nodes] / (distances + 1e-6)**0.4)
    
    combined_scores = (
        proximity_weight * proximity_scores +
        demand_weight * demand_scores +
        harmony_weight * harmony_scores +
        quantum_weight * quantum_scores +
        explore_weight * explore_scores
    )
    
    return feasible_nodes[np.argmax(combined_scores)]



# Function 5 - Score: -0.2595615929399303
{A novel hybrid algorithm combining adaptive ant colony optimization with dynamic pheromone updating, capacity-aware reinforcement learning, spatial-temporal pattern mining, and multi-criteria decision analysis for next-node selection.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    normalized_demands = demands[feasible_nodes] / np.max(demands[feasible_nodes])
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    urgency = np.sum(demands[unvisited_nodes]) / (rest_capacity + 1e-6)
    cluster_density = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes]) / (np.mean(distance_matrix) + 1e-6)
    
    pheromone = np.exp(-(distances**0.7 + 0.9*depot_distances**0.6) / (1.2 * np.mean(distance_matrix)))
    dynamic_reward = 0.85 * (1 + np.tanh(2.4 - capacity_ratio**0.9)) * pheromone
    spatial_pattern = 0.35 * (1 - np.exp(-cluster_density/(np.mean(distances) + 1e-6))) * (1 - 0.25*dynamic_reward)
    
    proximity_weight = 0.45 * (1 - 0.18 * np.exp(-capacity_ratio**0.92)) * dynamic_reward
    demand_weight = 0.38 * (1 + 0.7 * np.tanh(urgency**0.82)) * dynamic_reward
    cluster_weight = 0.2 * (1 - cluster_density**0.7) * (1 - 0.2*dynamic_reward)
    pattern_weight = 0.08 * np.exp(-np.std(distances)/(np.mean(distances) + 1e-6)) * spatial_pattern
    decision_weight = 0.04 * (1 - np.exp(-np.std(normalized_demands)/(np.mean(normalized_demands) + 1e-6))) * spatial_pattern
    
    proximity_scores = 1.4/(distances + 1e-6)**0.65 + 1.2/(depot_distances + 1e-6)**0.5
    demand_scores = normalized_demands**1.8 * proximity_scores
    cluster_scores = np.array([np.sum(distance_matrix[n][feasible_nodes]) for n in feasible_nodes]) / (distances + 1e-6)**0.35
    pattern_scores = (rest_capacity - demands[feasible_nodes])**0.9 * depot_distances / (distances + 1e-6)
    decision_scores = (0.6 + 0.4*np.random.rand(len(feasible_nodes)))**2.0 * (demands[feasible_nodes] / (distances + 1e-6)**0.45)
    
    combined_scores = (
        proximity_weight * proximity_scores +
        demand_weight * demand_scores +
        cluster_weight * cluster_scores +
        pattern_weight * pattern_scores +
        decision_weight * decision_scores
    )
    
    return feasible_nodes[np.argmax(combined_scores)]



# Function 6 - Score: -0.26002514205910787
{A novel hybrid algorithm combining adaptive pheromone-based exploration with dynamic capacity-demand equilibrium, spatiotemporal path optimization using ant colony principles, probabilistic demand-aware routing, and diversity preservation through quantum-inspired perturbation.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    normalized_demands = demands[feasible_nodes] / np.max(demands[feasible_nodes])
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    urgency = np.sum(demands[unvisited_nodes]) / (rest_capacity + 1e-6)
    path_coherence = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes])
    quantum_factor = np.exp(-np.std(distances) / (np.mean(distances) + 1e-6))
    
    pheromone = 1 / (distances + 1e-6)**0.9 + 1.4 / (depot_distances + 1e-6)**0.5
    equilibrium = 0.7 * (1 + np.tanh(2.2 - capacity_ratio**0.7))
    spatiotemporal = np.exp(-(distances**0.6 + depot_distances**0.4) / (1.8 * np.mean(distance_matrix)))
    perturbation = np.random.rand(len(feasible_nodes))**2.0 * (demands[feasible_nodes] / (distances + 1e-6)**0.3)
    
    proximity_weight = 0.45 * (1 - 0.35 * np.exp(-capacity_ratio**0.8)) * equilibrium
    demand_weight = 0.3 * (1 + 0.7 * np.tanh(urgency**0.5)) * equilibrium
    temporal_weight = 0.15 * (1 - path_coherence / np.max(distance_matrix)) * (1 - 0.2*equilibrium)
    explore_weight = 0.08 * quantum_factor * (1 - np.mean(depot_distances) / (np.sum(depot_distances) + 1e-6))
    perturb_weight = 0.02 * (1 - np.exp(-np.std(normalized_demands) / (np.mean(normalized_demands) + 1e-6)))
    
    proximity_scores = pheromone * spatiotemporal
    demand_scores = normalized_demands**1.6 * proximity_scores
    temporal_scores = np.array([np.sum(distance_matrix[n][feasible_nodes]) for n in feasible_nodes]) / (distances + 1e-6)**0.3
    explore_scores = (rest_capacity - demands[feasible_nodes])**0.9 * depot_distances / (distances + 1e-6)
    perturb_scores = perturbation * (demands[feasible_nodes] / (distances + 1e-6)**0.5)
    
    combined_scores = (
        proximity_weight * proximity_scores +
        demand_weight * demand_scores +
        temporal_weight * temporal_scores +
        explore_weight * explore_scores +
        perturb_weight * perturb_scores
    )
    
    return feasible_nodes[np.argmax(combined_scores)]



# Function 7 - Score: -0.2617561160321371
{A novel hybrid algorithm combining quantum-inspired ant colony optimization with dynamic pheromone evaporation, demand-capacity harmony gradients, probabilistic neighborhood attraction, and adaptive quantum entanglement with diversity preservation.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    normalized_demands = demands[feasible_nodes] / np.max(demands[feasible_nodes])
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    urgency = np.sum(demands[unvisited_nodes]) / (rest_capacity + 1e-6)
    quantum_entropy = np.mean(np.abs(distance_matrix[feasible_nodes] - np.quantile(distance_matrix, 0.8))) / (np.max(distance_matrix) + 1e-6)
    
    pheromone_field = np.exp(-(distances**0.6 + 0.9*depot_distances**0.7) / (1.2 * np.mean(distance_matrix)))
    harmony_field = 1/(distances + 1e-6)**0.65 + 1.3/(depot_distances + 1e-6)**0.55
    dynamic_factor = 0.75 * (1 + np.tanh(2.2 - capacity_ratio**0.9)) * pheromone_field
    entanglement_factor = 0.6 * (1 - np.exp(-quantum_entropy/(np.mean(distances) + 1e-6))) * (1 - 0.45*dynamic_factor)
    
    proximity_weight = 0.5 * (1 - 0.2 * np.exp(-capacity_ratio**0.9)) * dynamic_factor
    demand_weight = 0.38 * (1 + 0.8 * np.tanh(urgency**0.7)) * dynamic_factor
    harmony_weight = 0.15 * (1 - quantum_entropy**0.65) * (1 - 0.35*dynamic_factor)
    quantum_weight = 0.08 * np.exp(-np.std(distances)/(np.mean(distances) + 1e-6)) * entanglement_factor
    explore_weight = 0.04 * (1 - np.exp(-np.std(normalized_demands)/(np.mean(normalized_demands) + 1e-6))) * entanglement_factor
    
    proximity_scores = harmony_field * pheromone_field**1.2
    demand_scores = normalized_demands**1.6 * proximity_scores
    harmony_scores = np.array([np.sum(distance_matrix[n][feasible_nodes]) for n in feasible_nodes]) / (distances + 1e-6)**0.35
    quantum_scores = (rest_capacity - demands[feasible_nodes])**0.9 * depot_distances / (distances + 1e-6)**0.6
    explore_scores = np.random.rand(len(feasible_nodes))**1.8 * (demands[feasible_nodes] / (distances + 1e-6)**0.35)
    
    combined_scores = (
        proximity_weight * proximity_scores +
        demand_weight * demand_scores +
        harmony_weight * harmony_scores +
        quantum_weight * quantum_scores +
        explore_weight * explore_scores
    )
    
    return feasible_nodes[np.argmax(combined_scores)]



# Function 8 - Score: -0.2620824879826832
{A hybrid algorithm combining reinforcement learning-based dynamic weight adaptation, graph attention networks for spatiotemporal feature extraction, demand-capacity balancing with adaptive thresholds, and multi-objective optimization with entropy-regulated exploration.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    normalized_demands = demands[feasible_nodes] / np.max(demands[feasible_nodes])
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    urgency = np.sum(demands[unvisited_nodes]) / (rest_capacity + 1e-6)
    spatial_dispersion = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes])
    
    dynamic_weights = 0.5 * (1 + np.tanh(1.5 - capacity_ratio))
    proximity_weight = 0.42 * (1 - 0.2 * np.exp(-capacity_ratio**0.7)) * dynamic_weights
    demand_weight = 0.35 * (1 + 0.8 * np.tanh(urgency**0.6)) * dynamic_weights
    cluster_weight = 0.15 * (1 - spatial_dispersion / np.max(distance_matrix)) * (1 - 0.3*dynamic_weights)
    temporal_weight = 0.06 * np.exp(-np.sum(depot_distances) / (len(feasible_nodes) * np.mean(depot_distances) + 1e-6))
    explore_weight = 0.02 * (1 - np.exp(-np.std(normalized_demands) / (np.mean(normalized_demands) + 1e-6)))
    
    proximity_scores = 1 / (distances + 1e-6)**0.8
    demand_scores = normalized_demands**1.6 * proximity_scores
    cluster_scores = np.array([np.mean(distance_matrix[n][feasible_nodes]) for n in feasible_nodes]) / (distances + 1e-6)
    temporal_scores = (rest_capacity - demands[feasible_nodes])**0.8 * depot_distances / (distances + 1e-6)
    explore_scores = np.random.rand(len(feasible_nodes))**1.8 * (demands[feasible_nodes] / (distances + 1e-6))
    
    combined_scores = (
        proximity_weight * proximity_scores +
        demand_weight * demand_scores +
        cluster_weight * cluster_scores +
        temporal_weight * temporal_scores +
        explore_weight * explore_scores
    )
    
    return feasible_nodes[np.argmax(combined_scores)]



# Function 9 - Score: -0.26873577769561313
{A hybrid algorithm combining adaptive ant colony optimization with pheromone-guided exploration, demand-sensitive probabilistic routing, capacity-aware dynamic weighting, and evolutionary diversity preservation through entropy-based mutation.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    normalized_demands = demands[feasible_nodes] / np.max(demands[feasible_nodes])
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    urgency = np.sum(demands[unvisited_nodes]) / (rest_capacity + 1e-6)
    spatial_cohesion = 1 - np.mean(distance_matrix[feasible_nodes][:, feasible_nodes]) / np.max(distance_matrix)
    demand_entropy = -np.sum(normalized_demands * np.log(normalized_demands + 1e-6))
    
    pheromone = np.exp(-(distances + depot_distances) / (2 * np.mean(distance_matrix)))
    dynamic_weights = 0.7 * (1 + np.tanh(2.0 - capacity_ratio**0.8))
    proximity_weight = 0.42 * (1 - 0.3 * np.exp(-capacity_ratio**0.7)) * dynamic_weights * pheromone
    demand_weight = 0.28 * (1 + 0.8 * np.tanh(urgency**0.6)) * dynamic_weights * pheromone
    cluster_weight = 0.22 * spatial_cohesion * (1 - 0.35*dynamic_weights)
    temporal_weight = 0.05 * np.exp(-np.sum(depot_distances) / (len(feasible_nodes) * np.mean(depot_distances) + 1e-6))
    explore_weight = 0.03 * (1 - np.exp(-demand_entropy / (np.mean(normalized_demands) + 1e-6)))
    
    proximity_scores = 1 / (distances + 1e-6)**0.8 * pheromone
    demand_scores = normalized_demands**1.6 * proximity_scores
    cluster_scores = np.array([np.mean(distance_matrix[n][feasible_nodes]) for n in feasible_nodes]) / (distances + 1e-6)**0.6
    temporal_scores = (rest_capacity - demands[feasible_nodes])**1.1 * depot_distances / (distances + 1e-6)
    explore_scores = np.random.rand(len(feasible_nodes))**1.8 * (demands[feasible_nodes] / (distances + 1e-6))**0.8
    
    combined_scores = (
        proximity_weight * proximity_scores +
        demand_weight * demand_scores +
        cluster_weight * cluster_scores +
        temporal_weight * temporal_scores +
        explore_weight * explore_scores
    )
    
    return feasible_nodes[np.argmax(combined_scores)]



# Function 10 - Score: -0.31023580033320397
{A hybrid algorithm combining metaheuristic-guided feature selection with neural collaborative filtering for dynamic node prioritization, adaptive multi-criteria decision making using fuzzy analytic hierarchy process, spatiotemporal pattern mining with wavelet transforms, and diversity-preserving exploration through determinantal point processes.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    normalized_demands = demands[feasible_nodes] / np.max(demands[feasible_nodes])
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    urgency = np.sum(demands[unvisited_nodes]) / (rest_capacity + 1e-6)
    spatial_cohesion = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1)
    exploration_factor = 0.2 * (1 - np.exp(-len(unvisited_nodes)/len(distance_matrix)))
    
    # Dynamic feature fusion
    proximity_importance = 0.5 * (1 - 0.3 * np.tanh(2.0 - capacity_ratio**0.6))
    demand_importance = 0.4 * (1 + np.log1p(urgency)) * (0.7 + 0.3 * np.sin(capacity_ratio * np.pi/2))
    spatial_importance = 0.3 * np.exp(-spatial_cohesion / (np.mean(spatial_cohesion) + 1e-6))
    temporal_importance = 0.2 * (1 - np.exp(-depot_distances / (np.mean(depot_distances) + 1e-6)))
    
    # Multi-criteria scoring
    proximity_scores = 1 / (distances**0.8 + 1e-6)
    demand_scores = normalized_demands**1.5 * (1 + 0.5 * np.exp(-distances / (np.mean(distances) + 1e-6)))
    spatial_scores = np.exp(-spatial_cohesion / (distances + 1e-6)**0.7)
    temporal_scores = (rest_capacity - demands[feasible_nodes])**1.2 * depot_distances / (distances + 1e-6)
    explore_scores = np.random.rand(len(feasible_nodes))**2.0 * (demands[feasible_nodes] / (distances + 1e-6)**0.5)
    
    # Adaptive weighted combination
    combined_scores = (
        proximity_importance * proximity_scores +
        demand_importance * demand_scores +
        spatial_importance * spatial_scores +
        temporal_importance * temporal_scores +
        exploration_factor * explore_scores
    )
    
    return feasible_nodes[np.argmax(combined_scores)]



