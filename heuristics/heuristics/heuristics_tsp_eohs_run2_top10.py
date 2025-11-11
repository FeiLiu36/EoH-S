# Top 10 functions for eohs run 2

# Function 1 - Score: -0.1884985012217556
{The algorithm selects the next node by combining dynamic weights for proximity, progress toward the destination, local node clustering, future path flexibility, a stochastic exploration factor, a dynamic penalty for potential dead-ends, a "path entropy" metric, a "path momentum" factor, a "path curvature" metric, and introduces novel "path resonance" and "path divergence" metrics that evaluate harmonic relationships and directional spread of remaining nodes, with adaptive weight tuning based on traversal stage and remaining nodes.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    if len(unvisited_nodes) == 0:
        return destination_node

    current_to_dest = distance_matrix[current_node, destination_node]
    scores = []
    traversal_stage = 1 - (len(unvisited_nodes) / (len(distance_matrix) - 1))
    remaining_nodes_ratio = len(unvisited_nodes) / len(distance_matrix)

    for node in unvisited_nodes:
        dist_to_node = distance_matrix[current_node, node]
        node_to_dest = distance_matrix[node, destination_node]
        progress = current_to_dest - node_to_dest
        
        local_density = np.exp(-np.std(distance_matrix[node, unvisited_nodes]) / np.mean(distance_matrix[node, unvisited_nodes])) if len(unvisited_nodes) > 1 else 1
        
        remaining_nodes = unvisited_nodes[unvisited_nodes != node]
        if len(remaining_nodes) > 0:
            min_return_cost = np.min(distance_matrix[remaining_nodes, destination_node])
            avg_return_cost = np.mean(distance_matrix[remaining_nodes, destination_node])
            flexibility = (avg_return_cost - dist_to_node) / (node_to_dest + 1e-6)
            dead_end_penalty = np.log(1 + avg_return_cost - min_return_cost) if avg_return_cost > min_return_cost else 0
            
            path_entropy = np.sum(np.log(distance_matrix[node, remaining_nodes] + 1e-6)) / len(remaining_nodes)
            path_curvature = np.abs(np.mean(distance_matrix[node, remaining_nodes]) - dist_to_node) / (dist_to_node + 1e-6)
            path_resonance = np.mean(np.cos(distance_matrix[node, remaining_nodes] / np.max(distance_matrix[node, remaining_nodes]))) if len(remaining_nodes) > 0 else 0
            path_divergence = np.std(np.arctan2(distance_matrix[node, remaining_nodes], distance_matrix[remaining_nodes, destination_node])) if len(remaining_nodes) > 0 else 0
        else:
            flexibility = 0
            dead_end_penalty = 0
            path_entropy = 0
            path_curvature = 0
            path_resonance = 0
            path_divergence = 0
            
        exploration_bias = np.random.uniform(0.93, 1.07) * (1 - 0.12 * traversal_stage)
        path_momentum = np.exp(-dist_to_node / np.mean(distance_matrix[current_node, unvisited_nodes])) if len(unvisited_nodes) > 1 else 1
        
        weight_dist = 0.17 * (1 - np.tanh(progress / (current_to_dest + 1e-6))) * (1 + 0.07 * traversal_stage)
        weight_progress = 0.15 + 0.08 * np.arctan(progress / (current_to_dest + 1e-6)) * (1 - 0.05 * traversal_stage)
        weight_density = 0.12 * (1 - 0.22 * traversal_stage) * (1 + 0.11 * remaining_nodes_ratio)
        weight_flexibility = 0.11 * (1 - np.tanh(flexibility)) * (1 + 0.11 * traversal_stage)
        weight_penalty = 0.06 * (1 - 0.55 * traversal_stage)
        weight_entropy = 0.08 * (1 - traversal_stage) * (1 + 0.21 * remaining_nodes_ratio)
        weight_curvature = 0.07 * (1 - 0.42 * traversal_stage) * (1 - 0.16 * remaining_nodes_ratio)
        weight_momentum = 0.07 * (1 + 0.22 * traversal_stage)
        weight_resonance = 0.09 * (1 - 0.38 * traversal_stage) * (1 + 0.17 * remaining_nodes_ratio)
        weight_divergence = 0.08 * (1 - 0.52 * traversal_stage) * (1 + 0.14 * remaining_nodes_ratio)
        
        score = exploration_bias * (
            weight_dist * (1 / (dist_to_node + 1e-6)) +
            weight_progress * progress +
            weight_density * local_density +
            weight_flexibility * (1 / (1 + np.abs(flexibility))) -
            weight_penalty * dead_end_penalty +
            weight_entropy * path_entropy -
            weight_curvature * path_curvature +
            weight_momentum * path_momentum +
            weight_resonance * path_resonance -
            weight_divergence * path_divergence
        )
        scores.append(score)

    return unvisited_nodes[np.argmax(scores)]



# Function 2 - Score: -0.18898480540495433
{The algorithm selects the next node by dynamically balancing proximity to the current node, alignment toward the destination, local node density, and a penalty for nodes that would create future detours, using an adaptive weighted score of distance, progress, density, and detour potential.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    if len(unvisited_nodes) == 0:
        return destination_node

    current_to_dest = distance_matrix[current_node, destination_node]
    scores = []

    for node in unvisited_nodes:
        dist_to_node = distance_matrix[current_node, node]
        node_to_dest = distance_matrix[node, destination_node]
        progress = current_to_dest - node_to_dest
        
        local_density = np.sum(distance_matrix[node] < np.percentile(distance_matrix[node], 25))
        
        remaining_nodes = unvisited_nodes[unvisited_nodes != node]
        if len(remaining_nodes) > 0:
            detour_penalty = np.min(distance_matrix[node, remaining_nodes]) + np.min(distance_matrix[remaining_nodes, destination_node]) - node_to_dest
        else:
            detour_penalty = 0
            
        adaptive_weight_dist = 0.4 * (1 - progress / current_to_dest) if current_to_dest > 0 else 0.5
        adaptive_weight_progress = 0.3 + (0.1 * (progress / current_to_dest)) if current_to_dest > 0 else 0.3
        
        score = (adaptive_weight_dist * (1 / dist_to_node) + 
                 adaptive_weight_progress * progress + 
                 0.2 * local_density - 
                 0.1 * detour_penalty)
        scores.append(score)

    return unvisited_nodes[np.argmax(scores)]



# Function 3 - Score: -0.19528240466330382
{The algorithm selects the next node by dynamically adjusting weights based on remaining path length, local clustering, and a probabilistic exploration-exploitation trade-off, incorporating adaptive normalization and a randomness factor.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    if len(unvisited_nodes) == 0:
        return destination_node

    dist_to_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    avg_dist = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    min_cluster_dist = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)

    # Adaptive weights based on remaining nodes
    progress = len(unvisited_nodes) / distance_matrix.shape[0]
    w_current = 0.4 * (1 - (dist_to_current / np.max(dist_to_current + 1e-10)))
    w_dest = 0.3 * (1 - progress) * (1 - (dist_to_dest / np.max(dist_to_dest + 1e-10)))
    w_avg = 0.2 * (avg_dist / np.max(avg_dist + 1e-10))
    w_cluster = 0.1 * (1 - (min_cluster_dist / np.max(min_cluster_dist + 1e-10)))

    combined_score = w_current + w_dest + w_avg + w_cluster
    combined_score = combined_score * (1 + 0.1 * np.random.random())  # Exploration factor

    next_node = unvisited_nodes[np.argmax(combined_score)]
    return next_node



# Function 4 - Score: -0.19668141066437927
{The algorithm selects the next node by combining dynamic weights for proximity, progress toward the destination, local node clustering, future path flexibility, a stochastic exploration factor, a dynamic penalty for potential dead-ends, a "path entropy" metric, a "path momentum" factor, a "path curvature" metric, and introduces a novel "path synergy" metric that evaluates collective node relationships, with adaptive weight tuning based on traversal stage, remaining nodes, and a new "path resilience" metric to avoid fragile paths.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    if len(unvisited_nodes) == 0:
        return destination_node

    current_to_dest = distance_matrix[current_node, destination_node]
    scores = []
    traversal_stage = 1 - (len(unvisited_nodes) / (len(distance_matrix) - 1))
    remaining_nodes_ratio = len(unvisited_nodes) / len(distance_matrix)

    for node in unvisited_nodes:
        dist_to_node = distance_matrix[current_node, node]
        node_to_dest = distance_matrix[node, destination_node]
        progress = current_to_dest - node_to_dest
        
        local_density = np.exp(-np.std(distance_matrix[node, unvisited_nodes]) / np.mean(distance_matrix[node, unvisited_nodes])) if len(unvisited_nodes) > 1 else 1
        
        remaining_nodes = unvisited_nodes[unvisited_nodes != node]
        if len(remaining_nodes) > 0:
            min_return_cost = np.min(distance_matrix[remaining_nodes, destination_node])
            avg_return_cost = np.mean(distance_matrix[remaining_nodes, destination_node])
            flexibility = (avg_return_cost - dist_to_node) / (node_to_dest + 1e-6)
            dead_end_penalty = np.log(1 + avg_return_cost - min_return_cost) if avg_return_cost > min_return_cost else 0
            
            path_entropy = np.sum(np.log(distance_matrix[node, remaining_nodes] + 1e-6)) / len(remaining_nodes)
            path_curvature = np.abs(np.mean(distance_matrix[node, remaining_nodes]) - dist_to_node) / (dist_to_node + 1e-6)
            path_synergy = np.mean(np.exp(-distance_matrix[node, remaining_nodes] / np.mean(distance_matrix[node, remaining_nodes]))) if len(remaining_nodes) > 0 else 0
            path_resilience = np.min(distance_matrix[node, remaining_nodes]) / (np.mean(distance_matrix[node, remaining_nodes]) + 1e-6)
        else:
            flexibility = 0
            dead_end_penalty = 0
            path_entropy = 0
            path_curvature = 0
            path_synergy = 0
            path_resilience = 0
            
        exploration_bias = np.random.uniform(0.9, 1.1) * (1 - 0.2 * traversal_stage)
        path_momentum = np.exp(-dist_to_node / np.mean(distance_matrix[current_node, unvisited_nodes])) if len(unvisited_nodes) > 1 else 1
        
        weight_dist = 0.17 * (1 - np.tanh(progress / (current_to_dest + 1e-6))) * (1 + 0.1 * traversal_stage)
        weight_progress = 0.15 + 0.08 * np.arctan(progress / (current_to_dest + 1e-6)) * (1 - 0.05 * traversal_stage)
        weight_density = 0.1 * (1 - 0.3 * traversal_stage) * (1 + 0.15 * remaining_nodes_ratio)
        weight_flexibility = 0.1 * (1 - np.tanh(flexibility)) * (1 + 0.15 * traversal_stage)
        weight_penalty = 0.06 * (1 - 0.7 * traversal_stage)
        weight_entropy = 0.1 * (1 - traversal_stage) * (1 + 0.25 * remaining_nodes_ratio)
        weight_curvature = 0.08 * (1 - 0.5 * traversal_stage) * (1 - 0.2 * remaining_nodes_ratio)
        weight_momentum = 0.06 * (1 + 0.3 * traversal_stage)
        weight_synergy = 0.09 * (1 - 0.4 * traversal_stage) * (1 + 0.2 * remaining_nodes_ratio)
        weight_resilience = 0.09 * (1 - 0.6 * traversal_stage) * (1 + 0.15 * remaining_nodes_ratio)
        
        score = exploration_bias * (
            weight_dist * (1 / (dist_to_node + 1e-6)) +
            weight_progress * progress +
            weight_density * local_density +
            weight_flexibility * (1 / (1 + np.abs(flexibility))) -
            weight_penalty * dead_end_penalty +
            weight_entropy * path_entropy -
            weight_curvature * path_curvature +
            weight_momentum * path_momentum +
            weight_synergy * path_synergy +
            weight_resilience * path_resilience
        )
        scores.append(score)

    return unvisited_nodes[np.argmax(scores)]



# Function 5 - Score: -0.19819114701903795
{The algorithm selects the next node by combining dynamic weights for proximity, progress toward the destination, local node clustering, future path flexibility, a stochastic exploration factor, a dynamic penalty for potential dead-ends, a "path entropy" metric, a "path momentum" factor, a "path curvature" metric, and introduces novel "path harmony" and "path tension" metrics that evaluate global node relationships and path stability, with adaptive weight tuning based on traversal stage and remaining nodes.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    if len(unvisited_nodes) == 0:
        return destination_node

    current_to_dest = distance_matrix[current_node, destination_node]
    scores = []
    traversal_stage = 1 - (len(unvisited_nodes) / (len(distance_matrix) - 1))
    remaining_nodes_ratio = len(unvisited_nodes) / len(distance_matrix)

    for node in unvisited_nodes:
        dist_to_node = distance_matrix[current_node, node]
        node_to_dest = distance_matrix[node, destination_node]
        progress = current_to_dest - node_to_dest
        
        local_density = np.exp(-np.std(distance_matrix[node, unvisited_nodes]) / np.mean(distance_matrix[node, unvisited_nodes])) if len(unvisited_nodes) > 1 else 1
        
        remaining_nodes = unvisited_nodes[unvisited_nodes != node]
        if len(remaining_nodes) > 0:
            min_return_cost = np.min(distance_matrix[remaining_nodes, destination_node])
            avg_return_cost = np.mean(distance_matrix[remaining_nodes, destination_node])
            flexibility = (avg_return_cost - dist_to_node) / (node_to_dest + 1e-6)
            dead_end_penalty = np.log(1 + avg_return_cost - min_return_cost) if avg_return_cost > min_return_cost else 0
            
            path_entropy = np.sum(np.log(distance_matrix[node, remaining_nodes] + 1e-6)) / len(remaining_nodes)
            path_curvature = np.abs(np.mean(distance_matrix[node, remaining_nodes]) - dist_to_node) / (dist_to_node + 1e-6)
            path_harmony = np.mean(np.exp(-distance_matrix[node, remaining_nodes] / np.max(distance_matrix[node, remaining_nodes]))) if len(remaining_nodes) > 0 else 0
            path_tension = np.std(distance_matrix[node, remaining_nodes]) / (np.mean(distance_matrix[node, remaining_nodes]) + 1e-6)
        else:
            flexibility = 0
            dead_end_penalty = 0
            path_entropy = 0
            path_curvature = 0
            path_harmony = 0
            path_tension = 0
            
        exploration_bias = np.random.uniform(0.92, 1.08) * (1 - 0.15 * traversal_stage)
        path_momentum = np.exp(-dist_to_node / np.mean(distance_matrix[current_node, unvisited_nodes])) if len(unvisited_nodes) > 1 else 1
        
        weight_dist = 0.16 * (1 - np.tanh(progress / (current_to_dest + 1e-6))) * (1 + 0.08 * traversal_stage)
        weight_progress = 0.14 + 0.09 * np.arctan(progress / (current_to_dest + 1e-6)) * (1 - 0.04 * traversal_stage)
        weight_density = 0.11 * (1 - 0.25 * traversal_stage) * (1 + 0.12 * remaining_nodes_ratio)
        weight_flexibility = 0.11 * (1 - np.tanh(flexibility)) * (1 + 0.12 * traversal_stage)
        weight_penalty = 0.05 * (1 - 0.6 * traversal_stage)
        weight_entropy = 0.09 * (1 - traversal_stage) * (1 + 0.22 * remaining_nodes_ratio)
        weight_curvature = 0.07 * (1 - 0.45 * traversal_stage) * (1 - 0.18 * remaining_nodes_ratio)
        weight_momentum = 0.07 * (1 + 0.25 * traversal_stage)
        weight_harmony = 0.1 * (1 - 0.35 * traversal_stage) * (1 + 0.18 * remaining_nodes_ratio)
        weight_tension = 0.1 * (1 - 0.55 * traversal_stage) * (1 + 0.15 * remaining_nodes_ratio)
        
        score = exploration_bias * (
            weight_dist * (1 / (dist_to_node + 1e-6)) +
            weight_progress * progress +
            weight_density * local_density +
            weight_flexibility * (1 / (1 + np.abs(flexibility))) -
            weight_penalty * dead_end_penalty +
            weight_entropy * path_entropy -
            weight_curvature * path_curvature +
            weight_momentum * path_momentum +
            weight_harmony * path_harmony -
            weight_tension * path_tension
        )
        scores.append(score)

    return unvisited_nodes[np.argmax(scores)]



# Function 6 - Score: -0.19945169141710808
{The new algorithm enhances the original by introducing a dynamic multi-criteria decision system that integrates adaptive neural-inspired weights, quantum-inspired probabilistic exploration, topological persistence analysis for dead-end prevention, and a novel "route harmony" metric that balances exploration-exploitation tradeoffs while considering spatial-temporal path coherence and emergent cluster patterns.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    if len(unvisited_nodes) == 0:
        return destination_node

    current_to_dest = distance_matrix[current_node, destination_node]
    scores = []
    phase = 1 - (len(unvisited_nodes) / (len(distance_matrix) - 1))
    remaining_ratio = len(unvisited_nodes) / len(distance_matrix)
    
    centrality = np.mean(distance_matrix[unvisited_nodes, :], axis=1) if len(unvisited_nodes) > 0 else np.zeros(len(unvisited_nodes))
    direction_coherence = np.sum(distance_matrix[unvisited_nodes, :], axis=0) / len(unvisited_nodes) if len(unvisited_nodes) < len(distance_matrix) - 1 else np.zeros(len(distance_matrix))
    
    cluster_tendency = np.array([np.exp(-np.std(distance_matrix[n, unvisited_nodes]) / (np.mean(distance_matrix[n, unvisited_nodes]) + 1e-6)) if len(unvisited_nodes) > 1 else 1 for n in unvisited_nodes])
    
    for i, node in enumerate(unvisited_nodes):
        dist = distance_matrix[current_node, node]
        progress = current_to_dest - distance_matrix[node, destination_node]
        
        remaining = unvisited_nodes[unvisited_nodes != node]
        if len(remaining) > 0:
            min_return = np.min(distance_matrix[remaining, destination_node])
            avg_return = np.mean(distance_matrix[remaining, destination_node])
            flexibility = (avg_return - dist) / (distance_matrix[node, destination_node] + 1e-6)
            topological_risk = np.log(1 + np.max(distance_matrix[node, remaining]) - np.min(distance_matrix[node, remaining]))
            
            path_variance = np.sum(np.log(distance_matrix[node, remaining] + 1e-6)) / len(remaining)
            global_balance = np.min(distance_matrix[node, remaining]) / (np.mean(distance_matrix[node, remaining]) + 1e-6)
            quantum_bias = np.random.normal(1, 0.1 * (1 - phase))
            route_harmony = np.exp(-np.abs(direction_coherence[node] - np.mean(direction_coherence)))
        else:
            flexibility = 0
            topological_risk = 0
            path_variance = 0
            global_balance = 0
            quantum_bias = 1
            route_harmony = 1
            
        w_dist = 0.15 * (1 + 0.2 * phase) * (1 - np.tanh(progress / (current_to_dest + 1e-6)))
        w_progress = 0.2 + 0.1 * np.sin(progress / (current_to_dest + 1e-6)) * (1 - 0.15 * phase)
        w_cluster = 0.1 * (1 - 0.25 * phase) * (1 + 0.2 * remaining_ratio)
        w_flex = 0.08 * (1 - np.tanh(flexibility)) * (1 + 0.1 * phase)
        w_topology = 0.07 * (1 - 0.5 * phase)
        w_variance = 0.12 * (1 - phase) * (1 + 0.3 * remaining_ratio)
        w_global = 0.06 * (1 - 0.4 * phase)
        w_central = 0.08 * (1 - 0.35 * phase)
        w_harmony = 0.14 * (1 - 0.2 * phase)
        
        score = quantum_bias * (
            w_dist * (1 / (dist + 1e-6)) +
            w_progress * progress +
            w_cluster * cluster_tendency[i] +
            w_flex * (1 / (1 + np.abs(flexibility))) -
            w_topology * topological_risk +
            w_variance * path_variance +
            w_global * global_balance +
            w_central * centrality[i] +
            w_harmony * route_harmony
        )
        scores.append(score)

    return unvisited_nodes[np.argmax(scores)]



# Function 7 - Score: -0.20072057461356207
{The algorithm selects the next node by combining dynamic weights for proximity, progress toward the destination, local node clustering, future path flexibility, a stochastic exploration factor, a dynamic penalty for potential dead-ends, a "path entropy" metric, a "path momentum" factor, and a novel "path curvature" metric, with adaptive weight tuning based on traversal stage, remaining nodes, and a new "path potential" metric that estimates future path quality.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    if len(unvisited_nodes) == 0:
        return destination_node

    current_to_dest = distance_matrix[current_node, destination_node]
    scores = []
    traversal_stage = 1 - (len(unvisited_nodes) / (len(distance_matrix) - 1))
    remaining_nodes_ratio = len(unvisited_nodes) / len(distance_matrix)

    for node in unvisited_nodes:
        dist_to_node = distance_matrix[current_node, node]
        node_to_dest = distance_matrix[node, destination_node]
        progress = current_to_dest - node_to_dest
        
        local_density = np.exp(-np.std(distance_matrix[node, unvisited_nodes]) / np.mean(distance_matrix[node, unvisited_nodes])) if len(unvisited_nodes) > 1 else 1
        
        remaining_nodes = unvisited_nodes[unvisited_nodes != node]
        if len(remaining_nodes) > 0:
            min_return_cost = np.min(distance_matrix[remaining_nodes, destination_node])
            avg_return_cost = np.mean(distance_matrix[remaining_nodes, destination_node])
            flexibility = (avg_return_cost - dist_to_node) / (node_to_dest + 1e-6)
            dead_end_penalty = np.log(1 + avg_return_cost - min_return_cost) if avg_return_cost > min_return_cost else 0
            
            path_entropy = np.sum(np.log(distance_matrix[node, remaining_nodes] + 1e-6)) / len(remaining_nodes)
            path_curvature = np.abs(np.mean(distance_matrix[node, remaining_nodes]) - dist_to_node) / (dist_to_node + 1e-6)
            path_potential = np.mean(np.min(distance_matrix[remaining_nodes][:, remaining_nodes], axis=1)) if len(remaining_nodes) > 1 else 0
        else:
            flexibility = 0
            dead_end_penalty = 0
            path_entropy = 0
            path_curvature = 0
            path_potential = 0
            
        exploration_bias = np.random.uniform(0.9, 1.1) * (1 - 0.2 * traversal_stage)
        path_momentum = np.exp(-dist_to_node / np.mean(distance_matrix[current_node, unvisited_nodes])) if len(unvisited_nodes) > 1 else 1
        
        weight_dist = 0.18 * (1 - np.tanh(progress / (current_to_dest + 1e-6))) * (1 + 0.1 * traversal_stage)
        weight_progress = 0.2 + 0.08 * np.arctan(progress / (current_to_dest + 1e-6)) * (1 - 0.05 * traversal_stage)
        weight_density = 0.1 * (1 - 0.3 * traversal_stage) * (1 + 0.15 * remaining_nodes_ratio)
        weight_flexibility = 0.1 * (1 - np.tanh(flexibility)) * (1 + 0.15 * traversal_stage)
        weight_penalty = 0.06 * (1 - 0.7 * traversal_stage)
        weight_entropy = 0.1 * (1 - traversal_stage) * (1 + 0.25 * remaining_nodes_ratio)
        weight_curvature = 0.08 * (1 - 0.5 * traversal_stage) * (1 - 0.2 * remaining_nodes_ratio)
        weight_momentum = 0.1 * (1 + 0.3 * traversal_stage)
        weight_potential = 0.08 * (1 - 0.4 * traversal_stage)
        
        score = exploration_bias * (
            weight_dist * (1 / (dist_to_node + 1e-6)) +
            weight_progress * progress +
            weight_density * local_density +
            weight_flexibility * (1 / (1 + np.abs(flexibility))) -
            weight_penalty * dead_end_penalty +
            weight_entropy * path_entropy -
            weight_curvature * path_curvature +
            weight_momentum * path_momentum +
            weight_potential * path_potential
        )
        scores.append(score)

    return unvisited_nodes[np.argmax(scores)]



# Function 8 - Score: -0.2050923908085824
{The new algorithm employs a hybrid approach combining gravitational attraction modeling, fractal-based path complexity assessment, dynamic pheromone-like trail reinforcement, evolutionary fitness scoring, and a novel "path resonance" metric that evaluates harmonic relationships between potential paths while adapting weights using a chaos theory-inspired feedback loop.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    if len(unvisited_nodes) == 0:
        return destination_node

    current_to_dest = distance_matrix[current_node, destination_node]
    scores = []
    chaos_factor = 0.5 * (1 + np.cos(len(unvisited_nodes) / len(distance_matrix) * np.pi))
    fractal_dim = 1.5 + 0.5 * np.random.rand()
    
    pheromone = np.ones(len(unvisited_nodes))
    if hasattr(select_next_node, 'pheromone_trail'):
        for i, node in enumerate(unvisited_nodes):
            if node in select_next_node.pheromone_trail:
                pheromone[i] += select_next_node.pheromone_trail[node]
    
    centrality = np.mean(distance_matrix[unvisited_nodes, :], axis=1)
    harmonic_ratio = np.array([np.sum(1/(distance_matrix[n, unvisited_nodes] + 1e-6)) / len(unvisited_nodes) for n in unvisited_nodes])
    
    for i, node in enumerate(unvisited_nodes):
        dist = distance_matrix[current_node, node]
        progress = current_to_dest - distance_matrix[node, destination_node]
        
        remaining = unvisited_nodes[unvisited_nodes != node]
        if len(remaining) > 0:
            min_return = np.min(distance_matrix[remaining, destination_node])
            path_complexity = np.sum(np.power(distance_matrix[node, remaining], fractal_dim)) / len(remaining)
            fitness = np.exp(-(dist + min_return) / (2 * current_to_dest + 1e-6))
            resonance = np.mean(np.cos(distance_matrix[node, remaining] / (np.mean(distance_matrix[node, remaining]) + 1e-6)))
            gravitational_pull = (centrality[i] * harmonic_ratio[i]) / (dist + 1e-6)
        else:
            path_complexity = 0
            fitness = 1
            resonance = 1
            gravitational_pull = 1
            
        w_dist = 0.2 * (1 - 0.3 * chaos_factor)
        w_progress = 0.18 + 0.05 * np.sin(chaos_factor * np.pi)
        w_pheromone = 0.15 * (1 + chaos_factor)
        w_complexity = 0.12 * (1 - 0.4 * chaos_factor)
        w_fitness = 0.1 * (1 + 0.2 * chaos_factor)
        w_resonance = 0.13 * (1 - 0.25 * chaos_factor)
        w_gravity = 0.12 * (1 + 0.15 * chaos_factor)
        
        score = (w_dist * (1 / (dist + 1e-6)) +
                w_progress * progress +
                w_pheromone * pheromone[i] -
                w_complexity * path_complexity +
                w_fitness * fitness +
                w_resonance * resonance +
                w_gravity * gravitational_pull)
        scores.append(score)

    next_node = unvisited_nodes[np.argmax(scores)]
    if not hasattr(select_next_node, 'pheromone_trail'):
        select_next_node.pheromone_trail = {}
    select_next_node.pheromone_trail[next_node] = select_next_node.pheromone_trail.get(next_node, 0) + 1
    
    return next_node



# Function 9 - Score: -0.20649592139205156
{The algorithm selects the next node by combining dynamic weights for proximity, progress toward the destination, local node clustering, future path flexibility, a stochastic exploration factor, a novel "path tension" metric that measures the balance between attraction and repulsion forces in the path network, a "path harmony" metric that evaluates geometric alignment, a "path potential" metric that estimates future options, and introduces a "path diversity" metric to ensure varied path exploration, with adaptive weight tuning based on traversal stage and remaining nodes.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    if len(unvisited_nodes) == 0:
        return destination_node

    current_to_dest = distance_matrix[current_node, destination_node]
    scores = []
    traversal_stage = 1 - (len(unvisited_nodes) / (len(distance_matrix) - 1))
    remaining_nodes_ratio = len(unvisited_nodes) / len(distance_matrix)

    for node in unvisited_nodes:
        dist_to_node = distance_matrix[current_node, node]
        node_to_dest = distance_matrix[node, destination_node]
        progress = current_to_dest - node_to_dest
        
        local_density = np.exp(-np.std(distance_matrix[node, unvisited_nodes]) / np.mean(distance_matrix[node, unvisited_nodes])) if len(unvisited_nodes) > 1 else 1
        
        remaining_nodes = unvisited_nodes[unvisited_nodes != node]
        if len(remaining_nodes) > 0:
            min_return_cost = np.min(distance_matrix[remaining_nodes, destination_node])
            avg_return_cost = np.mean(distance_matrix[remaining_nodes, destination_node])
            flexibility = (avg_return_cost - dist_to_node) / (node_to_dest + 1e-6)
            
            path_tension = np.mean(np.abs(distance_matrix[current_node, remaining_nodes] - dist_to_node)) / (dist_to_node + 1e-6)
            
            direction_vector = distance_matrix[node] - distance_matrix[current_node]
            path_harmony = np.mean(np.abs(np.diff(distance_matrix[node, remaining_nodes]))) if len(remaining_nodes) > 1 else 0
            
            path_potential = np.mean(np.exp(-distance_matrix[node, remaining_nodes] / np.mean(distance_matrix[node, remaining_nodes]))) if len(remaining_nodes) > 0 else 0
            
            path_diversity = np.std(distance_matrix[node, remaining_nodes]) / (np.mean(distance_matrix[node, remaining_nodes]) + 1e-6)
        else:
            flexibility = 0
            path_tension = 0
            path_harmony = 0
            path_potential = 0
            path_diversity = 0
            
        exploration_bias = np.random.uniform(0.9, 1.1) * (1 - 0.2 * traversal_stage)
        
        weight_dist = 0.2 * (1 - np.tanh(progress / (current_to_dest + 1e-6))) * (1 + 0.1 * traversal_stage)
        weight_progress = 0.18 + 0.1 * np.arctan(progress / (current_to_dest + 1e-6)) * (1 - 0.05 * traversal_stage)
        weight_density = 0.12 * (1 - 0.3 * traversal_stage) * (1 + 0.15 * remaining_nodes_ratio)
        weight_flexibility = 0.12 * (1 - np.tanh(flexibility)) * (1 + 0.15 * traversal_stage)
        weight_tension = 0.08 * (1 - 0.5 * traversal_stage)
        weight_harmony = 0.1 * (1 - 0.4 * traversal_stage)
        weight_potential = 0.1 * (1 - 0.6 * traversal_stage) * (1 + 0.2 * remaining_nodes_ratio)
        weight_diversity = 0.1 * (1 - 0.3 * traversal_stage) * (1 + 0.15 * remaining_nodes_ratio)
        
        score = exploration_bias * (
            weight_dist * (1 / (dist_to_node + 1e-6)) +
            weight_progress * progress +
            weight_density * local_density +
            weight_flexibility * (1 / (1 + np.abs(flexibility))) -
            weight_tension * path_tension +
            weight_harmony * (1 / (1 + path_harmony)) +
            weight_potential * path_potential +
            weight_diversity * path_diversity
        )
        scores.append(score)

    return unvisited_nodes[np.argmax(scores)]



# Function 10 - Score: -0.2403795139192597
{The algorithm selects the next node by balancing proximity to the current node, proximity to the destination node, and the node's centrality in the remaining unvisited set, using a weighted score.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    if len(unvisited_nodes) == 0:  
        return destination_node  

    # Calculate weights for proximity to current, destination, and centrality  
    dist_to_current = distance_matrix[current_node, unvisited_nodes]  
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]  
    centrality = np.sum(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)  

    # Normalize and combine weights  
    w_current = 0.5 * (1 - (dist_to_current / np.max(dist_to_current)))  
    w_dest = 0.3 * (1 - (dist_to_dest / np.max(dist_to_dest)))  
    w_centrality = 0.2 * (centrality / np.max(centrality))  

    combined_score = w_current + w_dest + w_centrality  
    next_node = unvisited_nodes[np.argmax(combined_score)]  

    return next_node  



