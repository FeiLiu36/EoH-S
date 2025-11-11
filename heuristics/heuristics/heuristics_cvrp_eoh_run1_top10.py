# Top 10 functions for eoh run 1

# Function 1 - Score: -0.276024311373689
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
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    current_distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    distance_savings = (distance_matrix[current_node, depot] + distance_matrix[depot, feasible_nodes]) - current_distances
    
    demand_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    
    avg_distances = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1)
    density_ratio = current_distances / (avg_distances + 1e-6)
    
    capacity_factor = rest_capacity / (np.max(demands[feasible_nodes]) + 1e-6)
    proximity_factor = np.mean(current_distances) / (np.mean(depot_distances) + 1e-6)
    
    w1 = max(0.3, 0.7 - capacity_factor * 0.4)
    w2 = min(0.4, 0.2 + proximity_factor * 0.2)
    w3 = 1.0 - w1 - w2
    
    weights = w1 * distance_savings + w2 * demand_ratio + w3 * (1 / (density_ratio + 1e-6))
    next_node = feasible_nodes[np.argmax(weights)]
    return next_node



# Function 2 - Score: -0.2764849887020937
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
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    current_distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    distance_savings = (distance_matrix[current_node, depot] + distance_matrix[depot, feasible_nodes]) - current_distances
    
    demand_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    avg_distances = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1)
    density_ratio = current_distances / (avg_distances + 1e-6)
    
    capacity_factor = rest_capacity / np.max(demands[feasible_nodes] + 1e-6)
    proximity_factor = np.mean(current_distances) / (np.mean(depot_distances) + 1e-6)
    
    w1 = max(0.3, 0.7 - capacity_factor * 0.4)
    w2 = min(0.4, 0.2 + proximity_factor * 0.3)
    w3 = 1.0 - w1 - w2
    
    weights = w1 * distance_savings + w2 * demand_ratio + w3 * (1 / (density_ratio + 1e-6))
    next_node = feasible_nodes[np.argmax(weights)]
    return next_node



# Function 3 - Score: -0.2886288323998356
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
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    current_distances = distance_matrix[current_node, feasible_nodes]
    normalized_dist = current_distances / np.max(current_distances)
    demand_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    depot_penalty = distance_matrix[feasible_nodes, depot] / np.max(distance_matrix[feasible_nodes, depot])
    
    capacity_weight = 1.0 - (rest_capacity / np.max(rest_capacity))
    scores = (0.5 * normalized_dist) + (capacity_weight * demand_ratio) - (0.3 * depot_penalty)
    
    next_node = feasible_nodes[np.argmin(scores)]
    return next_node



# Function 4 - Score: -0.2886288323998356
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
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    current_distances = distance_matrix[current_node, feasible_nodes]
    normalized_dist = current_distances / np.max(current_distances)
    demand_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    depot_penalty = distance_matrix[feasible_nodes, depot] / np.max(distance_matrix[feasible_nodes, depot])
    
    capacity_weight = 1.0 - (rest_capacity / np.max(rest_capacity))
    distance_depot_ratio = distance_matrix[current_node, depot] / np.max(distance_matrix[current_node, :])
    demand_penalty = (demands[feasible_nodes] / np.max(demands[feasible_nodes])) * capacity_weight
    
    scores = (0.5 * normalized_dist) + (0.4 * capacity_weight * demand_ratio) - (0.3 * depot_penalty) + (0.2 * distance_depot_ratio) + (0.1 * demand_penalty)
    
    next_node = feasible_nodes[np.argmin(scores)]
    return next_node



# Function 5 - Score: -0.2886288323998356
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
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    current_distances = distance_matrix[current_node, feasible_nodes]
    normalized_dist = current_distances / np.max(current_distances)
    demand_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    depot_penalty = distance_matrix[feasible_nodes, depot] / np.max(distance_matrix[feasible_nodes, depot])
    
    capacity_weight = 1.0 - (rest_capacity / np.max(rest_capacity))
    distance_depot_ratio = distance_matrix[current_node, depot] / np.max(distance_matrix[current_node, :])
    high_demand_penalty = (demands[feasible_nodes] / np.max(demands[feasible_nodes])) * capacity_weight
    
    scores = (0.5 * normalized_dist) + (0.4 * capacity_weight * demand_ratio) - (0.3 * depot_penalty) + (0.2 * distance_depot_ratio) + (0.1 * high_demand_penalty)
    
    next_node = feasible_nodes[np.argmin(scores)]
    return next_node



# Function 6 - Score: -0.2886288323998356
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
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    current_distances = distance_matrix[current_node, feasible_nodes]
    normalized_dist = current_distances / np.max(current_distances)
    demand_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    depot_penalty = distance_matrix[feasible_nodes, depot] / np.max(distance_matrix[feasible_nodes, depot])
    
    capacity_weight = 1.0 - (rest_capacity / np.max(rest_capacity))
    distance_depot_ratio = distance_matrix[current_node, depot] / np.max(distance_matrix[current_node, :])
    high_demand_penalty = (demands[feasible_nodes] / np.max(demands[feasible_nodes])) * capacity_weight
    
    scores = (0.5 * normalized_dist) + (0.4 * capacity_weight * demand_ratio) - (0.3 * depot_penalty) + (0.2 * distance_depot_ratio) + (0.1 * high_demand_penalty)
    
    next_node = feasible_nodes[np.argmin(scores)]
    return next_node



# Function 7 - Score: -0.28904666215430097
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
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    current_distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    proximity = np.sqrt(current_distances) / (distance_matrix[current_node, depot] + 1e-6)
    demand_ratio = np.exp(demands[feasible_nodes]) / (current_distances + 1e-6)
    
    capacity_factor = np.sqrt(rest_capacity / np.mean(demands[feasible_nodes] + 1e-6))
    urgency = np.sqrt(depot_distances / (current_distances + 1e-6))
    
    cost = 0.5 * proximity + (1 / (demand_ratio + 1e-6)) * capacity_factor + urgency * (1 - capacity_factor)
    
    next_node = feasible_nodes[np.argmin(cost)]
    return next_node



# Function 8 - Score: -0.29229496399528954
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
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    current_distances = distance_matrix[current_node, feasible_nodes]
    demand_urgency = np.exp(demands[feasible_nodes] / (rest_capacity + 1e-6))
    exploration_bonus = np.sqrt(distance_matrix[feasible_nodes, depot] / (current_distances + 1e-6))
    
    scores = (1.0 / (current_distances + 1e-6)) * demand_urgency * exploration_bonus
    next_node = feasible_nodes[np.argmax(scores)]
    return next_node



# Function 9 - Score: -0.2924300491522513
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
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    current_distances = distance_matrix[current_node, feasible_nodes]
    normalized_dist = current_distances / np.max(current_distances)
    demand_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    depot_penalty = distance_matrix[feasible_nodes, depot] / np.max(distance_matrix[feasible_nodes, depot])
    
    capacity_weight = 1.0 - (rest_capacity / np.max(rest_capacity))
    distance_depot_ratio = distance_matrix[current_node, depot] / np.max(distance_matrix[current_node, :])
    
    scores = (0.4 * normalized_dist) + (0.6 * capacity_weight * demand_ratio) - (0.2 * depot_penalty) + (0.3 * distance_depot_ratio)
    
    next_node = feasible_nodes[np.argmin(scores)]
    return next_node



# Function 10 - Score: -0.2924300491522513
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
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    current_distances = distance_matrix[current_node, feasible_nodes]
    normalized_dist = current_distances / np.max(current_distances)
    demand_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    depot_penalty = distance_matrix[feasible_nodes, depot] / np.max(distance_matrix[feasible_nodes, depot])
    
    capacity_weight = 1.0 - (rest_capacity / np.max(rest_capacity))
    distance_depot_ratio = distance_matrix[current_node, depot] / np.max(distance_matrix[:, depot])
    
    scores = (0.4 * normalized_dist) + (0.5 * capacity_weight * demand_ratio) - (0.2 * depot_penalty) + (0.3 * distance_depot_ratio)
    
    next_node = feasible_nodes[np.argmin(scores)]
    return next_node



