# Top 10 functions for funsearch run 1

# Function 1 - Score: -0.29793086451602924
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
    best_score = -float('inf')
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            if distance > 0:
                urgency = (demand / (rest_capacity + 1e-6)) * (1.0 / (depot_distance + 1e-6))
                proximity = 1.0 / (distance + 1e-6)
                capacity_utilization = (rest_capacity - demand) / (rest_capacity + 1e-6)
                savings = (distance_matrix[current_node][depot] - distance + depot_distance) / (distance_matrix[current_node][depot] + 1e-6)
                score = proximity * (1.0 + urgency + capacity_utilization + savings) - (depot_distance / (rest_capacity + 1e-6))
            else:
                score = float('inf')
            
            if score > best_score:
                best_score = score
                next_node = node

    return next_node if next_node != -1 else depot



# Function 2 - Score: -0.30794950689206574
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
    best_score = -float('inf')
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            urgency = (rest_capacity - demand) / (rest_capacity + 1e-6)
            proximity = 1 / (distance + 1e-6)
            return_penalty = depot_distance / (distance + depot_distance + 1e-6)
            demand_ratio = demand / (rest_capacity + 1e-6)
            cluster_factor = np.mean(distance_matrix[node, unvisited_nodes]) if len(unvisited_nodes) > 1 else 1.0
            future_capacity = (rest_capacity - demand) / (rest_capacity + 1e-6)
            
            score = (demand_ratio * proximity * (1 + cluster_factor)) + (urgency * proximity * future_capacity) - (return_penalty * 0.5)

            if score > best_score:
                best_score = score
                next_node = node

    return next_node if next_node != -1 else depot



# Function 3 - Score: -0.3086997011804736
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
    best_score = -float('inf')
    next_node = -1
    current_to_depot = distance_matrix[current_node][depot]

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            urgency = (rest_capacity - demand) / (rest_capacity + 1e-6)
            proximity = 1 / (distance + 1e-6)
            return_penalty = depot_distance / (current_to_depot + 1e-6)
            savings = (current_to_depot + depot_distance - distance) / (current_to_depot + 1e-6)
            demand_factor = (demand / (rest_capacity + 1e-6)) * proximity
            cluster_factor = np.mean(distance_matrix[node][unvisited_nodes]) if len(unvisited_nodes) > 1 else 0
            time_criticality = (np.sum(demands[unvisited_nodes]) - demand) / (rest_capacity + 1e-6)
            score = (demand_factor * 2) + (urgency * proximity) - (0.5 * return_penalty) + savings - (0.1 * cluster_factor) - (0.2 * time_criticality)
            if score > best_score:
                best_score = score
                next_node = node

    return next_node if next_node != -1 else depot



# Function 4 - Score: -0.30964170927018086
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
    best_score = -float('inf')
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            if distance > 0:
                urgency = (demand / (rest_capacity + 1e-6)) * (1.0 / (depot_distance + 1e-6))
                proximity = 1.0 / (distance + 1e-6)
                capacity_utilization = (rest_capacity - demand) / (rest_capacity + 1e-6)
                savings = distance_matrix[current_node][depot] - distance + depot_distance
                score = proximity * (1.0 + urgency + capacity_utilization + savings) - (depot_distance / (rest_capacity + 1e-6))
            else:
                score = float('inf')
            
            if score > best_score:
                best_score = score
                next_node = node

    return next_node if next_node != -1 else depot



# Function 5 - Score: -0.30964170927018086
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
    best_score = -float('inf')
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            if distance > 0:
                urgency = (demand / (rest_capacity + 1e-6)) * (1.0 / (depot_distance + 1e-6))
                proximity = 1.0 / (distance + 1e-6)
                capacity_utilization = (rest_capacity - demand) / (rest_capacity + 1e-6)
                savings = distance_matrix[current_node][depot] - distance + distance_matrix[node][depot]
                score = proximity * (1.0 + urgency + capacity_utilization + savings) - (depot_distance / (rest_capacity + 1e-6))
            else:
                score = float('inf')
            
            if score > best_score:
                best_score = score
                next_node = node

    return next_node if next_node != -1 else depot



# Function 6 - Score: -0.30964170927018086
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
    best_score = -float('inf')
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            if distance > 0:
                urgency = (demand / (rest_capacity + 1e-6)) * (1.0 / (depot_distance + 1e-6))
                proximity = 1.0 / (distance + 1e-6)
                capacity_utilization = (rest_capacity - demand) / (rest_capacity + 1e-6)
                savings = distance_matrix[current_node][depot] + distance_matrix[depot][node] - distance
                score = proximity * (1.0 + urgency + capacity_utilization + savings) - (depot_distance / (rest_capacity + 1e-6))
            else:
                score = float('inf')
            
            if score > best_score:
                best_score = score
                next_node = node

    return next_node if next_node != -1 else depot



# Function 7 - Score: -0.3117799878390834
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
    best_score = -float('inf')
    next_node = -1
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    node_demands = demands[feasible_nodes]
    
    urgency = node_demands / (rest_capacity + 1e-6)
    proximity = 1.0 / (distances + 1e-6)
    return_cost = depot_distances / (rest_capacity + 1e-6)
    remaining_capacity = (rest_capacity - node_demands) / (rest_capacity + 1e-6)
    
    scores = proximity * (1.0 + urgency - 0.3 * return_cost + 0.2 * remaining_capacity)
    best_idx = np.argmax(scores)
    next_node = feasible_nodes[best_idx]
    
    return next_node if next_node != -1 else depot



# Function 8 - Score: -0.31212758170797017
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
    best_score = -float('inf')
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            if distance > 0:
                urgency = demand / (rest_capacity + 1e-6)
                proximity = 1.0 / (distance + 1e-6)
                return_cost = depot_distance / (rest_capacity + 1e-6)
                remaining_capacity = (rest_capacity - demand) / (rest_capacity + 1e-6)
                score = proximity * (1.0 + urgency - return_cost) + remaining_capacity
            else:
                score = float('inf')
            
            if score > best_score:
                best_score = score
                next_node = node

    return next_node if next_node != -1 else depot



# Function 9 - Score: -0.31212758170797017
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
    best_score = -float('inf')
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            if distance > 0:
                urgency = demand / (rest_capacity + 1e-6)
                proximity = 1.0 / (distance + 1e-6)
                return_cost = depot_distance / (rest_capacity + 1e-6)
                remaining_capacity_ratio = (rest_capacity - demand) / (rest_capacity + 1e-6)
                score = proximity * (1.0 + urgency - return_cost) + remaining_capacity_ratio
            else:
                score = float('inf')
            
            if score > best_score:
                best_score = score
                next_node = node

    return next_node if next_node != -1 else depot



# Function 10 - Score: -0.312146213330597
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
    best_score = -np.inf
    next_node = -1
    max_distance = distance_matrix.max() + 1e-6
    max_demand = demands.max() + 1e-6
    avg_demand = demands[unvisited_nodes].mean() if len(unvisited_nodes) > 0 else max_demand

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            normalized_demand = demand / max_demand
            normalized_distance = distance / max_distance
            normalized_depot_distance = depot_distance / max_distance
            demand_ratio = demand / (avg_demand + 1e-6)
            
            efficiency = (normalized_demand / (normalized_distance + 1e-6)) * (rest_capacity / (demand + 1e-6))
            urgency = demand_ratio * (1 - normalized_distance)
            return_penalty = 0.4 * normalized_depot_distance
            proximity_bonus = 1.0 / (distance + depot_distance + 1e-6)
            capacity_utilization = (rest_capacity - demand) / (rest_capacity + 1e-6)
            
            score = (0.5 * efficiency + 0.3 * urgency + 0.2 * proximity_bonus * (1 - capacity_utilization)) - return_penalty
            
            if score > best_score:
                best_score = score
                next_node = node

    return depot if next_node == -1 else next_node



