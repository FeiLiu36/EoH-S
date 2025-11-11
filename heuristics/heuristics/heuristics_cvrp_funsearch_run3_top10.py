# Top 10 functions for funsearch run 3

# Function 1 - Score: -0.30565703028396196
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
            urgency = demand / (rest_capacity + 1e-6)
            proximity = 1 / (distance + 1e-6)
            return_cost = depot_distance / (distance_matrix[current_node].mean() + 1e-6)
            capacity_ratio = rest_capacity / (demand + 1e-6)
            time_sensitivity = (distance_matrix[node].mean() + 1e-6) / (distance_matrix.mean() + 1e-6)
            score = (urgency * proximity * capacity_ratio * time_sensitivity) - return_cost
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        return depot
    return next_node



# Function 2 - Score: -0.3124822055859894
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
    best_score = -1
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            urgency = 1 / (1 + depot_distance)
            efficiency = demand / (distance + 1e-6)
            capacity_utilization = rest_capacity / (demand + 1e-6)
            proximity = 1 / (1 + np.mean(distance_matrix[node, unvisited_nodes]))
            distance_penalty = 1 / (1 + np.log1p(distance))
            demand_balance = 1 / (1 + np.abs(rest_capacity - demand))
            cluster_penalty = 1 / (1 + np.std(distance_matrix[node, unvisited_nodes]))
            time_window = 1 / (1 + np.max(distance_matrix[node, unvisited_nodes]) - np.min(distance_matrix[node, unvisited_nodes]))
            density_factor = 1 / (1 + np.sum(distance_matrix[node, unvisited_nodes] < np.mean(distance_matrix)))
            
            score = (efficiency * urgency * capacity_utilization * proximity * 
                    distance_penalty * demand_balance * cluster_penalty * time_window * 
                    density_factor * (1 + 0.1 * np.exp(-0.05 * rest_capacity)))
            
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        next_node = depot

    return next_node



# Function 3 - Score: -0.314773244139523
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
            urgency = demand / (rest_capacity + 1e-6)
            proximity = 1 / (distance + 1e-6)
            return_cost = depot_distance / (distance_matrix[current_node].mean() + 1e-6)
            capacity_ratio = rest_capacity / (demand + 1e-6)
            time_sensitivity = (distance_matrix[node].mean() + 1e-6) / (distance_matrix.mean() + 1e-6)
            cluster_penalty = distance_matrix[node][unvisited_nodes].mean() / (distance_matrix.mean() + 1e-6)
            score = (urgency * proximity * capacity_ratio * time_sensitivity) - return_cost - cluster_penalty
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        return depot
    return next_node



# Function 4 - Score: -0.31496960818925374
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
    best_score = -1
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            score = (demand / (distance + 1e-6)) - (depot_distance / (rest_capacity + 1e-6)) + (rest_capacity / (distance + 1e-6)) + (1.0 / (demand + 1e-6)) - (distance * depot_distance / (rest_capacity + 1e-6))
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        return depot
    return next_node



# Function 5 - Score: -0.31496960818925374
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
    best_score = -1
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            score = (demand / (distance + 1e-6)) - (depot_distance / (rest_capacity + 1e-6)) + (rest_capacity / (distance + 1e-6)) + (1.0 / (demand + 1e-6)) - (distance * depot_distance / (rest_capacity + 1e-6))
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        return depot
    return next_node



# Function 6 - Score: -0.31496960818925374
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
    best_score = -1
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            score = (demand / (distance + 1e-6)) - (depot_distance / (rest_capacity + 1e-6)) + (rest_capacity / (distance + 1e-6)) + (1.0 / (demand + 1e-6)) - (distance * depot_distance / (rest_capacity + 1e-6))
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        return depot
    return next_node



# Function 7 - Score: -0.31500470525727087
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
            score = (demand / (distance + 1e-6)) - (depot_distance / (rest_capacity + 1e-6)) + (rest_capacity / (distance + 1e-6)) - (distance / (rest_capacity + 1e-6)) + (1 / (demand + 1e-6))
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        return depot
    return next_node



# Function 8 - Score: -0.31506284169275367
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
    best_score = -1
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            score = (demand / (distance + 1e-6)) - (depot_distance / (rest_capacity + 1e-6)) + (rest_capacity / (distance + 1e-6)) + (1.0 / (demand + 1e-6))
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        return depot
    return next_node



# Function 9 - Score: -0.31506284169275367
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
    best_score = -1
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]

        if demand <= rest_capacity:
            score = (demand / (distance + 1e-6)) - (depot_distance / (rest_capacity + 1e-6)) + (rest_capacity / (distance + 1e-6)) + (1.0 / (demand + 1e-6))
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        return depot
    return next_node



# Function 10 - Score: -0.31517401029907094
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
            score = (demand / (distance + 1e-6)) - (0.5 * depot_distance / (rest_capacity + 1e-6)) + (rest_capacity / (distance + 1e-6)) - (0.05 * depot_distance) + (1.0 / (demand + 1e-6))
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        return depot
    return next_node



