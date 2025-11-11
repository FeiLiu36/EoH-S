# Top 10 functions for funsearch run 2

# Function 1 - Score: -0.27850795534719075
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

    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot

    current_to_nodes = distance_matrix[current_node][feasible_nodes]
    nodes_to_depot = distance_matrix[feasible_nodes, depot]
    node_demands = demands[feasible_nodes]
    
    proximity = 1 / (current_to_nodes + 1e-6)
    urgency = 1 / (1 + np.sum(demands[unvisited_nodes] <= rest_capacity))
    cluster_factor = 1 / (1 + np.sum(distance_matrix[feasible_nodes][:, unvisited_nodes] < np.mean(distance_matrix), axis=1))
    return_factor = 1 / (1 + nodes_to_depot)
    capacity_ratio = rest_capacity / (node_demands + 1e-6)
    efficiency = node_demands / (current_to_nodes + 1e-6)
    centrality = np.sum(distance_matrix[feasible_nodes], axis=1) / (np.sum(distance_matrix) + 1e-6)
    time_window = 1 / (1 + np.abs(rest_capacity - node_demands))
    
    scores = proximity * urgency * cluster_factor * return_factor * capacity_ratio * efficiency * centrality * time_window
    best_idx = np.argmax(scores)
    return feasible_nodes[best_idx]



# Function 2 - Score: -0.28056009510680835
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
        remaining_capacity_ratio = rest_capacity / (demand + 1e-6)
        urgency = 1 / (1 + np.sum(demands[unvisited_nodes] <= rest_capacity))
        cluster_penalty = 1 / (1 + np.sum(distance_matrix[node][unvisited_nodes] < np.mean(distance_matrix)))
        time_window = 1 / (1 + np.abs(rest_capacity - demand))
        centrality = np.sum(distance_matrix[node]) / (np.sum(distance_matrix) + 1e-6)
        proximity = 1 / (1 + np.mean(distance_matrix[node, unvisited_nodes]))
        
        if demand <= rest_capacity:
            efficiency = demand / (distance + 1e-6)
            urgency_factor = 1 / (1 + distance + depot_distance)
            score = efficiency * urgency_factor * remaining_capacity_ratio * urgency * cluster_penalty * time_window * centrality * proximity
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        next_node = depot

    return next_node



# Function 3 - Score: -0.28056009510680835
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

    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot

    current_to_nodes = distance_matrix[current_node][feasible_nodes]
    nodes_to_depot = distance_matrix[feasible_nodes, depot]
    node_demands = demands[feasible_nodes]
    remaining_capacity_ratio = rest_capacity / (node_demands + 1e-6)
    urgency = 1 / (1 + np.sum(demands[unvisited_nodes] <= rest_capacity))
    cluster_penalty = 1 / (1 + np.sum(distance_matrix[feasible_nodes][:, unvisited_nodes] < np.mean(distance_matrix), axis=1))
    time_window = 1 / (1 + np.abs(rest_capacity - node_demands))
    centrality = np.sum(distance_matrix[feasible_nodes], axis=1) / (np.sum(distance_matrix) + 1e-6)
    proximity = 1 / (1 + np.mean(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1))
    efficiency = node_demands / (current_to_nodes + 1e-6)
    urgency_factor = 1 / (1 + current_to_nodes + nodes_to_depot)

    scores = efficiency * urgency_factor * remaining_capacity_ratio * urgency * cluster_penalty * time_window * centrality * proximity
    best_idx = np.argmax(scores)
    return feasible_nodes[best_idx]



# Function 4 - Score: -0.29000180721663993
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
        remaining_capacity_ratio = rest_capacity / (demand + 1e-6)
        urgency = 1 / (1 + np.sum(demands[unvisited_nodes] <= rest_capacity))
        cluster_penalty = 1 / (1 + np.sum(distance_matrix[node][unvisited_nodes] < np.mean(distance_matrix)))
        time_window = 1 / (1 + np.abs(rest_capacity - demand))
        centrality = np.sum(distance_matrix[node]) / (np.sum(distance_matrix) + 1e-6)

        if demand <= rest_capacity:
            score = (demand / (distance + 1e-6)) * (1 / (1 + depot_distance)) * remaining_capacity_ratio * urgency * cluster_penalty * time_window * centrality
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        next_node = depot

    return next_node



# Function 5 - Score: -0.2944256489707571
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
        remaining_capacity_ratio = rest_capacity / (demand + 1e-6)
        urgency = 1 / (1 + np.sum(demands[unvisited_nodes] <= rest_capacity))
        cluster_penalty = 1 / (1 + np.sum(distance_matrix[node][unvisited_nodes] < np.mean(distance_matrix)))
        time_window = 1 / (1 + np.abs(rest_capacity - demand))

        if demand <= rest_capacity:
            score = (demand / (distance + 1e-6)) * (1 / (1 + depot_distance)) * remaining_capacity_ratio * urgency * cluster_penalty * time_window
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        next_node = depot

    return next_node



# Function 6 - Score: -0.2946806518017029
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
            remaining_capacity = rest_capacity - demand
            if distance > 0:
                urgency = (demand / (distance + 1e-6)) * (1 + (demand / (rest_capacity + 1e-6)))
                efficiency = (remaining_capacity * distance_matrix[depot][node]) / (distance + 1e-6)
                proximity = 1 / (distance + 1e-6)
                return_penalty = depot_distance / (remaining_capacity + 1)
                capacity_ratio = remaining_capacity / (rest_capacity + 1e-6)
                demand_ratio = demand / (np.sum(demands[unvisited_nodes]) + 1e-6)
                
                score = (0.5 * urgency + 0.3 * efficiency + 0.2 * proximity - 0.1 * return_penalty) * (1 + capacity_ratio - demand_ratio)
            else:
                score = float('inf')
            
            if score > best_score:
                best_score = score
                next_node = node

    return next_node if next_node != -1 else depot



# Function 7 - Score: -0.29852695419313796
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
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    
    if len(feasible_nodes) == 0:
        return depot
        
    for node in feasible_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]
        depot_distance = distance_matrix[node][depot]
        remaining_capacity_ratio = rest_capacity / (demand + 1e-6)
        urgency = 1 / (1 + np.sum(demands[feasible_nodes] <= rest_capacity - demand))
        cluster_penalty = 1 / (1 + np.sum(distance_matrix[node][feasible_nodes] < np.mean(distance_matrix)))
        time_window_penalty = 1 / (1 + np.abs(distance - np.mean(distance_matrix[current_node, feasible_nodes])))
        savings = distance_matrix[current_node][depot] + distance_matrix[depot][node] - distance_matrix[current_node][node]
        proximity = np.exp(-distance / (np.mean(distance_matrix[current_node, feasible_nodes]) + 1e-6))
        
        score = (demand / (distance + 1e-6)) * (1 / (1 + depot_distance)) * remaining_capacity_ratio * urgency * cluster_penalty * time_window_penalty * (1 + savings) * proximity
        
        if score > best_score:
            best_score = score
            next_node = node

    return next_node if next_node != -1 else depot



# Function 8 - Score: -0.30163253355971886
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
        remaining_capacity_ratio = rest_capacity / (demand + 1e-6)
        urgency = 1 / (1 + np.sum(demands[unvisited_nodes] <= rest_capacity))
        cluster_penalty = 1 / (1 + np.sum(distance_matrix[node][unvisited_nodes] < np.mean(distance_matrix)))
        time_window = 1 / (1 + np.abs(rest_capacity - demand))
        centrality = 1 / (1 + np.mean(distance_matrix[node]))

        if demand <= rest_capacity:
            score = (demand / (distance + 1e-6)) * (1 / (1 + depot_distance)) * remaining_capacity_ratio * urgency * cluster_penalty * time_window * centrality
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        next_node = depot

    return next_node



# Function 9 - Score: -0.30163253355971886
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
        remaining_capacity_ratio = rest_capacity / (demand + 1e-6)
        urgency = 1 / (1 + np.sum(demands[unvisited_nodes] <= rest_capacity))
        cluster_penalty = 1 / (1 + np.sum(distance_matrix[node][unvisited_nodes] < np.mean(distance_matrix)))
        time_window = 1 / (1 + np.abs(rest_capacity - demand))
        demand_ratio = demand / (np.max(demands[unvisited_nodes]) + 1e-6)
        centrality = 1 / (1 + np.mean(distance_matrix[node]))

        if demand <= rest_capacity:
            score = (demand_ratio / (distance + 1e-6)) * (1 / (1 + depot_distance)) * remaining_capacity_ratio * urgency * cluster_penalty * time_window * centrality
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        next_node = depot

    return next_node



# Function 10 - Score: -0.30163253355971886
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
        remaining_capacity_ratio = rest_capacity / (demand + 1e-6)
        urgency = 1 / (1 + np.sum(demands[unvisited_nodes] <= rest_capacity))
        cluster_penalty = 1 / (1 + np.sum(distance_matrix[node][unvisited_nodes] < np.mean(distance_matrix)))
        time_window = 1 / (1 + np.abs(rest_capacity - demand))
        centrality = 1 / (1 + np.mean(distance_matrix[node]))

        if demand <= rest_capacity:
            score = (demand / (distance + 1e-6)) * (1 / (1 + depot_distance)) * remaining_capacity_ratio * urgency * cluster_penalty * time_window * centrality
            if score > best_score:
                best_score = score
                next_node = node

    if next_node == -1:
        next_node = depot

    return next_node



