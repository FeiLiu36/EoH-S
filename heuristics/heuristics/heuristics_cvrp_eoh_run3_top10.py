# Top 10 functions for eoh run 3

# Function 1 - Score: -0.249526916712335
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
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible_nodes:
        return depot
    
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    urgency = 1 - np.minimum(rest_capacity / (demands[feasible_nodes] + 1e-10), 1)
    
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    proximity_weight = 0.6 * np.exp(-2 * (1 - capacity_ratio))
    return_weight = 0.4 * (1 - np.exp(-2 * (1 - capacity_ratio)))
    urgency_weight = 0.1
    
    scores = proximity_weight * current_dists + return_weight * depot_dists + urgency_weight * urgency
    return feasible_nodes[np.argmin(scores)]



# Function 2 - Score: -0.2527188315564633
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
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible_nodes:
        return depot
    
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    urgency = 1 - np.minimum(rest_capacity / (demands[feasible_nodes] + 1e-10), 1)
    
    capacity_ratio = (rest_capacity / np.max(demands[feasible_nodes])) ** 2
    proximity_weight = 0.6 * capacity_ratio
    return_weight = 0.4 * (1 - capacity_ratio)
    urgency_weight = 0.2
    
    scores = proximity_weight * current_dists + return_weight * depot_dists + urgency_weight * urgency
    return feasible_nodes[np.argmin(scores)]



# Function 3 - Score: -0.2527188315564633
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
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible_nodes:
        return depot
    
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    urgency = 1 - np.minimum(rest_capacity / (demands[feasible_nodes] + 1e-10), 1)
    
    capacity_ratio = (rest_capacity / np.max(demands[feasible_nodes])) ** 2
    proximity_weight = 0.6 * capacity_ratio
    return_weight = 0.4 * (1 - capacity_ratio)
    
    scores = proximity_weight * current_dists + return_weight * depot_dists + 0.2 * urgency
    return feasible_nodes[np.argmin(scores)]



# Function 4 - Score: -0.25449306396756466
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
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible_nodes:
        return depot
    
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    urgency = 1 - np.minimum(rest_capacity / (demands[feasible_nodes] + 1e-10), 1)
    
    high_urgency = urgency > 0.7
    medium_urgency = (urgency > 0.3) & ~high_urgency
    low_urgency = ~high_urgency & ~medium_urgency
    
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    
    if any(high_urgency):
        tier_nodes = np.array(feasible_nodes)[high_urgency]
        scores = 0.4 * current_dists[high_urgency] + 0.2 * depot_dists[high_urgency] + 0.4 * urgency[high_urgency]
    elif any(medium_urgency):
        tier_nodes = np.array(feasible_nodes)[medium_urgency]
        scores = (0.2 + 0.4 * capacity_ratio) * current_dists[medium_urgency] + (0.4 - 0.2 * capacity_ratio) * depot_dists[medium_urgency] + 0.4 * urgency[medium_urgency]
    else:
        tier_nodes = np.array(feasible_nodes)[low_urgency]
        scores = (0.1 + 0.5 * capacity_ratio) * current_dists[low_urgency] + (0.6 - 0.5 * capacity_ratio) * depot_dists[low_urgency] + 0.3 * urgency[low_urgency]
    
    return tier_nodes[np.argmin(scores)]



# Function 5 - Score: -0.2545810137978189
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
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible_nodes:
        return depot
    
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    urgency = 1 - np.minimum(rest_capacity / (demands[feasible_nodes] + 1e-10), 1)
    
    high_urgency = urgency > 0.8
    medium_urgency = (urgency > 0.5) & ~high_urgency
    low_urgency = ~high_urgency & ~medium_urgency
    
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    
    if any(high_urgency):
        tier_nodes = np.array(feasible_nodes)[high_urgency]
        scores = 0.5 * current_dists[high_urgency] + 0.1 * depot_dists[high_urgency] + 0.4 * urgency[high_urgency]
    elif any(medium_urgency):
        tier_nodes = np.array(feasible_nodes)[medium_urgency]
        scores = (0.3 + 0.3 * capacity_ratio) * current_dists[medium_urgency] + (0.3 - 0.1 * capacity_ratio) * depot_dists[medium_urgency] + 0.4 * urgency[medium_urgency]
    else:
        tier_nodes = np.array(feasible_nodes)[low_urgency]
        scores = (0.1 + 0.4 * capacity_ratio) * current_dists[low_urgency] + (0.5 - 0.4 * capacity_ratio) * depot_dists[low_urgency] + 0.4 * urgency[low_urgency]
    
    return tier_nodes[np.argmin(scores)]



# Function 6 - Score: -0.2547325232863
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
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible_nodes:
        return depot
    
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    urgency = 1 - np.minimum(rest_capacity / (demands[feasible_nodes] + 1e-10), 1)
    
    high_urgency = urgency > 0.7
    medium_urgency = (urgency > 0.3) & ~high_urgency
    low_urgency = ~high_urgency & ~medium_urgency
    
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    
    if any(high_urgency):
        tier_nodes = np.array(feasible_nodes)[high_urgency]
        scores = 0.6 * current_dists[high_urgency] + 0.2 * depot_dists[high_urgency] + 0.2 * urgency[high_urgency]
    elif any(medium_urgency):
        tier_nodes = np.array(feasible_nodes)[medium_urgency]
        scores = (0.4 + 0.2 * capacity_ratio) * current_dists[medium_urgency] + (0.4 - 0.2 * capacity_ratio) * depot_dists[medium_urgency] + 0.2 * urgency[medium_urgency]
    else:
        tier_nodes = np.array(feasible_nodes)[low_urgency]
        scores = (0.2 + 0.3 * capacity_ratio) * current_dists[low_urgency] + (0.5 - 0.3 * capacity_ratio) * depot_dists[low_urgency] + 0.3 * urgency[low_urgency]
    
    return tier_nodes[np.argmin(scores)]



# Function 7 - Score: -0.2568450636376274
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
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible_nodes:
        return depot
    
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    urgency = 1 - np.minimum(rest_capacity / (demands[feasible_nodes] + 1e-10), 1)
    
    high_urgency = urgency > 0.7
    medium_urgency = (urgency > 0.3) & ~high_urgency
    low_urgency = ~high_urgency & ~medium_urgency
    
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    
    if any(high_urgency):
        tier_nodes = np.array(feasible_nodes)[high_urgency]
        scores = 0.6 * current_dists[high_urgency] + 0.2 * depot_dists[high_urgency] + 0.2 * urgency[high_urgency]
    elif any(medium_urgency):
        tier_nodes = np.array(feasible_nodes)[medium_urgency]
        scores = (0.4 + 0.2 * capacity_ratio) * current_dists[medium_urgency] + (0.4 - 0.2 * capacity_ratio) * depot_dists[medium_urgency] + 0.2 * urgency[medium_urgency]
    else:
        tier_nodes = np.array(feasible_nodes)[low_urgency]
        scores = (0.2 + 0.3 * capacity_ratio) * current_dists[low_urgency] + (0.6 - 0.3 * capacity_ratio) * depot_dists[low_urgency] + 0.2 * urgency[low_urgency]
    
    return tier_nodes[np.argmin(scores)]



# Function 8 - Score: -0.2568450636376274
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
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible_nodes:
        return depot
    
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    urgency = 1 - np.minimum(rest_capacity / (demands[feasible_nodes] + 1e-10), 1)
    
    high_urgency = urgency > 0.7
    medium_urgency = (urgency > 0.3) & ~high_urgency
    low_urgency = ~high_urgency & ~medium_urgency
    
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    
    if any(high_urgency):
        tier_nodes = np.array(feasible_nodes)[high_urgency]
        scores = 0.6 * current_dists[high_urgency] + 0.2 * depot_dists[high_urgency] + 0.2 * urgency[high_urgency]
    elif any(medium_urgency):
        tier_nodes = np.array(feasible_nodes)[medium_urgency]
        scores = (0.4 + 0.2 * capacity_ratio) * current_dists[medium_urgency] + (0.4 - 0.2 * capacity_ratio) * depot_dists[medium_urgency] + 0.2 * urgency[medium_urgency]
    else:
        tier_nodes = np.array(feasible_nodes)[low_urgency]
        scores = (0.2 + 0.3 * capacity_ratio) * current_dists[low_urgency] + (0.6 - 0.3 * capacity_ratio) * depot_dists[low_urgency] + 0.2 * urgency[low_urgency]
    
    return tier_nodes[np.argmin(scores)]



# Function 9 - Score: -0.2572140311978424
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
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible_nodes:
        return depot
    
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    urgency = 1 - np.minimum(rest_capacity / (demands[feasible_nodes] + 1e-10), 1)
    
    high_urgency = urgency > 0.7
    medium_urgency = (urgency > 0.3) & ~high_urgency
    
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    
    if any(high_urgency):
        tier_nodes = np.array(feasible_nodes)[high_urgency]
        scores = 0.8 * current_dists[high_urgency] + 0.2 * depot_dists[high_urgency]
    elif any(medium_urgency):
        tier_nodes = np.array(feasible_nodes)[medium_urgency]
        scores = (0.6 + 0.2 * capacity_ratio) * current_dists[medium_urgency] + (0.4 - 0.2 * capacity_ratio) * depot_dists[medium_urgency]
    else:
        tier_nodes = np.array(feasible_nodes)
        scores = (0.4 + 0.4 * capacity_ratio) * current_dists + (0.6 - 0.4 * capacity_ratio) * depot_dists
    
    return tier_nodes[np.argmin(scores)]



# Function 10 - Score: -0.2633201145814183
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
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible_nodes:
        return depot
    
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    urgency = 1 - np.minimum(rest_capacity / (demands[feasible_nodes] + 1e-10), 1)
    
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    proximity_weight = 0.6 * np.exp(-2 * (1 - capacity_ratio))
    return_weight = 0.3 * (1 - np.exp(-2 * (1 - capacity_ratio)))
    urgency_weight = 0.1 + 0.2 * (1 - capacity_ratio)
    
    scores = proximity_weight * current_dists + return_weight * depot_dists + urgency_weight * urgency
    return feasible_nodes[np.argmin(scores)]



