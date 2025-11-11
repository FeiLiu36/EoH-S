# Top 10 functions for eoh run 2

# Function 1 - Score: -0.3039057047670427
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
    
    distances = distance_matrix[current_node, feasible_nodes]
    distance_score = 1 / (distances + 1e-6)
    
    urgency = (demands[feasible_nodes] / (rest_capacity + 1e-6)) ** 2
    capacity_score = np.where(urgency > 0.4, urgency * 2.5, urgency)
    
    cluster_density = np.sum(distance_matrix[feasible_nodes] < np.median(distance_matrix), axis=1)
    density_score = 1 - (cluster_density / np.max(cluster_density + 1e-6))
    
    capacity_pressure = 1 - (rest_capacity / np.max(rest_capacity))
    w_dist = 0.35 - (0.15 * capacity_pressure)
    w_cap = 0.35 + (0.15 * capacity_pressure)
    w_den = 0.3 + (0.1 * (1 - capacity_pressure))
    
    scores = w_dist * distance_score + w_cap * capacity_score + w_den * density_score
    return feasible_nodes[np.argmax(scores)]



# Function 2 - Score: -0.30427856915549656
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
    
    distances = distance_matrix[current_node, feasible_nodes]
    distance_score = 1 / (distances + 1e-6)
    
    urgency = (demands[feasible_nodes] / (rest_capacity + 1e-6)) ** 2
    capacity_score = np.where(urgency > 0.5, urgency * 2, urgency)
    
    centrality = np.mean(distance_matrix[feasible_nodes], axis=1)
    centrality_score = centrality / np.max(centrality + 1e-6)
    
    capacity_pressure = 1 - (rest_capacity / np.max(rest_capacity))
    w_dist = 0.4 - (0.2 * capacity_pressure)
    w_cap = 0.3 + (0.1 * capacity_pressure)
    w_cen = 0.3 + (0.1 * capacity_pressure)
    
    scores = w_dist * distance_score + w_cap * capacity_score + w_cen * centrality_score
    return feasible_nodes[np.argmax(scores)]



# Function 3 - Score: -0.305830976357802
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
    
    distances = distance_matrix[current_node, feasible_nodes]
    distance_score = 1 / (distances + 1e-6)
    
    urgency = (demands[feasible_nodes] / (rest_capacity + 1e-6)) ** 2
    capacity_score = np.where(urgency > 0.3, urgency * 3, urgency)
    
    isolation = np.max(distance_matrix[feasible_nodes], axis=1) - np.min(distance_matrix[feasible_nodes], axis=1)
    isolation_score = isolation / np.max(isolation + 1e-6)
    
    capacity_pressure = 1 - (rest_capacity / np.max(rest_capacity))
    w_dist = 0.3 - (0.1 * capacity_pressure)
    w_cap = 0.4 + (0.2 * capacity_pressure)
    w_iso = 0.3 + (0.1 * capacity_pressure)
    
    scores = w_dist * distance_score + w_cap * capacity_score + w_iso * isolation_score
    return feasible_nodes[np.argmax(scores)]



# Function 4 - Score: -0.3060891656957853
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
    
    distances = distance_matrix[current_node, feasible_nodes]
    distance_score = 1 / (distances + 1e-6)
    
    urgency = (demands[feasible_nodes] / (rest_capacity + 1e-6)) ** 2.0
    capacity_score = np.where(urgency > 0.6, urgency * 3, urgency)
    
    density = np.sum(1 / (distance_matrix[feasible_nodes] + 1e-6), axis=1)
    density_score = density / np.max(density)
    
    time_pressure = np.exp(len(unvisited_nodes) / len(distance_matrix)) - 1
    w_dist = 0.5 - (0.15 * time_pressure)
    w_cap = 0.4 + (0.15 * time_pressure)
    w_den = 0.1
    
    scores = w_dist * distance_score + w_cap * capacity_score + w_den * density_score
    return feasible_nodes[np.argmax(scores)]



# Function 5 - Score: -0.30614822828135335
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
    
    distances = distance_matrix[current_node, feasible_nodes]
    distance_score = 1 / (distances + 1e-6)
    
    urgency = (demands[feasible_nodes] / (rest_capacity + 1e-6)) ** 2
    capacity_score = np.where(urgency > 0.7, urgency * 3, urgency)
    
    density = np.sum(1 / (distance_matrix[feasible_nodes] + 1e-6), axis=1)
    density_score = density / np.max(density)
    
    time_pressure = (len(unvisited_nodes) / len(distance_matrix)) ** 2
    w_dist = 0.5 - (0.3 * time_pressure)
    w_cap = 0.4 + (0.3 * time_pressure)
    w_den = 0.1
    
    scores = w_dist * distance_score + w_cap * capacity_score + w_den * density_score
    return feasible_nodes[np.argmax(scores)]



# Function 6 - Score: -0.3069378956154998
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
    
    distances = distance_matrix[current_node, feasible_nodes]
    distance_score = 1 / (distances + 1e-6)
    
    urgency = (demands[feasible_nodes] / (rest_capacity + 1e-6)) ** 1.5
    capacity_score = np.where(urgency > 0.8, urgency * 4, urgency)
    
    density = np.sum(1 / (distance_matrix[feasible_nodes] + 1e-6), axis=1)
    density_score = density / np.max(density)
    
    time_pressure = np.log(len(unvisited_nodes) / len(distance_matrix) + 1)
    w_dist = 0.6 - (0.2 * time_pressure)
    w_cap = 0.3 + (0.2 * time_pressure)
    w_den = 0.1
    
    scores = w_dist * distance_score + w_cap * capacity_score + w_den * density_score
    return feasible_nodes[np.argmax(scores)]



# Function 7 - Score: -0.3069378956154998
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
    
    distances = distance_matrix[current_node, feasible_nodes]
    distance_score = 1 / (distances + 1e-6)
    
    urgency = (demands[feasible_nodes] / (rest_capacity + 1e-6)) ** 1.5
    capacity_score = np.where(urgency > 0.8, urgency * 4, urgency)
    
    density = np.sum(1 / (distance_matrix[feasible_nodes] + 1e-6), axis=1)
    density_score = density / np.max(density)
    
    time_pressure = np.log(len(unvisited_nodes) / len(distance_matrix) + 1)
    w_dist = 0.6 - (0.2 * time_pressure)
    w_cap = 0.3 + (0.2 * time_pressure)
    w_den = 0.1
    
    scores = w_dist * distance_score + w_cap * capacity_score + w_den * density_score
    return feasible_nodes[np.argmax(scores)]



# Function 8 - Score: -0.3069378956154998
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
    
    distances = distance_matrix[current_node, feasible_nodes]
    distance_score = 1 / (distances + 1e-6)
    
    urgency = (demands[feasible_nodes] / (rest_capacity + 1e-6)) ** 1.5
    capacity_score = np.where(urgency > 0.8, urgency * 4, urgency)
    
    density = np.sum(1 / (distance_matrix[feasible_nodes] + 1e-6), axis=1)
    density_score = density / np.max(density)
    
    time_pressure = np.log(len(unvisited_nodes) / len(distance_matrix) + 1)
    w_dist = 0.6 - (0.2 * time_pressure)
    w_cap = 0.3 + (0.2 * time_pressure)
    w_den = 0.1
    
    scores = w_dist * distance_score + w_cap * capacity_score + w_den * density_score
    return feasible_nodes[np.argmax(scores)]



# Function 9 - Score: -0.30698206590696175
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
    
    distances = distance_matrix[current_node, feasible_nodes]
    distance_score = 1 / (distances + 1e-6)
    
    urgency = (demands[feasible_nodes] / (rest_capacity + 1e-6)) ** 2.0
    capacity_score = np.where(urgency > 0.7, urgency * 3, urgency)
    
    density = np.sum(1 / (distance_matrix[feasible_nodes] + 1e-6), axis=1)
    density_score = density / np.max(density)
    
    time_pressure = len(unvisited_nodes) / len(distance_matrix)
    w_dist = 0.5 - (0.3 * time_pressure)
    w_cap = 0.4 + (0.3 * time_pressure)
    w_den = 0.1
    
    scores = w_dist * distance_score + w_cap * capacity_score + w_den * density_score
    return feasible_nodes[np.argmax(scores)]



# Function 10 - Score: -0.30698206590696175
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
    
    distances = distance_matrix[current_node, feasible_nodes]
    distance_score = 1 / (distances + 1e-6)
    
    urgency = (demands[feasible_nodes] / (rest_capacity + 1e-6)) ** 2.0
    capacity_score = np.where(urgency > 0.7, urgency * 3, urgency)
    
    density = np.sum(1 / (distance_matrix[feasible_nodes] + 1e-6), axis=1)
    density_score = density / np.max(density)
    
    time_pressure = len(unvisited_nodes) / len(distance_matrix)
    w_dist = 0.5 - (0.3 * time_pressure)
    w_cap = 0.4 + (0.3 * time_pressure)
    w_den = 0.1
    
    scores = w_dist * distance_score + w_cap * capacity_score + w_den * density_score
    return feasible_nodes[np.argmax(scores)]



