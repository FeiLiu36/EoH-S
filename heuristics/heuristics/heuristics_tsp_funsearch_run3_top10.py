# Top 10 functions for funsearch run 3

# Function 1 - Score: -0.15178350447520644
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    distances_to_destination = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    progress_scores = distances_from_current - distances_to_destination
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    max_distances = np.max(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    combined_scores = 0.5 * progress_scores + 0.2 * min_distances + 0.2 * distances_to_destination - 0.1 * max_distances
    next_node = unvisited_nodes[np.argmin(combined_scores)]
    
    return next_node



# Function 2 - Score: -0.15406512072180084
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    distances_to_destination = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    progress_scores = distances_from_current - distances_to_destination
    proximity_scores = distances_from_current
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    max_distances = np.max(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    combined_scores = 0.4 * progress_scores + 0.3 * proximity_scores + 0.2 * min_distances - 0.1 * max_distances
    next_node = unvisited_nodes[np.argmin(combined_scores)]
    
    return next_node



# Function 3 - Score: -0.15406512072180084
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    distances_to_destination = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    progress_scores = distances_from_current - distances_to_destination
    proximity_scores = distances_from_current
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    max_distances = np.max(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    diversity_scores = max_distances - min_distances
    combined_scores = 0.4 * progress_scores + 0.3 * proximity_scores + 0.2 * min_distances - 0.1 * diversity_scores
    next_node = unvisited_nodes[np.argmin(combined_scores)]
    
    return next_node



# Function 4 - Score: -0.15406512072180084
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    distances_to_destination = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    progress_scores = distances_from_current - distances_to_destination
    proximity_scores = distances_from_current
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    max_distances = np.max(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    combined_scores = 0.4 * progress_scores + 0.3 * proximity_scores + 0.2 * min_distances - 0.1 * max_distances
    next_node = unvisited_nodes[np.argmin(combined_scores)]
    
    return next_node



# Function 5 - Score: -0.15481095570410874
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    distances_to_destination = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    progress_scores = distances_from_current - distances_to_destination
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    avg_distances = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    std_distances = np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    combined_scores = 0.6 * progress_scores + 0.2 * distances_from_current + 0.1 * min_distances + 0.1 * std_distances
    next_node = unvisited_nodes[np.argmin(combined_scores)]
    
    return next_node



# Function 6 - Score: -0.155441113073765
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    distances_to_destination = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    progress_scores = distances_from_current - distances_to_destination
    proximity_scores = distances_from_current
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    combined_scores = progress_scores + 0.4 * proximity_scores + 0.2 * min_distances
    next_node = unvisited_nodes[np.argmin(combined_scores)]
    
    return next_node



# Function 7 - Score: -0.15806335439337751
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    distances_to_destination = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    progress_scores = distances_from_current - distances_to_destination
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    max_distances = np.max(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    std_distances = np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    combined_scores = (
        0.4 * progress_scores + 
        0.2 * min_distances + 
        0.1 * distances_to_destination - 
        0.1 * max_distances + 
        0.2 * std_distances
    )
    next_node = unvisited_nodes[np.argmin(combined_scores)]
    
    return next_node



# Function 8 - Score: -0.15806335439337751
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    distances_to_destination = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    progress_scores = distances_from_current - distances_to_destination
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    max_distances = np.max(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    std_distances = np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    combined_scores = (
        0.4 * progress_scores + 
        0.2 * min_distances + 
        0.1 * distances_to_destination - 
        0.1 * max_distances + 
        0.2 * std_distances
    )
    next_node = unvisited_nodes[np.argmin(combined_scores)]
    
    return next_node



# Function 9 - Score: -0.15867063784948388
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    distances_to_destination = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    progress_scores = distances_from_current - distances_to_destination
    proximity_scores = distances_from_current
    combined_scores = progress_scores + 0.5 * proximity_scores
    next_node = unvisited_nodes[np.argmin(combined_scores)]
    
    return next_node



# Function 10 - Score: -0.15867063784948388
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    distances_to_destination = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    progress_scores = distances_from_current - distances_to_destination
    proximity_scores = distances_from_current
    exploration_scores = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    combined_scores = progress_scores + 0.5 * proximity_scores + 0.2 * exploration_scores
    next_node = unvisited_nodes[np.argmin(combined_scores)]
    
    return next_node



