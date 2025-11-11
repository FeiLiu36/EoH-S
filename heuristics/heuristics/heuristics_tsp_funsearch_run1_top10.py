# Top 10 functions for funsearch run 1

# Function 1 - Score: -0.15676277526859383
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
    unvisited_nodes = np.array(unvisited_nodes)
    current_to_nodes = distance_matrix[current_node, unvisited_nodes]
    nodes_to_dest = distance_matrix[unvisited_nodes, destination_node]
    progress = current_to_nodes - nodes_to_dest
    normalized_current = current_to_nodes / np.max(current_to_nodes)
    normalized_progress = progress / np.max(np.abs(progress))
    weights = 0.4 * normalized_current + 0.6 * normalized_progress
    min_idx = np.argmin(weights)
    return unvisited_nodes[min_idx]



# Function 2 - Score: -0.15822602794542923
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
    if destination_node in unvisited_nodes:
        return destination_node
    
    current_to_nodes = distance_matrix[current_node, unvisited_nodes]
    nodes_to_dest = distance_matrix[unvisited_nodes, destination_node]
    progress = current_to_nodes - nodes_to_dest
    
    norm_current = current_to_nodes / (np.max(current_to_nodes) + 1e-8)
    norm_progress = progress / (np.max(np.abs(progress)) + 1e-8)
    
    weights = 0.5 * norm_current + 0.5 * norm_progress
    next_node_idx = np.argmin(weights)
    return unvisited_nodes[next_node_idx]



# Function 3 - Score: -0.1585210254668393
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
    if destination_node in unvisited_nodes:
        return destination_node
    
    unvisited_nodes = np.array(unvisited_nodes)
    current_to_nodes = distance_matrix[current_node, unvisited_nodes]
    nodes_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Combine distance and progress with dynamic weighting
    progress = current_to_nodes - nodes_to_dest
    normalized_distance = current_to_nodes / (np.max(current_to_nodes) + 1e-8)
    normalized_progress = progress / (np.max(np.abs(progress)) + 1e-8)
    
    # Dynamic weights based on current progress
    distance_weight = 0.7 if np.min(nodes_to_dest) > np.median(distance_matrix) else 0.5
    progress_weight = 1 - distance_weight
    
    weights = distance_weight * normalized_distance + progress_weight * normalized_progress
    min_idx = np.argmin(weights)
    return unvisited_nodes[min_idx]



# Function 4 - Score: -0.15878176835355862
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
    if destination_node in unvisited_nodes:
        return destination_node
    
    unvisited_nodes = np.array(unvisited_nodes)
    current_to_nodes = distance_matrix[current_node, unvisited_nodes]
    nodes_to_dest = distance_matrix[unvisited_nodes, destination_node]
    progress = current_to_nodes - nodes_to_dest
    
    normalized_distance = current_to_nodes / (np.sum(current_to_nodes) + 1e-8)
    normalized_progress = progress / (np.sum(np.abs(progress)) + 1e-8)
    
    weights = 0.4 * normalized_distance + 0.6 * normalized_progress
    min_idx = np.argmin(weights)
    return unvisited_nodes[min_idx]



# Function 5 - Score: -0.15878176835355862
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
    if destination_node in unvisited_nodes:
        return destination_node
    
    unvisited_nodes = np.array(unvisited_nodes)
    current_to_nodes = distance_matrix[current_node, unvisited_nodes]
    nodes_to_dest = distance_matrix[unvisited_nodes, destination_node]
    progress = current_to_nodes - nodes_to_dest
    
    normalized_distance = current_to_nodes / (np.sum(current_to_nodes) + 1e-8)
    normalized_progress = progress / (np.sum(np.abs(progress)) + 1e-8)
    
    weights = 0.4 * normalized_distance + 0.6 * normalized_progress
    min_idx = np.argmin(weights)
    return unvisited_nodes[min_idx]



# Function 6 - Score: -0.1591659280905549
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
    if destination_node in unvisited_nodes:
        return destination_node
    
    unvisited_nodes = np.array(unvisited_nodes)
    current_to_nodes = distance_matrix[current_node, unvisited_nodes]
    nodes_to_dest = distance_matrix[unvisited_nodes, destination_node]
    progress = current_to_nodes - nodes_to_dest
    
    normalized_distance = np.log1p(current_to_nodes) / (np.log1p(np.max(current_to_nodes)) + 1e-8)
    normalized_progress = np.tanh(progress / (np.max(np.abs(progress)) + 1e-8))
    
    weights = 0.4 * normalized_distance + 0.6 * normalized_progress
    min_idx = np.argmin(weights)
    return unvisited_nodes[min_idx]



# Function 7 - Score: -0.15926171208713097
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
    unvisited_nodes = np.array(unvisited_nodes)
    if destination_node in unvisited_nodes:
        return destination_node
    
    current_to_nodes = distance_matrix[current_node, unvisited_nodes]
    nodes_to_dest = distance_matrix[unvisited_nodes, destination_node]
    progress = current_to_nodes - nodes_to_dest
    normalized_current = current_to_nodes / (np.max(current_to_nodes) + 1e-8)
    normalized_progress = progress / (np.max(np.abs(progress)) + 1e-8)
    normalized_dest = nodes_to_dest / (np.max(nodes_to_dest) + 1e-8)
    weights = 0.5 * normalized_current + 0.3 * normalized_progress + 0.2 * (1 - normalized_dest)
    min_idx = np.argmin(weights)
    return unvisited_nodes[min_idx]



# Function 8 - Score: -0.15926171208713097
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
    unvisited_nodes = np.array(unvisited_nodes)
    if destination_node in unvisited_nodes:
        return destination_node
    
    current_to_nodes = distance_matrix[current_node, unvisited_nodes]
    nodes_to_dest = distance_matrix[unvisited_nodes, destination_node]
    progress = current_to_nodes - nodes_to_dest
    normalized_current = current_to_nodes / (np.max(current_to_nodes) + 1e-8)
    normalized_progress = progress / (np.max(np.abs(progress)) + 1e-8)
    normalized_dest = nodes_to_dest / (np.max(nodes_to_dest) + 1e-8)
    weights = 0.5 * normalized_current + 0.3 * normalized_progress + 0.2 * (1 - normalized_dest)
    min_idx = np.argmin(weights)
    return unvisited_nodes[min_idx]



# Function 9 - Score: -0.16288388145781307
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
    if destination_node in unvisited_nodes:
        return destination_node
    
    current_to_nodes = distance_matrix[current_node, unvisited_nodes]
    nodes_to_dest = distance_matrix[unvisited_nodes, destination_node]
    total_path = current_to_nodes + nodes_to_dest
    progress = current_to_nodes - nodes_to_dest
    
    norm_current = current_to_nodes / np.max(current_to_nodes)
    norm_progress = progress / (np.max(np.abs(progress)) + 1e-8)
    norm_total = total_path / np.max(total_path)
    
    weights = 0.3 * norm_current + 0.5 * norm_progress + 0.2 * norm_total
    next_node_idx = np.argmin(weights)
    return unvisited_nodes[next_node_idx]



# Function 10 - Score: -0.1649570769885283
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
    unvisited_nodes = np.array(unvisited_nodes)
    current_to_nodes = distance_matrix[current_node, unvisited_nodes]
    nodes_to_dest = distance_matrix[unvisited_nodes, destination_node]
    progress = current_to_nodes - nodes_to_dest
    normalized_current = current_to_nodes / (np.max(current_to_nodes) + 1e-8)
    normalized_progress = progress / (np.max(np.abs(progress)) + 1e-8)
    weights = 0.3 * normalized_current + 0.7 * normalized_progress
    min_idx = np.argmin(weights)
    return unvisited_nodes[min_idx]



