# Top 10 functions for funsearch run 2

# Function 1 - Score: -0.15997033856170967
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
    
    distances_to_dest = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    progress = distances_from_current - distances_to_dest
    normalized_progress = progress / (np.max(np.abs(progress)) + 1e-8)
    
    cluster_tendency = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    normalized_cluster = cluster_tendency / (np.max(cluster_tendency) + 1e-8)
    
    alpha = 0.4 - 0.1 * (np.max(distances_to_dest) / (np.max(distance_matrix) + 1e-8))
    beta = 0.3 + 0.1 * (np.min(distances_to_dest) / (np.max(distances_to_dest) + 1e-8))
    gamma = 0.2 + 0.1 * (np.mean(min_distances) / (np.max(min_distances) + 1e-8))
    delta = 0.1 * (1 - normalized_cluster)
    
    combined_score = alpha * distances_from_current + beta * distances_to_dest + gamma * min_distances + 0.2 * normalized_progress + delta
    next_node_idx = np.argmin(combined_score)
    next_node = unvisited_nodes[next_node_idx]
    
    return next_node



# Function 2 - Score: -0.16060827337447223
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
    
    distances_to_dest = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    progress = distances_from_current - distances_to_dest
    normalized_progress = progress / (np.max(np.abs(progress)) + 1e-8)
    
    alpha = 0.4 - 0.2 * (np.max(distances_to_dest) / (np.max(distance_matrix) + 1e-8))
    beta = 0.3 + 0.2 * (np.min(distances_to_dest) / (np.max(distances_to_dest) + 1e-8))
    gamma = 0.3 + 0.1 * (np.mean(min_distances) / (np.max(min_distances) + 1e-8))
    
    local_density = np.mean(distance_matrix[unvisited_nodes], axis=1)
    normalized_density = local_density / (np.max(local_density) + 1e-8)
    
    combined_score = (
        alpha * distances_from_current + 
        beta * distances_to_dest + 
        gamma * min_distances + 
        0.2 * normalized_progress - 
        0.1 * normalized_density
    )
    next_node_idx = np.argmin(combined_score)
    next_node = unvisited_nodes[next_node_idx]
    
    return next_node



# Function 3 - Score: -0.17267850463424117
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
    distances_to_dest = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    combined_scores = distances_from_current - 0.5 * distances_to_dest
    next_node_idx = np.argmin(combined_scores)
    return unvisited_nodes[next_node_idx]



# Function 4 - Score: -0.17733063822496453
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
    distances_to_dest = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    combined_scores = 0.7 * distances_from_current - 0.3 * distances_to_dest
    next_node_idx = np.argmin(combined_scores)
    return unvisited_nodes[next_node_idx]



# Function 5 - Score: -0.17733063822496453
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
    distances_to_dest = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    combined_scores = 0.7 * distances_from_current - 0.3 * distances_to_dest
    next_node_idx = np.argmin(combined_scores)
    return unvisited_nodes[next_node_idx]



# Function 6 - Score: -0.17733063822496453
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
    distances_to_dest = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    combined_scores = 0.7 * distances_from_current - 0.3 * distances_to_dest
    next_node_idx = np.argmin(combined_scores)
    return unvisited_nodes[next_node_idx]



# Function 7 - Score: -0.1781917450402322
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
    
    distances_to_dest = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    progress = distances_from_current - distances_to_dest
    normalized_progress = progress / (np.max(np.abs(progress)) + 1e-8)
    
    alpha = 0.4 - 0.2 * (np.max(distances_to_dest) / (np.max(distance_matrix) + 1e-8))
    beta = 0.3 + 0.2 * (np.min(distances_to_dest) / (np.max(distances_to_dest) + 1e-8))
    gamma = 0.3 + 0.1 * (np.mean(min_distances) / (np.max(min_distances) + 1e-8))
    
    clustering_coeff = np.sum(distance_matrix[unvisited_nodes][:, unvisited_nodes] < np.mean(distance_matrix), axis=1)
    normalized_clustering = clustering_coeff / (np.max(clustering_coeff) + 1e-8)
    
    combined_score = (
        alpha * distances_from_current + 
        beta * distances_to_dest + 
        gamma * min_distances + 
        0.2 * normalized_progress + 
        0.1 * normalized_clustering
    )
    next_node_idx = np.argmin(combined_score)
    next_node = unvisited_nodes[next_node_idx]
    
    return next_node



# Function 8 - Score: -0.18899118888192928
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
    
    distances_to_dest = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    progress = distances_from_current - distances_to_dest
    normalized_progress = progress / (np.max(np.abs(progress)) + 1e-8)
    
    alpha = 0.4 - 0.2 * (np.max(distances_to_dest) / (np.max(distance_matrix) + 1e-8))
    beta = 0.3 + 0.2 * (np.min(distances_to_dest) / (np.max(distances_to_dest) + 1e-8))
    gamma = 0.3 + 0.1 * (np.mean(min_distances) / (np.max(min_distances) + 1e-8))
    
    combined_score = alpha * distances_from_current + beta * distances_to_dest + gamma * min_distances + 0.2 * normalized_progress
    next_node_idx = np.argmin(combined_score)
    next_node = unvisited_nodes[next_node_idx]
    
    return next_node



# Function 9 - Score: -0.1940728932963247
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
    min_score = float('inf')
    next_node = unvisited_nodes[0]
    for node in unvisited_nodes:
        current_to_node = distance_matrix[current_node][node]
        node_to_dest = distance_matrix[node][destination_node]
        avg_dist_unvisited = np.mean(distance_matrix[node][unvisited_nodes])
        degree = np.sum(distance_matrix[node] < np.inf)
        centrality = 1 / (np.mean(distance_matrix[node]) + 1e-6)
        score = 0.3 * current_to_node + 0.3 * node_to_dest + 0.2 * (current_to_node / (node_to_dest + 1e-6)) + 0.1 * avg_dist_unvisited + 0.1 * centrality
        if score < min_score:
            min_score = score
            next_node = node
    return next_node



# Function 10 - Score: -0.19414858674938096
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
    
    distances_to_dest = distance_matrix[unvisited_nodes, destination_node]
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    min_distances = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    progress = distances_from_current - distances_to_dest
    normalized_progress = progress / (np.max(np.abs(progress)) + 1e-8)
    
    alpha = 0.5 - 0.2 * (np.max(distances_to_dest) / (np.max(distance_matrix) + 1e-8))
    beta = 0.3 + 0.1 * (np.min(distances_to_dest) / (np.max(distances_to_dest) + 1e-8))
    gamma = 0.2 + 0.1 * (np.mean(min_distances) / (np.max(min_distances) + 1e-8))
    
    combined_score = alpha * distances_from_current + beta * distances_to_dest + gamma * min_distances + 0.2 * normalized_progress
    next_node_idx = np.argmin(combined_score)
    next_node = unvisited_nodes[next_node_idx]
    
    return next_node



