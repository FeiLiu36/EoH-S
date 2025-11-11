# Top 10 functions for eoh run 2

# Function 1 - Score: -0.15006812998831665
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
    
    scores = []
    for node in unvisited_nodes:
        dist_current = distance_matrix[current_node, node]
        harmonic_mean = len(unvisited_nodes) / np.sum(1.0 / (distance_matrix[node, unvisited_nodes] + 1e-10))
        centrality_penalty = np.mean(distance_matrix[node, unvisited_nodes])
        score = dist_current + harmonic_mean - centrality_penalty
        scores.append(score)
    
    return unvisited_nodes[np.argmin(scores)]



# Function 2 - Score: -0.15006812998831665
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
    
    scores = []
    for node in unvisited_nodes:
        dist_current = distance_matrix[current_node, node]
        
        # MST cost of remaining unvisited nodes + current candidate
        subgraph_nodes = np.append(unvisited_nodes[unvisited_nodes != node], node)
        subgraph_dist = distance_matrix[np.ix_(subgraph_nodes, subgraph_nodes)]
        mst_cost = np.sum(np.min(subgraph_dist + np.diag(np.full(len(subgraph_nodes), np.inf)), axis=1))
        
        # Geometric median reward (approximated by mean distance)
        mean_dist = np.mean(distance_matrix[node, unvisited_nodes])
        
        score = -dist_current - mst_cost + mean_dist
        scores.append(score)
    
    return unvisited_nodes[np.argmax(scores)]



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
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    dist_to_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    geometric_mean = np.prod(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1) ** (1.0 / len(unvisited_nodes))
    
    score = 0.4 * dist_to_current + 0.4 * geometric_mean - 0.2 * dist_to_dest
    return unvisited_nodes[np.argmin(score)]



# Function 4 - Score: -0.1817720695379905
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
    
    # Calculate factors
    dist_to_current = distance_matrix[current_node, unvisited_nodes]
    median_dist_to_others = np.median(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    visit_counts = np.zeros(len(unvisited_nodes))  # Placeholder for tracking visits (simplified)
    
    # Normalize factors
    norm_dist_current = (dist_to_current - np.min(dist_to_current)) / (np.max(dist_to_current) - np.min(dist_to_current) + 1e-10)
    norm_median_dist = (median_dist_to_others - np.min(median_dist_to_others)) / (np.max(median_dist_to_others) - np.min(median_dist_to_others) + 1e-10)
    norm_visits = (visit_counts - np.min(visit_counts)) / (np.max(visit_counts) - np.min(visit_counts) + 1e-10)
    
    # Dynamic weights based on progress
    progress = 1 - (len(unvisited_nodes) / len(distance_matrix))
    w1 = 0.7 - 0.2 * progress
    w2 = 0.2 + 0.1 * progress
    w3 = 0.1 + 0.1 * progress
    
    # Combined score
    score = w1 * (1 - norm_dist_current) + w2 * norm_median_dist + w3 * (1 - norm_visits)
    
    return unvisited_nodes[np.argmax(score)]



# Function 5 - Score: -0.18205897367143326
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
    
    dist_to_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    harmonic_mean = len(unvisited_nodes) / np.sum(1.0 / (distance_matrix[unvisited_nodes][:, unvisited_nodes] + 1e-10), axis=1)
    
    score = 0.5 * dist_to_current + 0.3 * harmonic_mean - 0.2 * dist_to_dest
    return unvisited_nodes[np.argmin(score)]



# Function 6 - Score: -0.18205897367143326
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
    
    dist_to_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    harmonic_mean = len(unvisited_nodes) / np.sum(1.0 / (distance_matrix[unvisited_nodes][:, unvisited_nodes] + 1e-10), axis=1)
    
    score = 0.5 * dist_to_current + 0.3 * harmonic_mean - 0.2 * dist_to_dest
    return unvisited_nodes[np.argmin(score)]



# Function 7 - Score: -0.18541486256755724
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
    if len(unvisited_nodes) == 0:
        return destination_node

    dist_to_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    ratio = (dist_to_dest + 1e-10) / (dist_to_current + 1e-10)
    weighted_score = ratio * np.exp(-0.1 * dist_to_current)

    next_node_idx = np.argmax(weighted_score)
    return unvisited_nodes[next_node_idx]



# Function 8 - Score: -0.18577748484207068
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

    remaining_nodes = len(unvisited_nodes)
    scores = []
    for node in unvisited_nodes:
        angular_dev = np.abs(np.arctan2(distance_matrix[node, destination_node], distance_matrix[current_node, node]))
        harm_mean_dist = remaining_nodes / np.sum(1.0 / (distance_matrix[node, unvisited_nodes] + 1e-10))
        proximity_penalty = np.exp(-distance_matrix[current_node, node] / remaining_nodes)
        score = angular_dev * harm_mean_dist * proximity_penalty
        scores.append(score)

    return unvisited_nodes[np.argmax(scores)]



# Function 9 - Score: -0.18842255760880833
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
    if len(unvisited_nodes) == 0:  
        return destination_node  

    dist_to_current = distance_matrix[current_node, unvisited_nodes]  
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]  
    ratio = dist_to_current / (dist_to_current + dist_to_dest + 1e-10)  
    nonlinear_score = np.exp(-ratio)  

    next_node_idx = np.argmax(nonlinear_score)  
    return unvisited_nodes[next_node_idx]  



# Function 10 - Score: -0.18855392018216963
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
    
    scores = []
    remaining_nodes = len(unvisited_nodes)
    for node in unvisited_nodes:
        inv_dist_current = 1.0 / (distance_matrix[current_node, node] + 1e-10)
        harmonic_dist_others = remaining_nodes / np.sum(1.0 / (distance_matrix[node, unvisited_nodes] + 1e-10))
        dest_penalty = np.exp(-distance_matrix[node, destination_node] / remaining_nodes)
        score = (0.5 * inv_dist_current + 0.3 * harmonic_dist_others) * (1.0 - dest_penalty)
        scores.append(score)
    
    return unvisited_nodes[np.argmax(scores)]



