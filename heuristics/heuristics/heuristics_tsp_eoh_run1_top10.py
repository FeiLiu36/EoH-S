# Top 10 functions for eoh run 1

# Function 1 - Score: -0.15303399823207148
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
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    
    momentum = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes].reshape(-1, 1), axis=1)
    degree_attraction = np.sum(distance_matrix[unvisited_nodes] > 0, axis=1)
    cluster_penalty = -np.mean(distance_matrix[:, unvisited_nodes], axis=0)
    exploitation = np.random.rand(len(unvisited_nodes))
    
    remaining_nodes = len(unvisited_nodes)
    total_nodes = len(distance_matrix)
    progress = remaining_nodes / total_nodes
    
    momentum_weight = np.tanh(progress)
    attraction_weight = 1 / (1 + np.exp(-5 * (1 - progress)))
    penalty_weight = np.tanh(1 - progress)
    exploitation_weight = 0.1 * progress
    
    combined_score = (momentum_weight * momentum) + (attraction_weight * degree_attraction) + (penalty_weight * cluster_penalty) + (exploitation_weight * exploitation)
    return unvisited_nodes[np.argmax(combined_score)]



# Function 2 - Score: -0.153509539253602
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
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    
    momentum = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes].reshape(-1, 1), axis=1)
    degree_attraction = np.sum(distance_matrix[unvisited_nodes] > 0, axis=1)
    cluster_penalty = -np.mean(distance_matrix[:, unvisited_nodes], axis=0)
    exploration = np.random.rand(len(unvisited_nodes))
    
    remaining_nodes = len(unvisited_nodes)
    total_nodes = len(distance_matrix)
    progress = remaining_nodes / total_nodes
    
    momentum_weight = np.tanh(5 * progress)
    attraction_weight = np.tanh(5 * (1 - progress))
    penalty_weight = 1 / (1 + np.exp(-5 * progress))
    exploration_weight = 0.1 * (1 - progress)
    
    combined_score = (momentum_weight * momentum) + (attraction_weight * degree_attraction) + (penalty_weight * cluster_penalty) + (exploration_weight * exploration)
    return unvisited_nodes[np.argmax(combined_score)]



# Function 3 - Score: -0.1536575026567046
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
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    
    momentum = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes].reshape(-1, 1), axis=1)
    degree_pull = np.sum(distance_matrix[unvisited_nodes] > 0, axis=1)
    proximity_penalty = np.min(distance_matrix[:, unvisited_nodes], axis=0)
    exploration = np.random.rand(len(unvisited_nodes))
    
    remaining_nodes = len(unvisited_nodes)
    total_nodes = len(distance_matrix)
    progress = remaining_nodes / total_nodes
    
    momentum_weight = np.tanh(3 * progress)
    degree_weight = np.tanh(3 * (1 - progress))
    penalty_weight = 1 - np.tanh(progress)
    exploration_weight = 0.1 * (1 - progress)
    
    combined_score = (momentum_weight * momentum) + (degree_weight * degree_pull) - (penalty_weight * proximity_penalty) + (exploration_weight * exploration)
    return unvisited_nodes[np.argmax(combined_score)]



# Function 4 - Score: -0.1538141227141504
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
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    
    momentum = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes].reshape(-1, 1), axis=1)
    degree_repulsion = -np.sum(distance_matrix[unvisited_nodes] > 0, axis=1)
    cluster_reward = np.mean(distance_matrix[:, unvisited_nodes], axis=0)
    exploration = np.random.rand(len(unvisited_nodes))
    
    remaining_nodes = len(unvisited_nodes)
    total_nodes = len(distance_matrix)
    progress = remaining_nodes / total_nodes
    
    momentum_weight = 1 / (1 + np.exp(-3 * progress))
    repulsion_weight = 1 / (1 + np.exp(-3 * (1 - progress)))
    reward_weight = np.tanh(progress * 2)
    exploration_weight = 0.1 * (1 - progress)
    
    combined_score = (momentum_weight * momentum) + (repulsion_weight * degree_repulsion) + (reward_weight * cluster_reward) + (exploration_weight * exploration)
    return unvisited_nodes[np.argmax(combined_score)]



# Function 5 - Score: -0.1546377825789089
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
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    
    momentum = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes].reshape(-1, 1), axis=1)
    degree_pull = np.sum(distance_matrix[unvisited_nodes] > 0, axis=1)
    proximity_penalty = np.min(distance_matrix[:, unvisited_nodes], axis=0)
    
    remaining_nodes = len(unvisited_nodes)
    total_nodes = len(distance_matrix)
    progress = remaining_nodes / total_nodes
    
    momentum_weight = 1 / (1 + np.exp(-5 * (progress - 0.5)))
    degree_weight = 1 / (1 + np.exp(-5 * (0.5 - progress)))
    penalty_weight = np.exp(-2 * progress)
    
    combined_score = (momentum_weight * momentum) + (degree_weight * degree_pull) - (penalty_weight * proximity_penalty)
    return unvisited_nodes[np.argmax(combined_score)]



# Function 6 - Score: -0.1546377825789089
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
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    
    momentum = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes].reshape(-1, 1), axis=1)
    degree_pull = np.sum(distance_matrix[unvisited_nodes] > 0, axis=1)
    proximity_penalty = np.min(distance_matrix[:, unvisited_nodes], axis=0)
    
    remaining_nodes = len(unvisited_nodes)
    total_nodes = len(distance_matrix)
    progress = remaining_nodes / total_nodes
    
    momentum_weight = 1 / (1 + np.exp(-5 * (progress - 0.5)))
    degree_weight = 1 / (1 + np.exp(-5 * (0.5 - progress)))
    penalty_weight = np.exp(-2 * progress)
    
    combined_score = (momentum_weight * momentum) + (degree_weight * degree_pull) - (penalty_weight * proximity_penalty)
    return unvisited_nodes[np.argmax(combined_score)]



# Function 7 - Score: -0.1547787942688424
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
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    
    momentum = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes].reshape(-1, 1), axis=1)
    degree_pull = np.sum(distance_matrix[unvisited_nodes] > 0, axis=1)
    proximity_penalty = np.min(distance_matrix[:, unvisited_nodes], axis=0)
    exploration = np.random.rand(len(unvisited_nodes))
    
    remaining_nodes = len(unvisited_nodes)
    total_nodes = len(distance_matrix)
    progress = remaining_nodes / total_nodes
    
    momentum_weight = np.tanh(3 * progress)
    degree_weight = np.tanh(3 * (1 - progress))
    penalty_weight = 1 - progress
    exploration_weight = 0.1 * (1 - progress)
    
    combined_score = (momentum_weight * momentum) + (degree_weight * degree_pull) - (penalty_weight * proximity_penalty) + (exploration_weight * exploration)
    return unvisited_nodes[np.argmax(combined_score)]



# Function 8 - Score: -0.15658721748136623
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
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    
    momentum = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes].reshape(-1, 1), axis=1)
    degree_repulsion = -np.sum(distance_matrix[unvisited_nodes] > 0, axis=1)
    cluster_reward = np.mean(distance_matrix[:, unvisited_nodes], axis=0)
    exploration = np.random.rand(len(unvisited_nodes))
    
    remaining_nodes = len(unvisited_nodes)
    total_nodes = len(distance_matrix)
    progress = remaining_nodes / total_nodes
    
    momentum_weight = 1 / (1 + np.exp(-5 * progress))
    repulsion_weight = 1 / (1 + np.exp(-5 * (1 - progress)))
    reward_weight = np.tanh(progress)
    exploration_weight = 0.2 * (1 - progress)
    
    combined_score = (momentum_weight * momentum) + (repulsion_weight * degree_repulsion) + (reward_weight * cluster_reward) + (exploration_weight * exploration)
    return unvisited_nodes[np.argmax(combined_score)]



# Function 9 - Score: -0.1569424947863794
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
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    
    repulsion = -np.sum(distance_matrix[unvisited_nodes] > 0, axis=1)
    dest_reward = -dest_dist
    exploration = np.random.rand(len(unvisited_nodes))
    momentum = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes].reshape(-1, 1), axis=1)
    
    remaining_nodes = len(unvisited_nodes)
    total_nodes = len(distance_matrix)
    progress = remaining_nodes / total_nodes
    
    repulsion_weight = 1 / (1 + np.exp(-5 * progress))
    reward_weight = np.tanh(1 - progress)
    exploration_weight = 0.2 * progress
    momentum_weight = np.tanh(progress)
    
    combined_score = (repulsion_weight * repulsion) + (reward_weight * dest_reward) + (exploration_weight * exploration) + (momentum_weight * momentum)
    return unvisited_nodes[np.argmax(combined_score)]



# Function 10 - Score: -0.15774220566968344
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
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    
    momentum = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes].reshape(-1, 1), axis=1)
    repulsion = np.mean(distance_matrix[:, unvisited_nodes], axis=0)
    harmonic_mean = len(unvisited_nodes) / np.sum(1 / (distance_matrix[unvisited_nodes][:, unvisited_nodes] + 1e-6), axis=1)
    
    remaining_nodes = len(unvisited_nodes)
    total_nodes = len(distance_matrix)
    progress = remaining_nodes / total_nodes
    
    momentum_weight = np.exp(-2 * progress)
    repulsion_weight = np.exp(-3 * (1 - progress))
    harmonic_weight = np.exp(-progress)
    
    combined_score = (momentum_weight * momentum) + (repulsion_weight * repulsion) + (harmonic_weight * harmonic_mean)
    return unvisited_nodes[np.argmax(combined_score)]



