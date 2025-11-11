# Top 10 functions for eoh run 3

# Function 1 - Score: -0.1531653517590077
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
    remaining_nodes = len(unvisited_nodes)
    proximity_weight = 0.4 + (0.3 * (1 / remaining_nodes))
    progress_weight = 0.3 - (0.1 * (1 / remaining_nodes))
    curvature_weight = 0.3 * np.exp(-remaining_nodes / 5)
    if remaining_nodes == 1:
        return unvisited_nodes[0]
    next_dist = distance_matrix[unvisited_nodes, :][:, unvisited_nodes].min(axis=1)
    curvature_term = np.abs(current_dist + next_dist - distance_matrix[current_node, unvisited_nodes])
    score = (proximity_weight * current_dist) + (progress_weight * (distance_matrix[current_node, destination_node] - dest_dist)) + (curvature_weight * curvature_term)
    return unvisited_nodes[np.argmin(score)]



# Function 2 - Score: -0.1532018658190086
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
    remaining_nodes = len(unvisited_nodes)
    gravity_weight = 0.5 * np.log(remaining_nodes + 1) / np.log(20)
    repulsion_weight = 0.3 * (1 - np.exp(-remaining_nodes / 5))
    inertia_weight = 0.1 * (remaining_nodes / (remaining_nodes + 3))
    noise_weight = 0.1 * (1 - np.log(remaining_nodes + 1) / np.log(30))
    repulsion_dir = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    inertia_dir = np.mean(distance_matrix[unvisited_nodes] - distance_matrix[destination_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    score = (gravity_weight * dest_dist) - (repulsion_weight * repulsion_dir) + (inertia_weight * inertia_dir) + (noise_weight * np.random.rand() * np.max(current_dist))
    return unvisited_nodes[np.argmin(score)]



# Function 3 - Score: -0.1533811168657203
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
    remaining_nodes = len(unvisited_nodes)
    geo_weight = 0.4 * (1 - np.exp(-remaining_nodes / 8))
    centrality_weight = 0.3 * np.log(remaining_nodes + 1) / np.log(40)
    momentum_weight = 0.2 * (1 / (remaining_nodes + 2))
    exploration_weight = 0.1 * np.exp(-remaining_nodes / 12)
    geo_score = np.sqrt(current_dist * dest_dist)
    centrality_dir = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    momentum_dir = np.mean(distance_matrix[unvisited_nodes] - distance_matrix[destination_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    score = (geo_weight * geo_score) - (centrality_weight * centrality_dir) + (momentum_weight * momentum_dir) + (exploration_weight * np.random.rand() * np.max(current_dist))
    return unvisited_nodes[np.argmin(score)]



# Function 4 - Score: -0.15375948115922622
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
    remaining_nodes = len(unvisited_nodes)
    sigmoid = 1 / (1 + np.exp(-remaining_nodes / len(distance_matrix)))
    proximity_weight = 0.6 * sigmoid
    progress_weight = 0.4 * (1 - sigmoid)
    diversity_weight = 0.2 * (1 - sigmoid)
    progress_term = distance_matrix[current_node, destination_node] - dest_dist
    diversity_term = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    score = (proximity_weight * current_dist) + (progress_weight * progress_term) - (diversity_weight * diversity_term)
    return unvisited_nodes[np.argmin(score)]



# Function 5 - Score: -0.15376513442749923
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
    remaining_nodes = len(unvisited_nodes)
    harmonic_weight = 0.5 * (1 - np.exp(-remaining_nodes / 10))
    clustering_weight = 0.3 * np.log(remaining_nodes + 1) / np.log(25)
    curvature_weight = 0.15 * (remaining_nodes / (remaining_nodes + 5))
    temperature_weight = 0.05 * np.exp(-remaining_nodes / 15)
    harmonic_score = 2 * (current_dist * dest_dist) / (current_dist + dest_dist + 1e-8)
    clustering_dir = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    curvature_dir = np.abs(current_dist - np.mean(distance_matrix[unvisited_nodes], axis=1)) if remaining_nodes > 1 else 0
    score = (harmonic_weight * harmonic_score) - (clustering_weight * clustering_dir) + (curvature_weight * curvature_dir) + (temperature_weight * np.random.rand() * np.max(current_dist))
    return unvisited_nodes[np.argmin(score)]



# Function 6 - Score: -0.15393791252976335
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
    remaining_nodes = len(unvisited_nodes)
    harmonic_weight = 0.5 * (1 - np.exp(-remaining_nodes / 10))
    density_weight = 0.2 * np.log(remaining_nodes + 1) / np.log(50)
    curvature_weight = 0.2 * (1 / (remaining_nodes + 1))
    exploration_weight = 0.1 * np.exp(-remaining_nodes / 15)
    harmonic_score = 2 * (current_dist * dest_dist) / (current_dist + dest_dist + 1e-10)
    density_dir = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    curvature_dir = np.mean(distance_matrix[unvisited_nodes] - distance_matrix[destination_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    score = (harmonic_weight * harmonic_score) - (density_weight * density_dir) + (curvature_weight * curvature_dir) + (exploration_weight * np.random.rand() * np.max(current_dist))
    return unvisited_nodes[np.argmin(score)]



# Function 7 - Score: -0.15490539249092683
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
    remaining_nodes = len(unvisited_nodes)
    gravity_weight = 0.6 * (1 - np.exp(-remaining_nodes / 10))
    repulsion_weight = 0.2 * np.log(remaining_nodes + 2) / np.log(30)
    inertia_weight = 0.15 * (1 / (remaining_nodes + 2))
    noise_weight = 0.05 * np.exp(-remaining_nodes / 15)
    repulsion_dir = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    inertia_dir = np.mean(distance_matrix[unvisited_nodes] - distance_matrix[destination_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    score = (gravity_weight * dest_dist) - (repulsion_weight * repulsion_dir) + (inertia_weight * inertia_dir) + (noise_weight * np.random.rand() * np.max(current_dist))
    return unvisited_nodes[np.argmin(score)]



# Function 8 - Score: -0.15529520503639851
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
    remaining_nodes = len(unvisited_nodes)
    gravity_weight = 0.5 * (1 - np.exp(-remaining_nodes / 5))
    repulsion_weight = 0.3 * np.log(remaining_nodes + 1) / np.log(50)
    momentum_weight = 0.1 * (1 / (remaining_nodes + 1))
    noise_weight = 0.1 * np.exp(-remaining_nodes / 20)
    repulsion_dir = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    momentum_dir = np.mean(distance_matrix[unvisited_nodes] - distance_matrix[destination_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    score = (gravity_weight * dest_dist) - (repulsion_weight * repulsion_dir) + (momentum_weight * momentum_dir) + (noise_weight * np.random.rand() * np.max(current_dist))
    return unvisited_nodes[np.argmin(score)]



# Function 9 - Score: -0.1555169439789373
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
    remaining_nodes = len(unvisited_nodes)
    proximity_weight = 0.4 - (0.1 * (1 / remaining_nodes))
    progress_weight = 0.3 + (0.05 * (1 / remaining_nodes))
    repulsion_weight = 0.2 * np.exp(-remaining_nodes / 10)
    exploration_weight = 0.1 * (1 - np.log(remaining_nodes + 1) / np.log(50))
    repulsion_dir = np.mean(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    score = (proximity_weight * current_dist) + (progress_weight * (distance_matrix[current_node, destination_node] - dest_dist)) - (repulsion_weight * repulsion_dir) + (exploration_weight * np.random.rand() * np.max(current_dist))
    return unvisited_nodes[np.argmin(score)]



# Function 10 - Score: -0.1562353896052205
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
    remaining_nodes = len(unvisited_nodes)
    proximity_weight = 0.6 * np.exp(-remaining_nodes / 10)
    cluster_weight = 0.2 * (1 - np.exp(-remaining_nodes / 8))
    explore_weight = 0.15 * np.log(remaining_nodes + 1) / np.log(15)
    random_weight = 0.05 * (remaining_nodes / (remaining_nodes + 5))
    cluster_dir = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    explore_dir = np.mean(distance_matrix[unvisited_nodes] - distance_matrix[destination_node, unvisited_nodes][:, None], axis=1) if remaining_nodes > 1 else 0
    score = (proximity_weight * dest_dist) - (cluster_weight * cluster_dir) + (explore_weight * explore_dir) + (random_weight * np.random.rand() * np.max(current_dist))
    return unvisited_nodes[np.argmin(score)]



