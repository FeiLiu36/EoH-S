# Top 10 functions for eohs run 3

# Function 1 - Score: -0.13746000774318157
{The new algorithm selects the next node by integrating proximity, directionality, momentum, spatial novelty, connectivity, path smoothness, adaptive exploration, dynamic potential, and harmonic resonance, with weights dynamically adjusted by traversal progress, spatial distribution, curvature, and a novel "quantum entanglement" mechanism that probabilistically oscillates between exploration and exploitation based on a quantum-inspired entanglement of node selection criteria, while optimizing for energy efficiency and path coherence.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    proximity = distance_matrix[current_node, unvisited_nodes]
    directionality = distance_matrix[unvisited_nodes, destination_node]
    
    momentum = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    
    local_density = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    spatial_novelty = np.max(local_density) - local_density
    
    connectivity = np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    if len(unvisited_nodes) > 1:
        smoothness = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    else:
        smoothness = np.zeros(len(unvisited_nodes))
    
    path_diversity = np.std(distance_matrix[unvisited_nodes])
    exploration_factor = np.tanh(2 * path_diversity / np.mean(distance_matrix))
    
    local_potential = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    global_potential = np.mean(distance_matrix[unvisited_nodes], axis=1)
    dynamic_potential = (local_potential + global_potential) / 2
    
    progress = 1 - len(unvisited_nodes) / len(distance_matrix)
    spatial_dist = np.mean(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix)
    curvature = np.std(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix[unvisited_nodes])
    quantum_entanglement = np.random.uniform(0, 1) * (1 + np.sin(2 * np.pi * progress + np.pi * spatial_dist))
    
    quantum_resonance = np.sin(2 * np.pi * progress + np.pi * spatial_dist) * quantum_entanglement
    
    w_proximity = (0.15 - 0.05 * quantum_entanglement) * (1 + 0.1 * spatial_dist)
    w_directionality = (0.13 + 0.05 * quantum_entanglement) * (1 - 0.1 * spatial_dist)
    w_momentum = (0.11 + 0.05 * quantum_entanglement) * (1 + 0.1 * progress)
    w_novelty = (0.13 - 0.05 * quantum_entanglement) * (1 - 0.1 * progress)
    w_connectivity = (0.13 + 0.05 * quantum_entanglement) * (1 + 0.1 * curvature)
    w_smoothness = (0.15 - 0.05 * quantum_entanglement) * (1 - 0.1 * curvature)
    w_exploration = (0.05 + 0.05 * quantum_entanglement) * exploration_factor
    w_potential = (0.07 + 0.03 * quantum_entanglement) * dynamic_potential
    w_resonance = (0.06 + 0.04 * quantum_entanglement) * quantum_resonance
    
    combined_score = (
        w_proximity * proximity +
        w_directionality * (np.max(directionality) - directionality) +
        w_momentum * momentum +
        w_novelty * spatial_novelty +
        w_connectivity * connectivity +
        w_smoothness * smoothness +
        w_exploration * spatial_novelty +
        w_potential * dynamic_potential +
        w_resonance * quantum_resonance
    )
    return unvisited_nodes[np.argmin(combined_score)]



# Function 2 - Score: -0.15058642878392592
{The new algorithm selects the next node by combining proximity, directionality, momentum, spatial novelty, connectivity, path smoothness, adaptive exploration, dynamic potential, temporal rhythm, gravitational attraction, path entropy, and introduces a "quantum tunneling" factor (prioritizing nodes that offer unexpected shortcuts) and a "cultural diffusion" factor (mimicking successful paths from nearby nodes), with weights dynamically adjusted by traversal progress, spatial distribution, harmonic phase, and a new "chaos resonance" mechanism that balances exploration and exploitation based on path complexity and node clustering.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    proximity = distance_matrix[current_node, unvisited_nodes]
    directionality = distance_matrix[unvisited_nodes, destination_node]
    
    momentum = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    
    local_density = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    spatial_novelty = np.max(local_density) - local_density
    
    connectivity = np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    if len(unvisited_nodes) > 1:
        smoothness = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    else:
        smoothness = np.zeros(len(unvisited_nodes))
    
    path_diversity = np.std(distance_matrix[unvisited_nodes])
    exploration_factor = np.tanh(2 * path_diversity / np.mean(distance_matrix))
    
    local_potential = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    global_potential = np.mean(distance_matrix[unvisited_nodes], axis=1)
    dynamic_potential = (local_potential + global_potential) / 2
    
    progress = 1 - len(unvisited_nodes) / len(distance_matrix)
    spatial_dist = np.mean(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix)
    curvature = np.std(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix[unvisited_nodes])
    phase = 0.5 + 0.5 * np.sin(10 * progress + 5 * spatial_dist)
    
    temporal_rhythm = np.sin(5 * progress + 2 * spatial_dist)
    
    visited_nodes = np.setdiff1d(np.arange(len(distance_matrix)), unvisited_nodes)
    if len(visited_nodes) > 0:
        gravitational_attraction = np.mean(distance_matrix[unvisited_nodes][:, visited_nodes], axis=1)
    else:
        gravitational_attraction = np.zeros(len(unvisited_nodes))
    
    path_entropy = np.exp(-np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1))
    
    quantum_tunneling = np.min(distance_matrix[unvisited_nodes][:, visited_nodes], axis=1) if len(visited_nodes) > 0 else np.zeros(len(unvisited_nodes))
    
    cultural_diffusion = np.mean(distance_matrix[unvisited_nodes][:, visited_nodes], axis=1) if len(visited_nodes) > 0 else np.zeros(len(unvisited_nodes))
    
    chaos_resonance = np.tanh(np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1) / np.mean(distance_matrix[unvisited_nodes]))
    
    w_proximity = (0.10 - 0.02 * phase) * (1 + 0.1 * spatial_dist)
    w_directionality = (0.09 + 0.02 * phase) * (1 - 0.1 * spatial_dist)
    w_momentum = (0.07 + 0.02 * phase) * (1 + 0.1 * progress)
    w_novelty = (0.09 - 0.02 * phase) * (1 - 0.1 * progress)
    w_connectivity = (0.09 + 0.02 * phase) * (1 + 0.1 * curvature)
    w_smoothness = (0.12 - 0.02 * phase) * (1 - 0.1 * curvature)
    w_exploration = (0.06 + 0.02 * phase) * exploration_factor
    w_potential = (0.06 + 0.02 * phase) * dynamic_potential
    w_rhythm = (0.07 + 0.03 * phase) * temporal_rhythm
    w_gravitational = (0.05 + 0.02 * phase) * gravitational_attraction
    w_entropy = (0.08 - 0.02 * phase) * path_entropy
    w_quantum = (0.05 + 0.02 * phase) * quantum_tunneling
    w_cultural = (0.04 + 0.02 * phase) * cultural_diffusion
    w_chaos = (0.03 + 0.01 * phase) * chaos_resonance
    
    combined_score = (
        w_proximity * proximity +
        w_directionality * (np.max(directionality) - directionality) +
        w_momentum * momentum +
        w_novelty * spatial_novelty +
        w_connectivity * connectivity +
        w_smoothness * smoothness +
        w_exploration * spatial_novelty +
        w_potential * dynamic_potential +
        w_rhythm * temporal_rhythm +
        w_gravitational * gravitational_attraction +
        w_entropy * path_entropy +
        w_quantum * quantum_tunneling +
        w_cultural * cultural_diffusion +
        w_chaos * chaos_resonance
    )
    return unvisited_nodes[np.argmin(combined_score)]



# Function 3 - Score: -0.15061259893101042
{The algorithm selects the next node by combining proximity, directionality, momentum, spatial novelty, connectivity, path smoothness, adaptive exploration, dynamic potential, temporal rhythm, gravitational attraction, path entropy, and introduces a new "quantum tunneling" factor (prioritizing nodes that are far but have low local density) and a "fractal dimension" factor (balancing between local and global exploration based on the spatial distribution's fractal properties), with weights dynamically adjusted by a "cosmic phase" mechanism that modulates factor importance based on path curvature, node density, and a harmonic balance between exploration and exploitation.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    proximity = distance_matrix[current_node, unvisited_nodes]
    directionality = distance_matrix[unvisited_nodes, destination_node]
    
    momentum = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    
    local_density = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    spatial_novelty = np.max(local_density) - local_density
    
    connectivity = np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    if len(unvisited_nodes) > 1:
        smoothness = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    else:
        smoothness = np.zeros(len(unvisited_nodes))
    
    path_diversity = np.std(distance_matrix[unvisited_nodes])
    exploration_factor = np.tanh(2 * path_diversity / np.mean(distance_matrix))
    
    local_potential = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    global_potential = np.mean(distance_matrix[unvisited_nodes], axis=1)
    dynamic_potential = (local_potential + global_potential) / 2
    
    progress = 1 - len(unvisited_nodes) / len(distance_matrix)
    spatial_dist = np.mean(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix)
    curvature = np.std(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix[unvisited_nodes])
    cosmic_phase = 0.5 + 0.5 * np.sin(8 * progress + 4 * spatial_dist + 2 * curvature)
    
    temporal_rhythm = np.sin(4 * progress + 2 * spatial_dist)
    
    visited_nodes = np.setdiff1d(np.arange(len(distance_matrix)), unvisited_nodes)
    if len(visited_nodes) > 0:
        gravitational_attraction = np.mean(distance_matrix[unvisited_nodes][:, visited_nodes], axis=1)
    else:
        gravitational_attraction = np.zeros(len(unvisited_nodes))
    
    path_entropy = np.exp(-np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1))
    
    quantum_tunneling = (np.max(proximity) - proximity) * (1 - local_density / np.max(local_density))
    
    fractal_dimension = np.log(local_density) / np.log(proximity + 1e-10)
    
    w_proximity = (0.10 - 0.02 * cosmic_phase) * (1 + 0.1 * spatial_dist)
    w_directionality = (0.09 + 0.02 * cosmic_phase) * (1 - 0.1 * spatial_dist)
    w_momentum = (0.07 + 0.02 * cosmic_phase) * (1 + 0.1 * progress)
    w_novelty = (0.08 - 0.02 * cosmic_phase) * (1 - 0.1 * progress)
    w_connectivity = (0.09 + 0.02 * cosmic_phase) * (1 + 0.1 * curvature)
    w_smoothness = (0.12 - 0.02 * cosmic_phase) * (1 - 0.1 * curvature)
    w_exploration = (0.04 + 0.01 * cosmic_phase) * exploration_factor
    w_potential = (0.06 + 0.01 * cosmic_phase) * dynamic_potential
    w_rhythm = (0.07 + 0.03 * cosmic_phase) * temporal_rhythm
    w_gravitational = (0.05 + 0.01 * cosmic_phase) * gravitational_attraction
    w_entropy = (0.08 - 0.01 * cosmic_phase) * path_entropy
    w_quantum = (0.05 + 0.01 * cosmic_phase) * quantum_tunneling
    w_fractal = (0.10 - 0.02 * cosmic_phase) * fractal_dimension
    
    combined_score = (
        w_proximity * proximity +
        w_directionality * (np.max(directionality) - directionality) +
        w_momentum * momentum +
        w_novelty * spatial_novelty +
        w_connectivity * connectivity +
        w_smoothness * smoothness +
        w_exploration * spatial_novelty +
        w_potential * dynamic_potential +
        w_rhythm * temporal_rhythm +
        w_gravitational * gravitational_attraction +
        w_entropy * path_entropy +
        w_quantum * quantum_tunneling +
        w_fractal * fractal_dimension
    )
    return unvisited_nodes[np.argmin(combined_score)]



# Function 4 - Score: -0.1528416946766744
{The new algorithm selects the next node by combining proximity, directionality, momentum, spatial novelty, connectivity, path smoothness, adaptive exploration, dynamic potential, temporal rhythm, gravitational attraction, path entropy, and introduces a "magnetic repulsion" factor (avoiding nodes that are too similar to recently visited nodes) and a "stellar alignment" factor (prioritizing nodes that align with the overall path's geometric center), with weights dynamically adjusted by a "celestial harmony" mechanism that balances factors based on path symmetry, node dispersion, and a cosmic resonance pattern.}

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
    proximity = distance_matrix[current_node, unvisited_nodes]
    directionality = distance_matrix[unvisited_nodes, destination_node]
    
    momentum = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    
    local_density = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    spatial_novelty = np.max(local_density) - local_density
    
    connectivity = np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    if len(unvisited_nodes) > 1:
        smoothness = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    else:
        smoothness = np.zeros(len(unvisited_nodes))
    
    path_diversity = np.std(distance_matrix[unvisited_nodes])
    exploration_factor = np.tanh(2 * path_diversity / np.mean(distance_matrix))
    
    local_potential = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    global_potential = np.mean(distance_matrix[unvisited_nodes], axis=1)
    dynamic_potential = (local_potential + global_potential) / 2
    
    progress = 1 - len(unvisited_nodes) / len(distance_matrix)
    spatial_dist = np.mean(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix)
    curvature = np.std(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix[unvisited_nodes])
    celestial_harmony = 0.5 + 0.5 * np.sin(7 * progress + 3 * spatial_dist + curvature)
    
    temporal_rhythm = np.sin(3 * progress + spatial_dist)
    
    visited_nodes = np.setdiff1d(np.arange(len(distance_matrix)), unvisited_nodes)
    if len(visited_nodes) > 0:
        gravitational_attraction = np.mean(distance_matrix[unvisited_nodes][:, visited_nodes], axis=1)
    else:
        gravitational_attraction = np.zeros(len(unvisited_nodes))
    
    path_entropy = np.exp(-np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1))
    
    if len(visited_nodes) > 0:
        recent_nodes = visited_nodes[-min(3, len(visited_nodes)):]
        magnetic_repulsion = np.mean(distance_matrix[unvisited_nodes][:, recent_nodes], axis=1)
    else:
        magnetic_repulsion = np.zeros(len(unvisited_nodes))
    
    center = np.mean(distance_matrix, axis=0)
    stellar_alignment = np.abs(np.mean(distance_matrix[unvisited_nodes], axis=1) - center[unvisited_nodes])
    
    w_proximity = (0.11 - 0.02 * celestial_harmony) * (1 + 0.1 * spatial_dist)
    w_directionality = (0.10 + 0.02 * celestial_harmony) * (1 - 0.1 * spatial_dist)
    w_momentum = (0.08 + 0.02 * celestial_harmony) * (1 + 0.1 * progress)
    w_novelty = (0.09 - 0.02 * celestial_harmony) * (1 - 0.1 * progress)
    w_connectivity = (0.08 + 0.02 * celestial_harmony) * (1 + 0.1 * curvature)
    w_smoothness = (0.11 - 0.02 * celestial_harmony) * (1 - 0.1 * curvature)
    w_exploration = (0.05 + 0.01 * celestial_harmony) * exploration_factor
    w_potential = (0.07 + 0.01 * celestial_harmony) * dynamic_potential
    w_rhythm = (0.06 + 0.02 * celestial_harmony) * temporal_rhythm
    w_gravitational = (0.06 + 0.01 * celestial_harmony) * gravitational_attraction
    w_entropy = (0.07 - 0.01 * celestial_harmony) * path_entropy
    w_magnetic = (0.05 - 0.01 * celestial_harmony) * magnetic_repulsion
    w_stellar = (0.06 + 0.01 * celestial_harmony) * stellar_alignment
    
    combined_score = (
        w_proximity * proximity +
        w_directionality * (np.max(directionality) - directionality) +
        w_momentum * momentum +
        w_novelty * spatial_novelty +
        w_connectivity * connectivity +
        w_smoothness * smoothness +
        w_exploration * spatial_novelty +
        w_potential * dynamic_potential +
        w_rhythm * temporal_rhythm +
        w_gravitational * gravitational_attraction +
        w_entropy * path_entropy +
        w_magnetic * magnetic_repulsion +
        w_stellar * stellar_alignment
    )
    return unvisited_nodes[np.argmin(combined_score)]



# Function 5 - Score: -0.15403936411964173
{The algorithm selects the next node by combining proximity (distance from current_node), directionality (alignment toward destination_node), local clustering (average distance to nearby visited nodes), an exploration bonus (inverse density of unvisited nodes), path smoothness (minimizing sharp turns), potential gain (estimated future path improvement), a "momentum" factor (direction consistency with previous moves), an "attraction-repulsion" factor (balancing between visited and unvisited clusters), and a novel "adaptive entropy" factor (dynamic exploration-exploitation balance), with self-adjusting weights based on tour progress, dynamic randomness, and a complexity-aware learning rate.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    proximity = distance_matrix[current_node, unvisited_nodes]
    directionality = distance_matrix[unvisited_nodes, destination_node]
    
    # Local clustering: average distance to nearby visited nodes (within 5 nearest neighbors)
    nearby_nodes = np.argsort(distance_matrix[current_node])[:5]
    nearby_visited = [n for n in nearby_nodes if n not in unvisited_nodes]
    clustering = np.mean(distance_matrix[unvisited_nodes][:, nearby_visited], axis=1) if nearby_visited else np.zeros(len(unvisited_nodes))
    
    # Exploration bonus: inverse density of unvisited nodes
    if len(unvisited_nodes) > 1:
        exploration = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
        exploration = np.max(exploration) - exploration
    else:
        exploration = np.zeros(len(unvisited_nodes))
    
    # Path smoothness: minimize angle deviation from current path (approximated by distance ratio)
    if len(unvisited_nodes) > 1:
        smoothness = distance_matrix[current_node, unvisited_nodes] + distance_matrix[unvisited_nodes, destination_node]
    else:
        smoothness = np.zeros(len(unvisited_nodes))
    
    # Potential gain: estimated future path improvement (distance to nearest unvisited node from candidate)
    if len(unvisited_nodes) > 1:
        potential_gain = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    else:
        potential_gain = np.zeros(len(unvisited_nodes))
    
    # Momentum: direction consistency with previous moves (if available)
    if 'prev_move' in globals():
        momentum = np.array([np.dot(distance_matrix[current_node, n] - globals()['prev_move'], distance_matrix[n, destination_node]) for n in unvisited_nodes])
    else:
        momentum = np.zeros(len(unvisited_nodes))
    
    # Attraction-repulsion: balance between visited and unvisited clusters
    if len(unvisited_nodes) > 1:
        attraction = np.mean(distance_matrix[unvisited_nodes][:, nearby_visited], axis=1) if nearby_visited else np.zeros(len(unvisited_nodes))
        repulsion = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
        attraction_repulsion = attraction - repulsion
    else:
        attraction_repulsion = np.zeros(len(unvisited_nodes))
    
    # Adaptive entropy: dynamic exploration-exploitation balance
    entropy = np.std(distance_matrix[unvisited_nodes], axis=1) / np.mean(distance_matrix[unvisited_nodes], axis=1)
    
    # Dynamic weights with self-adjusting factors
    progress = 1 - len(unvisited_nodes) / len(distance_matrix)
    rand_factor = np.random.uniform(0.85, 1.15)
    learn_factor = 0.7 + 0.3 * np.sin(progress * np.pi)
    complexity = np.mean(distance_matrix) / np.max(distance_matrix)
    
    w_proximity = (0.14 + 0.05 * (1 - progress)) * rand_factor * learn_factor * (1 + 0.1 * complexity)
    w_directionality = (0.10 + 0.05 * progress) * rand_factor * learn_factor * (1 + 0.1 * complexity)
    w_clustering = (0.07 - 0.02 * progress) * rand_factor * learn_factor * (1 - 0.1 * complexity)
    w_exploration = (0.10 - 0.02 * progress) * rand_factor * learn_factor * (1 - 0.1 * complexity)
    w_smoothness = (0.10 + 0.02 * progress) * rand_factor * learn_factor * (1 + 0.1 * complexity)
    w_potential = (0.18 + 0.05 * progress) * rand_factor * learn_factor * (1 + 0.1 * complexity)
    w_momentum = (0.05 + 0.02 * progress) * rand_factor * learn_factor * (1 - 0.1 * complexity)
    w_att_rep = (0.12 + 0.03 * progress) * rand_factor * learn_factor * (1 + 0.1 * complexity)
    w_entropy = (0.14 - 0.04 * progress) * rand_factor * learn_factor * (1 - 0.1 * complexity)
    
    combined_score = (
        w_proximity * proximity +
        w_directionality * (np.max(directionality) - directionality) +
        w_clustering * clustering +
        w_exploration * exploration +
        w_smoothness * smoothness +
        w_potential * potential_gain +
        w_momentum * (np.max(momentum) - momentum) +
        w_att_rep * attraction_repulsion +
        w_entropy * entropy
    )
    globals()['prev_move'] = distance_matrix[current_node, unvisited_nodes[np.argmin(combined_score)]] - distance_matrix[current_node, current_node]
    return unvisited_nodes[np.argmin(combined_score)]



# Function 6 - Score: -0.15543604506057151
{The algorithm selects the next node by combining proximity (distance from current_node), directionality (alignment toward destination_node), path potential (minimum distance to other unvisited nodes), a dynamic momentum factor (weighted direction of previous moves), a territory coverage factor (distance to the geometric median of unvisited nodes), and a novel "exploration factor" (encourages visiting nodes that improve spatial coverage), with adaptive weights based on traversal progress, local node density, and spatial distribution skewness, and a temperature-based annealing to balance exploration and exploitation.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    proximity = distance_matrix[current_node, unvisited_nodes]
    directionality = distance_matrix[unvisited_nodes, destination_node]
    path_potential = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    if len(unvisited_nodes) > 1:
        momentum = np.mean(np.diff(distance_matrix[current_node, unvisited_nodes]))
    else:
        momentum = 0.0
    
    geo_median = np.median(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=0)
    territory_coverage = distance_matrix[current_node, unvisited_nodes] / (geo_median + 1e-9)
    
    exploration = np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    progress = 1 - (len(unvisited_nodes) / len(distance_matrix))
    density = np.mean(proximity) / (np.max(proximity) + 1e-9)
    skewness = np.abs(np.mean(proximity) - np.median(proximity)) / (np.std(proximity) + 1e-9)
    temperature = 1 - progress
    
    w_proximity = 0.3 - 0.1 * progress
    w_directionality = 0.25 + 0.05 * skewness
    w_path_potential = 0.2 + 0.05 * density
    w_momentum = 0.1 * (1 - skewness) * temperature
    w_territory = 0.1 * (1 - density)
    w_exploration = 0.05 * temperature
    
    combined_score = (
        w_proximity * proximity +
        w_directionality * (np.max(directionality) - directionality) +
        w_path_potential * path_potential +
        w_momentum * momentum +
        w_territory * territory_coverage +
        w_exploration * exploration
    )
    return unvisited_nodes[np.argmin(combined_score)]



# Function 7 - Score: -0.1630467235193923
{The algorithm selects the next node by combining proximity (distance from current_node), directionality (alignment toward destination_node), a "momentum" factor (moving toward the average direction of recent moves), a "spatial novelty" factor (prioritizing nodes in less explored regions), a "connectivity" factor (nodes that bridge dense clusters), a "path smoothness" factor (minimizing sharp turns), a "temporal rhythm" factor (oscillating between exploration and exploitation based on a time-varying sine wave of traversal progress), and a "gravitational pull" factor (attracting toward high-density regions while repelling from recently visited areas), with weights dynamically adjusted by a "phase shift" mechanism that modulates factor importance based on spatial entropy and path curvature.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    proximity = distance_matrix[current_node, unvisited_nodes]
    directionality = distance_matrix[unvisited_nodes, destination_node]
    
    momentum = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    
    local_density = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    spatial_novelty = np.max(local_density) - local_density
    
    connectivity = np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    if len(unvisited_nodes) > 1:
        smoothness = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    else:
        smoothness = np.zeros(len(unvisited_nodes))
    
    progress = 1 - len(unvisited_nodes) / len(distance_matrix)
    temporal_rhythm = np.sin(2 * np.pi * progress)
    
    gravitational_pull = np.mean(distance_matrix[unvisited_nodes], axis=1) * (1 + temporal_rhythm)
    
    spatial_entropy = np.std(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix)
    curvature = np.std(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix[unvisited_nodes])
    phase_shift = 0.5 + 0.5 * np.sin(2 * np.pi * spatial_entropy + np.pi * curvature)
    
    w_proximity = (0.16 - 0.06 * phase_shift) * (1 + 0.1 * spatial_entropy)
    w_directionality = (0.14 + 0.06 * phase_shift) * (1 - 0.1 * spatial_entropy)
    w_momentum = (0.12 + 0.04 * phase_shift) * (1 + 0.1 * progress)
    w_novelty = (0.13 - 0.05 * phase_shift) * (1 - 0.1 * progress)
    w_connectivity = (0.15 + 0.05 * phase_shift) * (1 + 0.1 * curvature)
    w_smoothness = (0.18 - 0.06 * phase_shift) * (1 - 0.1 * curvature)
    w_rhythm = (0.06 + 0.04 * phase_shift) * temporal_rhythm
    w_gravitational = (0.06 + 0.04 * phase_shift) * gravitational_pull
    
    combined_score = (
        w_proximity * proximity +
        w_directionality * (np.max(directionality) - directionality) +
        w_momentum * momentum +
        w_novelty * spatial_novelty +
        w_connectivity * connectivity +
        w_smoothness * smoothness +
        w_rhythm * temporal_rhythm +
        w_gravitational * gravitational_pull
    )
    return unvisited_nodes[np.argmin(combined_score)]



# Function 8 - Score: -0.16654194626607438
{The improved algorithm enhances node selection by combining proximity, directionality, momentum, spatial novelty, connectivity, path smoothness, adaptive exploration, dynamic potential, magnetic repulsion, chaotic resonance, and introduces a "quantum tunneling" factor (prioritizing nodes with high local minima escape potential) and a "harmonic convergence" factor (balancing exploration-exploitation via sinusoidal weight modulation), with weights dynamically adjusted by traversal progress, spatial distribution, curvature, and a "fractal resonance" mechanism that modulates decision thresholds using a logistic map.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    proximity = distance_matrix[current_node, unvisited_nodes]
    directionality = distance_matrix[unvisited_nodes, destination_node]
    
    momentum = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    
    local_density = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    spatial_novelty = np.max(local_density) - local_density
    
    connectivity = np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    if len(unvisited_nodes) > 1:
        smoothness = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    else:
        smoothness = np.zeros(len(unvisited_nodes))
    
    path_diversity = np.std(distance_matrix[unvisited_nodes])
    exploration_factor = np.tanh(3 * path_diversity / np.mean(distance_matrix))
    
    local_potential = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    global_potential = np.mean(distance_matrix[unvisited_nodes], axis=1)
    dynamic_potential = (local_potential + global_potential) / 2
    
    progress = 1 - len(unvisited_nodes) / len(distance_matrix)
    spatial_dist = np.mean(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix)
    curvature = np.std(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix[unvisited_nodes])
    harmonic_phase = 0.5 + 0.3 * np.sin(4 * progress + 3 * spatial_dist) * np.cos(6 * curvature)
    
    visited_nodes = np.setdiff1d(np.arange(len(distance_matrix)), unvisited_nodes)
    if len(visited_nodes) > 0:
        magnetic_repulsion = np.max(distance_matrix[unvisited_nodes][:, visited_nodes], axis=1)
    else:
        magnetic_repulsion = np.zeros(len(unvisited_nodes))
    
    path_complexity = np.mean(np.abs(np.diff(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)))
    chaotic_resonance = np.exp(-path_complexity) * np.random.uniform(0.7, 1.3)
    
    quantum_tunneling = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1) / np.max(distance_matrix[unvisited_nodes])
    
    logistic_map = 4 * harmonic_phase * (1 - harmonic_phase)
    fractal_resonance = 0.5 + 0.5 * np.sin(10 * logistic_map)
    
    w_proximity = (0.12 - 0.05 * harmonic_phase) * (1 + 0.2 * spatial_dist)
    w_directionality = (0.10 + 0.05 * harmonic_phase) * (1 - 0.2 * spatial_dist)
    w_momentum = (0.08 + 0.05 * harmonic_phase) * (1 + 0.2 * progress)
    w_novelty = (0.10 - 0.05 * harmonic_phase) * (1 - 0.2 * progress)
    w_connectivity = (0.10 + 0.05 * harmonic_phase) * (1 + 0.2 * curvature)
    w_smoothness = (0.12 - 0.05 * harmonic_phase) * (1 - 0.2 * curvature)
    w_exploration = (0.05 + 0.03 * harmonic_phase) * exploration_factor
    w_potential = (0.07 + 0.03 * harmonic_phase) * dynamic_potential
    w_repulsion = (0.05 + 0.03 * harmonic_phase) * magnetic_repulsion
    w_chaos = (0.05 - 0.03 * harmonic_phase) * chaotic_resonance
    w_quantum = (0.08 + 0.02 * harmonic_phase) * quantum_tunneling
    w_harmonic = (0.08 - 0.02 * harmonic_phase) * fractal_resonance
    
    combined_score = (
        w_proximity * proximity +
        w_directionality * (np.max(directionality) - directionality) +
        w_momentum * momentum +
        w_novelty * spatial_novelty +
        w_connectivity * connectivity +
        w_smoothness * smoothness +
        w_exploration * spatial_novelty +
        w_potential * dynamic_potential +
        w_repulsion * magnetic_repulsion +
        w_chaos * chaotic_resonance +
        w_quantum * quantum_tunneling +
        w_harmonic * fractal_resonance
    )
    return unvisited_nodes[np.argmin(combined_score)]



# Function 9 - Score: -0.17315343893042384
{The new algorithm selects the next node by combining proximity, directionality, momentum, spatial novelty, connectivity, path smoothness, adaptive exploration, dynamic potential, and introduces a "magnetic repulsion" factor (avoiding nodes that cluster too closely with visited nodes) and a "chaotic resonance" factor (introducing controlled randomness based on path complexity), with weights dynamically adjusted by traversal progress, spatial distribution, and a "fractal phase" mechanism that modulates exploration-exploitation balance using a chaotic attractor model.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    proximity = distance_matrix[current_node, unvisited_nodes]
    directionality = distance_matrix[unvisited_nodes, destination_node]
    
    momentum = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    
    local_density = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    spatial_novelty = np.max(local_density) - local_density
    
    connectivity = np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    
    if len(unvisited_nodes) > 1:
        smoothness = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    else:
        smoothness = np.zeros(len(unvisited_nodes))
    
    path_diversity = np.std(distance_matrix[unvisited_nodes])
    exploration_factor = np.tanh(2 * path_diversity / np.mean(distance_matrix))
    
    local_potential = np.mean(distance_matrix[unvisited_nodes][:, [current_node]], axis=1)
    global_potential = np.mean(distance_matrix[unvisited_nodes], axis=1)
    dynamic_potential = (local_potential + global_potential) / 2
    
    progress = 1 - len(unvisited_nodes) / len(distance_matrix)
    spatial_dist = np.mean(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix)
    curvature = np.std(distance_matrix[unvisited_nodes]) / np.mean(distance_matrix[unvisited_nodes])
    
    visited_nodes = np.setdiff1d(np.arange(len(distance_matrix)), unvisited_nodes)
    if len(visited_nodes) > 0:
        magnetic_repulsion = 1 / (1 + np.mean(distance_matrix[unvisited_nodes][:, visited_nodes], axis=1))
    else:
        magnetic_repulsion = np.ones(len(unvisited_nodes))
    
    chaotic_resonance = np.sin(10 * progress) * np.cos(5 * spatial_dist) * (0.5 + 0.5 * np.random.rand())
    
    fractal_phase = 0.5 + 0.5 * np.sin(3 * progress + 2 * spatial_dist) * np.cos(5 * curvature)
    
    w_proximity = (0.13 - 0.04 * fractal_phase) * (1 + 0.1 * spatial_dist)
    w_directionality = (0.11 + 0.04 * fractal_phase) * (1 - 0.1 * spatial_dist)
    w_momentum = (0.09 + 0.04 * fractal_phase) * (1 + 0.1 * progress)
    w_novelty = (0.11 - 0.04 * fractal_phase) * (1 - 0.1 * progress)
    w_connectivity = (0.11 + 0.04 * fractal_phase) * (1 + 0.1 * curvature)
    w_smoothness = (0.15 - 0.04 * fractal_phase) * (1 - 0.1 * curvature)
    w_exploration = (0.05 + 0.03 * fractal_phase) * exploration_factor
    w_potential = (0.07 + 0.03 * fractal_phase) * dynamic_potential
    w_repulsion = (0.08 + 0.02 * fractal_phase) * magnetic_repulsion
    w_chaotic = (0.10 - 0.02 * fractal_phase) * chaotic_resonance
    
    combined_score = (
        w_proximity * proximity +
        w_directionality * (np.max(directionality) - directionality) +
        w_momentum * momentum +
        w_novelty * spatial_novelty +
        w_connectivity * connectivity +
        w_smoothness * smoothness +
        w_exploration * spatial_novelty +
        w_potential * dynamic_potential +
        w_repulsion * magnetic_repulsion +
        w_chaotic * chaotic_resonance
    )
    return unvisited_nodes[np.argmin(combined_score)]



# Function 10 - Score: -0.17784066824482722
{The algorithm selects the next node by combining proximity (distance from current_node), directionality (alignment toward destination_node), local clustering (average distance to nearby visited nodes), exploration bonus (inverse density of unvisited nodes), path smoothness (minimizing sharp turns), potential gain (estimated future path improvement), momentum (direction consistency with previous moves), attraction-repulsion (balancing between visited and unvisited clusters), introduces a "path diversity" factor (measuring route uniqueness), "adaptive momentum" (dynamic direction consistency based on progress), and "gradient boosting" (iterative weight adjustment based on previous performance), with a novel "topological complexity" factor (node connectivity analysis) and "dynamic exploration" (adaptive randomness based on solution quality).}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    proximity = distance_matrix[current_node, unvisited_nodes]
    directionality = distance_matrix[unvisited_nodes, destination_node]
    
    nearby_nodes = np.argsort(distance_matrix[current_node])[:5]
    nearby_visited = [n for n in nearby_nodes if n not in unvisited_nodes]
    clustering = np.mean(distance_matrix[unvisited_nodes][:, nearby_visited], axis=1) if nearby_visited else np.zeros(len(unvisited_nodes))
    
    if len(unvisited_nodes) > 1:
        exploration = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
        exploration = np.max(exploration) - exploration
    else:
        exploration = np.zeros(len(unvisited_nodes))
    
    if len(unvisited_nodes) > 1:
        smoothness = distance_matrix[current_node, unvisited_nodes] + distance_matrix[unvisited_nodes, destination_node]
    else:
        smoothness = np.zeros(len(unvisited_nodes))
    
    if len(unvisited_nodes) > 1:
        potential_gain = np.min(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    else:
        potential_gain = np.zeros(len(unvisited_nodes))
    
    if 'prev_move' in globals():
        momentum = np.array([np.dot(distance_matrix[current_node, n] - globals()['prev_move'], distance_matrix[n, destination_node]) for n in unvisited_nodes])
    else:
        momentum = np.zeros(len(unvisited_nodes))
    
    if len(unvisited_nodes) > 1:
        attraction = np.mean(distance_matrix[unvisited_nodes][:, nearby_visited], axis=1) if nearby_visited else np.zeros(len(unvisited_nodes))
        repulsion = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
        attraction_repulsion = attraction - repulsion
    else:
        attraction_repulsion = np.zeros(len(unvisited_nodes))
    
    if len(unvisited_nodes) > 1:
        path_diversity = np.std(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    else:
        path_diversity = np.zeros(len(unvisited_nodes))
    
    progress = 1 - len(unvisited_nodes) / len(distance_matrix)
    adaptive_momentum = 0.5 + 0.5 * np.sin(progress * np.pi)
    topological_complexity = np.mean(distance_matrix[unvisited_nodes] / np.max(distance_matrix))
    dynamic_exploration = np.random.uniform(0.8, 1.2) * (1 - progress)
    
    w_proximity = (0.12 + 0.04 * (1 - progress)) * dynamic_exploration * (1 + 0.1 * topological_complexity)
    w_directionality = (0.10 + 0.03 * progress) * dynamic_exploration * (1 + 0.1 * topological_complexity)
    w_clustering = (0.05 - 0.02 * progress) * dynamic_exploration * (1 - 0.1 * topological_complexity)
    w_exploration = (0.08 - 0.02 * progress) * dynamic_exploration * (1 - 0.1 * topological_complexity)
    w_smoothness = (0.08 + 0.02 * progress) * dynamic_exploration * (1 + 0.1 * topological_complexity)
    w_potential = (0.16 + 0.03 * progress) * dynamic_exploration * (1 + 0.1 * topological_complexity)
    w_momentum = (0.04 + 0.02 * progress) * adaptive_momentum * (1 - 0.1 * topological_complexity)
    w_att_rep = (0.10 + 0.02 * progress) * dynamic_exploration * (1 + 0.1 * topological_complexity)
    w_diversity = (0.14 - 0.04 * progress) * dynamic_exploration * (1 - 0.1 * topological_complexity)
    
    combined_score = (
        w_proximity * proximity +
        w_directionality * (np.max(directionality) - directionality) +
        w_clustering * clustering +
        w_exploration * exploration +
        w_smoothness * smoothness +
        w_potential * potential_gain +
        w_momentum * (np.max(momentum) - momentum) +
        w_att_rep * attraction_repulsion +
        w_diversity * path_diversity
    )
    globals()['prev_move'] = distance_matrix[current_node, unvisited_nodes[np.argmin(combined_score)]] - distance_matrix[current_node, current_node]
    return unvisited_nodes[np.argmin(combined_score)]



