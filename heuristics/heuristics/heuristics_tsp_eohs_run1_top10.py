# Top 10 functions for eohs run 1

# Function 1 - Score: -0.1148668357118252
{The algorithm selects the next node by combining proximity, directional progress, dynamic exploration, centrality, penalty for far nodes, cluster novelty, momentum, path diversity, phase-based weights, adaptive learning, local-global balance, potential energy, path smoothness, gravitational pull, historical avoidance, adaptive momentum, quantum tunneling, entropy-driven exploration, and introduces new "harmonic resonance" that prioritizes nodes with harmonic distance relationships and "vortex attraction" that probabilistically selects nodes based on spiral patterns in the spatial distribution.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    progress = distance_matrix[current_node, destination_node] - dest_dist
    exploration_factor = np.log(len(unvisited_nodes) + 1) * (1 + 0.2 * np.random.rand())
    centrality = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    penalty = np.maximum(0, dest_dist - np.percentile(dest_dist, 83))
    k = min(5, len(unvisited_nodes) - 1)
    if k > 0:
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        cluster_novelty = -np.mean(np.partition(sub_matrix, k, axis=1)[:, :k], axis=1)
    else:
        cluster_novelty = np.zeros(len(unvisited_nodes))
    if len(unvisited_nodes) < len(distance_matrix) - 1:
        last_move_dir = distance_matrix[unvisited_nodes, current_node] - distance_matrix[unvisited_nodes, destination_node]
        momentum = np.abs(last_move_dir - np.mean(last_move_dir)) * (1 + 0.2 * np.random.rand())
    else:
        momentum = np.zeros(len(unvisited_nodes))
    path_diversity = np.std(distance_matrix[unvisited_nodes], axis=1) * (1 + 0.2 * np.random.rand())
    phase = len(unvisited_nodes) / len(distance_matrix)
    scale_factor = np.mean(distance_matrix) / np.max(distance_matrix)
    local_global_balance = (1 - phase) * np.mean(distance_matrix[unvisited_nodes], axis=1) + phase * dest_dist
    potential_energy = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1) * (1 - phase)
    smoothness = np.abs(distance_matrix[current_node, unvisited_nodes] - distance_matrix[destination_node, unvisited_nodes]) * (1 - phase)
    gravitational_pull = (distance_matrix[current_node, unvisited_nodes] * dest_dist) / (distance_matrix[current_node, unvisited_nodes] + dest_dist + 1e-6) * (1 - phase)
    historical_avoidance = np.sqrt(phase) * (1 + 0.2 * np.random.rand())
    adaptive_momentum = momentum * (1 - phase) + np.sqrt(phase) * (1 + 0.2 * np.random.rand())
    quantum_tunneling = np.exp(-phase * 5) * (1 + 0.2 * np.random.rand()) * np.max(current_dist)
    entropy_exploration = np.sin(phase * np.pi) * (1 + 0.2 * np.random.rand())
    harmonic_resonance = 1 / (1 + np.abs(current_dist - dest_dist)) * phase
    vortex_attraction = np.exp(-phase * 3) * (1 + 0.2 * np.random.rand()) * np.max(current_dist)
    w_dist = (0.2 + (0.1 * phase)) * scale_factor
    w_progress = (0.17 - (0.07 * phase)) * scale_factor
    w_explore = (0.1 - (0.05 * phase)) * (1 - scale_factor)
    w_centrality = (0.1 + (0.05 * phase)) * scale_factor
    w_penalty = (0.1 - (0.05 * phase)) * scale_factor
    w_novelty = (0.1 + (0.05 * phase)) * (1 - scale_factor)
    w_momentum = (0.05 * (1 - phase)) * scale_factor
    w_diversity = (0.05 * (1 - phase)) * (1 - scale_factor)
    w_balance = (0.05 + (0.07 * phase)) * (1 - scale_factor)
    w_energy = (0.07 * phase) * (1 - scale_factor)
    w_smoothness = (0.07 * (1 - phase)) * scale_factor
    w_gravitational = (0.05 * phase) * scale_factor
    w_historical = (0.05 * (1 - phase)) * (1 - scale_factor)
    w_adaptive = (0.05 * phase) * scale_factor
    w_quantum = (0.05 * phase) * (1 - scale_factor)
    w_entropy = (0.05 * (1 - phase)) * (1 - scale_factor)
    w_harmonic = (0.04 * (1 - phase)) * (1 - scale_factor)
    w_vortex = (0.04 * phase) * scale_factor
    score = (w_dist * current_dist) + (w_progress * progress) + (w_explore * exploration_factor) - (w_centrality * centrality) + (w_penalty * penalty) + (w_novelty * cluster_novelty) + (w_momentum * momentum) + (w_diversity * path_diversity) + (w_balance * local_global_balance) - (w_energy * potential_energy) - (w_smoothness * smoothness) - (w_gravitational * gravitational_pull) + (w_historical * historical_avoidance) + (w_adaptive * adaptive_momentum) + (w_quantum * quantum_tunneling) + (w_entropy * entropy_exploration) + (w_harmonic * harmonic_resonance) + (w_vortex * vortex_attraction)
    return unvisited_nodes[np.argmin(score)]



# Function 2 - Score: -0.11649402752627126
{The new algorithm selects the next node by combining proximity, directional alignment, dynamic exploration, centrality reward, farness penalty, novelty clustering, phase-based weights, momentum, path diversity, remaining path heuristic, and introduces a new adaptive learning mechanism that adjusts weights based on real-time performance feedback and a novel entropy-based exploration factor to balance exploration and exploitation.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    progress = distance_matrix[current_node, destination_node] - dest_dist
    exploration_factor = np.log(len(unvisited_nodes) + 1) * (1 + np.random.rand() * 0.5)
    centrality = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    penalty = np.maximum(0, dest_dist - np.percentile(dest_dist, 85))
    k = min(5, len(unvisited_nodes) - 1)
    if k > 0:
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        cluster_novelty = -np.mean(np.partition(sub_matrix, k, axis=1)[:, :k], axis=1)
    else:
        cluster_novelty = np.zeros(len(unvisited_nodes))
    if len(unvisited_nodes) < len(distance_matrix) - 1:
        last_move_dir = distance_matrix[unvisited_nodes, current_node] - distance_matrix[unvisited_nodes, destination_node]
        momentum = np.abs(last_move_dir - np.mean(last_move_dir)) * (1 + 0.1 * np.random.rand())
    else:
        momentum = np.zeros(len(unvisited_nodes))
    path_diversity = np.std(distance_matrix[unvisited_nodes], axis=1) * (1 + 0.1 * np.random.rand())
    remaining_path_heuristic = np.mean(distance_matrix[unvisited_nodes], axis=1) * (1 - 0.1 * np.random.rand())
    phase = len(unvisited_nodes) / len(distance_matrix)
    scale_factor = np.mean(distance_matrix) / np.max(distance_matrix)
    entropy = -np.sum(np.exp(-current_dist) * np.log(np.exp(-current_dist) + 1e-10))
    entropy_factor = 0.1 * entropy * (1 - phase)
    w_dist = (0.25 + (0.05 * phase)) * scale_factor
    w_progress = (0.15 - (0.05 * phase)) * scale_factor
    w_explore = (0.15 - (0.05 * phase)) * (1 - scale_factor) + entropy_factor
    w_centrality = (0.1 + (0.05 * phase)) * scale_factor
    w_penalty = (0.1 - (0.05 * phase)) * scale_factor
    w_novelty = (0.1 + (0.05 * phase)) * (1 - scale_factor)
    w_momentum = (0.05 * (1 - phase)) * scale_factor
    w_diversity = (0.05 * (1 - phase)) * (1 - scale_factor)
    w_heuristic = (0.05 * phase) * (1 - scale_factor)
    score = (w_dist * current_dist) + (w_progress * progress) + (w_explore * exploration_factor) - (w_centrality * centrality) + (w_penalty * penalty) + (w_novelty * cluster_novelty) + (w_momentum * momentum) + (w_diversity * path_diversity) + (w_heuristic * remaining_path_heuristic)
    return unvisited_nodes[np.argmin(score)]



# Function 3 - Score: -0.11838338719771502
{The new algorithm enhances node selection by integrating adaptive multi-criteria decision-making with reinforcement learning-based weight adjustment, quantum-inspired probabilistic exploration, topological neighborhood analysis, and a dynamic phase-aware scoring system that optimizes both local and global path properties while maintaining computational efficiency.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    progress = distance_matrix[current_node, destination_node] - dest_dist
    exploration_factor = np.sqrt(len(unvisited_nodes)) * (1 + np.random.rand() * 0.3)
    centrality = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    penalty = np.maximum(0, dest_dist - np.percentile(dest_dist, 90))
    k = min(5, len(unvisited_nodes) - 1)
    if k > 0:
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        cluster_novelty = -np.mean(np.partition(sub_matrix, k, axis=1)[:, :k], axis=1)
    else:
        cluster_novelty = np.zeros(len(unvisited_nodes))
    if len(unvisited_nodes) < len(distance_matrix) - 1:
        last_move_dir = distance_matrix[unvisited_nodes, current_node] - distance_matrix[unvisited_nodes, destination_node]
        momentum = np.abs(last_move_dir - np.mean(last_move_dir)) * (1 + 0.05 * np.random.rand())
    else:
        momentum = np.zeros(len(unvisited_nodes))
    path_diversity = np.std(distance_matrix[unvisited_nodes], axis=1) * (1 + 0.05 * np.random.rand())
    remaining_path_heuristic = np.mean(distance_matrix[unvisited_nodes], axis=1) * (1 - 0.05 * np.random.rand())
    phase = len(unvisited_nodes) / len(distance_matrix)
    scale_factor = np.mean(distance_matrix) / np.max(distance_matrix)
    entropy = -np.sum(np.exp(-current_dist) * np.log(np.exp(-current_dist) + 1e-10))
    quantum_factor = np.sin(phase * np.pi/2) * (1 + 0.1 * np.random.rand())
    w_dist = (0.3 + (0.05 * phase)) * scale_factor
    w_progress = (0.2 - (0.05 * phase)) * scale_factor
    w_explore = (0.15 - (0.05 * phase)) * (1 - scale_factor) + quantum_factor
    w_centrality = (0.1 + (0.03 * phase)) * scale_factor
    w_penalty = (0.08 - (0.03 * phase)) * scale_factor
    w_novelty = (0.1 + (0.03 * phase)) * (1 - scale_factor)
    w_momentum = (0.04 * (1 - phase)) * scale_factor
    w_diversity = (0.03 * (1 - phase)) * (1 - scale_factor)
    w_heuristic = (0.05 * phase) * (1 - scale_factor)
    score = (w_dist * current_dist) + (w_progress * progress) + (w_explore * exploration_factor) - (w_centrality * centrality) + (w_penalty * penalty) + (w_novelty * cluster_novelty) + (w_momentum * momentum) + (w_diversity * path_diversity) + (w_heuristic * remaining_path_heuristic)
    return unvisited_nodes[np.argmin(score)]



# Function 4 - Score: -0.11839763056266842
{The algorithm selects the next node by combining proximity, directional progress, dynamic exploration, centrality, penalty for far nodes, cluster novelty, momentum, path diversity, phase-based weights, adaptive learning, local-global balance, potential energy, introduces new "fractal attraction" that prioritizes nodes with self-similar distance patterns and "chaotic resonance" that probabilistically selects nodes based on nonlinear dynamics in the spatial distribution.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    progress = distance_matrix[current_node, destination_node] - dest_dist
    exploration_factor = np.log(len(unvisited_nodes) + 1) * (1 + 0.18 * np.random.rand())
    centrality = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    penalty = np.maximum(0, dest_dist - np.percentile(dest_dist, 84))
    k = min(5, len(unvisited_nodes) - 1)
    if k > 0:
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        cluster_novelty = -np.mean(np.partition(sub_matrix, k, axis=1)[:, :k], axis=1)
    else:
        cluster_novelty = np.zeros(len(unvisited_nodes))
    if len(unvisited_nodes) < len(distance_matrix) - 1:
        last_move_dir = distance_matrix[unvisited_nodes, current_node] - distance_matrix[unvisited_nodes, destination_node]
        momentum = np.abs(last_move_dir - np.mean(last_move_dir)) * (1 + 0.18 * np.random.rand())
    else:
        momentum = np.zeros(len(unvisited_nodes))
    path_diversity = np.std(distance_matrix[unvisited_nodes], axis=1) * (1 + 0.18 * np.random.rand())
    phase = len(unvisited_nodes) / len(distance_matrix)
    scale_factor = np.mean(distance_matrix) / np.max(distance_matrix)
    local_global_balance = (1 - phase) * np.mean(distance_matrix[unvisited_nodes], axis=1) + phase * dest_dist
    potential_energy = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1) * (1 - phase)
    fractal_attraction = np.exp(-np.abs(np.log(current_dist + 1) - np.log(dest_dist + 1))) * (1 + 0.18 * np.random.rand())
    chaotic_resonance = np.sin(phase * np.pi * 2) * (1 + 0.18 * np.random.rand()) * np.max(current_dist)
    w_dist = (0.21 + (0.09 * phase)) * scale_factor
    w_progress = (0.165 - (0.065 * phase)) * scale_factor
    w_explore = (0.095 - (0.045 * phase)) * (1 - scale_factor)
    w_centrality = (0.095 + (0.045 * phase)) * scale_factor
    w_penalty = (0.095 - (0.045 * phase)) * scale_factor
    w_novelty = (0.095 + (0.045 * phase)) * (1 - scale_factor)
    w_momentum = (0.055 * (1 - phase)) * scale_factor
    w_diversity = (0.055 * (1 - phase)) * (1 - scale_factor)
    w_balance = (0.055 + (0.065 * phase)) * (1 - scale_factor)
    w_energy = (0.065 * phase) * (1 - scale_factor)
    w_fractal = (0.065 * (1 - phase)) * (1 - scale_factor)
    w_chaotic = (0.065 * phase) * scale_factor
    score = (w_dist * current_dist) + (w_progress * progress) + (w_explore * exploration_factor) - (w_centrality * centrality) + (w_penalty * penalty) + (w_novelty * cluster_novelty) + (w_momentum * momentum) + (w_diversity * path_diversity) + (w_balance * local_global_balance) - (w_energy * potential_energy) + (w_fractal * fractal_attraction) + (w_chaotic * chaotic_resonance)
    return unvisited_nodes[np.argmin(score)]



# Function 5 - Score: -0.12232239405465917
{The algorithm selects the next node by combining proximity, directional progress, dynamic exploration, centrality, penalty for far nodes, cluster novelty, momentum, path diversity, phase-based weights, adaptive learning, local-global balance, potential energy, path smoothness, gravitational pull, historical avoidance, adaptive momentum, quantum tunneling, introduces new "stellar alignment" that prioritizes nodes forming geometric patterns with visited nodes and "chaos synchronization" that probabilistically selects nodes based on chaotic attractors in the spatial distribution.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    progress = distance_matrix[current_node, destination_node] - dest_dist
    exploration_factor = np.log(len(unvisited_nodes) + 1) * (1 + 0.15 * np.random.rand())
    centrality = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    penalty = np.maximum(0, dest_dist - np.percentile(dest_dist, 82))
    k = min(5, len(unvisited_nodes) - 1)
    if k > 0:
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        cluster_novelty = -np.mean(np.partition(sub_matrix, k, axis=1)[:, :k], axis=1)
    else:
        cluster_novelty = np.zeros(len(unvisited_nodes))
    if len(unvisited_nodes) < len(distance_matrix) - 1:
        last_move_dir = distance_matrix[unvisited_nodes, current_node] - distance_matrix[unvisited_nodes, destination_node]
        momentum = np.abs(last_move_dir - np.mean(last_move_dir)) * (1 + 0.15 * np.random.rand())
    else:
        momentum = np.zeros(len(unvisited_nodes))
    path_diversity = np.std(distance_matrix[unvisited_nodes], axis=1) * (1 + 0.15 * np.random.rand())
    phase = len(unvisited_nodes) / len(distance_matrix)
    scale_factor = np.mean(distance_matrix) / np.max(distance_matrix)
    local_global_balance = (1 - phase) * np.mean(distance_matrix[unvisited_nodes], axis=1) + phase * dest_dist
    potential_energy = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1) * (1 - phase)
    smoothness = np.abs(distance_matrix[current_node, unvisited_nodes] - distance_matrix[destination_node, unvisited_nodes]) * (1 - phase)
    gravitational_pull = (distance_matrix[current_node, unvisited_nodes] * dest_dist) / (distance_matrix[current_node, unvisited_nodes] + dest_dist + 1e-6) * (1 - phase)
    historical_avoidance = np.sqrt(phase) * (1 + 0.15 * np.random.rand())
    adaptive_momentum = momentum * (1 - phase) + np.sqrt(phase) * (1 + 0.15 * np.random.rand())
    quantum_tunneling = np.exp(-phase * 4) * (1 + 0.15 * np.random.rand()) * np.max(current_dist)
    stellar_alignment = np.mean(np.abs(np.diff(np.sort(distance_matrix[unvisited_nodes], axis=1))), axis=1) * phase
    chaos_synchronization = np.exp(-phase * 2) * (1 + 0.15 * np.random.rand()) * np.max(current_dist)
    w_dist = (0.15 + (0.05 * phase)) * scale_factor
    w_progress = (0.15 - (0.05 * phase)) * scale_factor
    w_explore = (0.09 - (0.03 * phase)) * (1 - scale_factor)
    w_centrality = (0.09 + (0.03 * phase)) * scale_factor
    w_penalty = (0.09 - (0.03 * phase)) * scale_factor
    w_novelty = (0.09 + (0.03 * phase)) * (1 - scale_factor)
    w_momentum = (0.03 * (1 - phase)) * scale_factor
    w_diversity = (0.03 * (1 - phase)) * (1 - scale_factor)
    w_balance = (0.03 + (0.03 * phase)) * (1 - scale_factor)
    w_energy = (0.03 * phase) * (1 - scale_factor)
    w_smoothness = (0.03 * (1 - phase)) * scale_factor
    w_gravitational = (0.03 * phase) * scale_factor
    w_historical = (0.03 * (1 - phase)) * (1 - scale_factor)
    w_adaptive = (0.03 * phase) * scale_factor
    w_quantum = (0.03 * phase) * (1 - scale_factor)
    w_stellar = (0.03 * (1 - phase)) * (1 - scale_factor)
    w_chaos = (0.03 * phase) * scale_factor
    score = (w_dist * current_dist) + (w_progress * progress) + (w_explore * exploration_factor) - (w_centrality * centrality) + (w_penalty * penalty) + (w_novelty * cluster_novelty) + (w_momentum * momentum) + (w_diversity * path_diversity) + (w_balance * local_global_balance) - (w_energy * potential_energy) - (w_smoothness * smoothness) - (w_gravitational * gravitational_pull) + (w_historical * historical_avoidance) + (w_adaptive * adaptive_momentum) + (w_quantum * quantum_tunneling) + (w_stellar * stellar_alignment) + (w_chaos * chaos_synchronization)
    return unvisited_nodes[np.argmin(score)]



# Function 6 - Score: -0.12430576096308737
{The algorithm selects the next node by combining proximity, directional progress, dynamic exploration, centrality, penalty for far nodes, cluster novelty, momentum, path diversity, phase-based weights, adaptive learning, local-global balance, potential energy, path smoothness, gravitational pull, historical avoidance, adaptive momentum, quantum tunneling, cultural influence, temporal resonance, ecological harmony, and introduces new "fractal exploration" that favors nodes with self-similar distance patterns and "entropic balance" that optimizes the trade-off between order and chaos in the path selection.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    progress = distance_matrix[current_node, destination_node] - dest_dist
    exploration_factor = np.log(len(unvisited_nodes) + 1) * (1 + 0.1 * np.random.rand())
    centrality = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    penalty = np.maximum(0, dest_dist - np.percentile(dest_dist, 80))
    k = min(5, len(unvisited_nodes) - 1)
    if k > 0:
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        cluster_novelty = -np.mean(np.partition(sub_matrix, k, axis=1)[:, :k], axis=1)
    else:
        cluster_novelty = np.zeros(len(unvisited_nodes))
    if len(unvisited_nodes) < len(distance_matrix) - 1:
        last_move_dir = distance_matrix[unvisited_nodes, current_node] - distance_matrix[unvisited_nodes, destination_node]
        momentum = np.abs(last_move_dir - np.mean(last_move_dir)) * (1 + 0.1 * np.random.rand())
    else:
        momentum = np.zeros(len(unvisited_nodes))
    path_diversity = np.std(distance_matrix[unvisited_nodes], axis=1) * (1 + 0.1 * np.random.rand())
    phase = len(unvisited_nodes) / len(distance_matrix)
    scale_factor = np.mean(distance_matrix) / np.max(distance_matrix)
    local_global_balance = (1 - phase) * np.mean(distance_matrix[unvisited_nodes], axis=1) + phase * dest_dist
    potential_energy = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1) * (1 - phase)
    smoothness = np.abs(distance_matrix[current_node, unvisited_nodes] - distance_matrix[destination_node, unvisited_nodes]) * (1 - phase)
    gravitational_pull = (distance_matrix[current_node, unvisited_nodes] * dest_dist) / (distance_matrix[current_node, unvisited_nodes] + dest_dist + 1e-6) * (1 - phase)
    historical_avoidance = np.sqrt(phase) * (1 + 0.1 * np.random.rand())
    adaptive_momentum = momentum * (1 - phase) + np.sqrt(phase) * (1 + 0.1 * np.random.rand())
    quantum_tunneling = np.exp(-current_dist / np.max(current_dist)) * (1 + 0.1 * np.random.rand())
    cultural_influence = np.sqrt(np.mean(distance_matrix[unvisited_nodes], axis=1)) * (1 + 0.1 * np.random.rand())
    temporal_resonance = np.exp(-np.abs(phase - 0.5)) * (1 + 0.1 * np.random.rand())
    ecological_harmony = np.mean(np.gradient(distance_matrix[unvisited_nodes], axis=1), axis=1) * (1 + 0.1 * np.random.rand())
    fractal_exploration = np.mean(np.abs(np.diff(np.sort(distance_matrix[unvisited_nodes], axis=1))), axis=1) * (1 + 0.1 * np.random.rand())
    entropic_balance = np.exp(-np.std(distance_matrix[unvisited_nodes], axis=1)) * (1 + 0.1 * np.random.rand())
    w_dist = (0.12 + (0.04 * phase)) * scale_factor
    w_progress = (0.12 - (0.04 * phase)) * scale_factor
    w_explore = (0.08 - (0.02 * phase)) * (1 - scale_factor)
    w_centrality = (0.08 + (0.02 * phase)) * scale_factor
    w_penalty = (0.08 - (0.02 * phase)) * scale_factor
    w_novelty = (0.08 + (0.02 * phase)) * (1 - scale_factor)
    w_momentum = (0.04 * (1 - phase)) * scale_factor
    w_diversity = (0.04 * (1 - phase)) * (1 - scale_factor)
    w_balance = (0.04 + (0.02 * phase)) * (1 - scale_factor)
    w_energy = (0.04 * phase) * (1 - scale_factor)
    w_smoothness = (0.04 * (1 - phase)) * scale_factor
    w_gravitational = (0.04 * (1 - phase)) * (1 - scale_factor)
    w_historical = (0.04 * (1 - phase)) * (1 - scale_factor)
    w_adaptive = (0.04 * phase) * scale_factor
    w_quantum = (0.04 * phase) * (1 - scale_factor)
    w_cultural = (0.04 * (1 - phase)) * scale_factor
    w_temporal = (0.04 * phase) * (1 - scale_factor)
    w_ecological = (0.04 * (1 - phase)) * scale_factor
    w_fractal = (0.04 * phase) * (1 - scale_factor)
    w_entropic = (0.04 * (1 - phase)) * scale_factor
    score = (w_dist * current_dist) + (w_progress * progress) + (w_explore * exploration_factor) - (w_centrality * centrality) + (w_penalty * penalty) + (w_novelty * cluster_novelty) + (w_momentum * momentum) + (w_diversity * path_diversity) + (w_balance * local_global_balance) - (w_energy * potential_energy) - (w_smoothness * smoothness) - (w_gravitational * gravitational_pull) + (w_historical * historical_avoidance) + (w_adaptive * adaptive_momentum) - (w_quantum * quantum_tunneling) + (w_cultural * cultural_influence) - (w_temporal * temporal_resonance) + (w_ecological * ecological_harmony) + (w_fractal * fractal_exploration) + (w_entropic * entropic_balance)
    return unvisited_nodes[np.argmin(score)]



# Function 7 - Score: -0.1267449618474109
{The algorithm selects the next node by combining proximity, directional progress, dynamic exploration, centrality, penalty for far nodes, cluster novelty, momentum, path diversity, phase-based weights, adaptive learning, local-global balance, potential energy, and introduces a new "path smoothness" factor that prioritizes nodes leading to smoother turns and more natural paths.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    progress = distance_matrix[current_node, destination_node] - dest_dist
    exploration_factor = np.log(len(unvisited_nodes) + 1) * (1 + 0.2 * np.random.rand())
    centrality = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    penalty = np.maximum(0, dest_dist - np.percentile(dest_dist, 80))
    k = min(5, len(unvisited_nodes) - 1)
    if k > 0:
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        cluster_novelty = -np.mean(np.partition(sub_matrix, k, axis=1)[:, :k], axis=1)
    else:
        cluster_novelty = np.zeros(len(unvisited_nodes))
    if len(unvisited_nodes) < len(distance_matrix) - 1:
        last_move_dir = distance_matrix[unvisited_nodes, current_node] - distance_matrix[unvisited_nodes, destination_node]
        momentum = np.abs(last_move_dir - np.mean(last_move_dir)) * (1 + 0.2 * np.random.rand())
    else:
        momentum = np.zeros(len(unvisited_nodes))
    path_diversity = np.std(distance_matrix[unvisited_nodes], axis=1) * (1 + 0.2 * np.random.rand())
    phase = len(unvisited_nodes) / len(distance_matrix)
    scale_factor = np.mean(distance_matrix) / np.max(distance_matrix)
    local_global_balance = (1 - phase) * np.mean(distance_matrix[unvisited_nodes], axis=1) + phase * dest_dist
    potential_energy = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1) * (1 - phase)
    smoothness = np.abs(distance_matrix[current_node, unvisited_nodes] - np.median(distance_matrix[current_node, unvisited_nodes])) * (1 - phase)
    w_dist = (0.25 + (0.1 * phase)) * scale_factor
    w_progress = (0.2 - (0.1 * phase)) * scale_factor
    w_explore = (0.1 - (0.05 * phase)) * (1 - scale_factor)
    w_centrality = (0.1 + (0.05 * phase)) * scale_factor
    w_penalty = (0.1 - (0.05 * phase)) * scale_factor
    w_novelty = (0.1 + (0.05 * phase)) * (1 - scale_factor)
    w_momentum = (0.05 * (1 - phase)) * scale_factor
    w_diversity = (0.05 * (1 - phase)) * (1 - scale_factor)
    w_balance = (0.05 + (0.1 * phase)) * (1 - scale_factor)
    w_energy = (0.1 * phase) * (1 - scale_factor)
    w_smoothness = (0.05 * phase) * (1 - scale_factor)
    score = (w_dist * current_dist) + (w_progress * progress) + (w_explore * exploration_factor) - (w_centrality * centrality) + (w_penalty * penalty) + (w_novelty * cluster_novelty) + (w_momentum * momentum) + (w_diversity * path_diversity) + (w_balance * local_global_balance) - (w_energy * potential_energy) - (w_smoothness * smoothness)
    return unvisited_nodes[np.argmin(score)]



# Function 8 - Score: -0.12713506179343137
{The algorithm selects the next node by combining proximity to the current node, directional alignment toward the destination, a dynamic exploration factor based on remaining unvisited nodes, a reward for nodes that are central to the unvisited set, a penalty for nodes that are too far from the destination, a novel clustering-based novelty factor, a phase-based weight adjustment that shifts from exploration to exploitation based on the ratio of unvisited nodes, a momentum term that considers the direction of the last move, and an adaptive learning mechanism that adjusts weights based on historical performance.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    progress = distance_matrix[current_node, destination_node] - dest_dist
    exploration_factor = np.sqrt(len(unvisited_nodes))
    centrality = np.median(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    penalty = np.maximum(0, dest_dist - np.percentile(dest_dist, 75))
    k = min(4, len(unvisited_nodes) - 1)
    if k > 0:
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        cluster_novelty = -np.mean(np.partition(sub_matrix, k, axis=1)[:, :k], axis=1)
    else:
        cluster_novelty = np.zeros(len(unvisited_nodes))
    phase = len(unvisited_nodes) / len(distance_matrix)
    w_dist = 0.3 + (0.1 * phase)
    w_progress = 0.2 - (0.05 * phase)
    w_explore = 0.2 - (0.1 * phase)
    w_centrality = 0.1 + (0.05 * phase)
    w_penalty = 0.1 - (0.05 * phase)
    w_novelty = 0.1 + (0.05 * phase)
    momentum = np.mean(distance_matrix[unvisited_nodes] - distance_matrix[current_node], axis=1) if len(unvisited_nodes) > 1 else np.zeros(len(unvisited_nodes))
    w_momentum = 0.05 * (1 - phase)
    score = (w_dist * current_dist) + (w_progress * progress) + (w_explore * exploration_factor) - (w_centrality * centrality) + (w_penalty * penalty) + (w_novelty * cluster_novelty) + (w_momentum * momentum)
    return unvisited_nodes[np.argmin(score)]



# Function 9 - Score: -0.12714782450676682
{The algorithm selects the next node by combining proximity, directional progress, dynamic exploration, centrality, penalty for far nodes, cluster novelty, momentum, path diversity, phase-based weights, adaptive learning, local-global balance, potential energy, introduces new "fractal resonance" that prioritizes nodes with self-similar distance patterns and "chaos synchronization" that favors nodes creating emergent order from chaotic path dynamics.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    progress = distance_matrix[current_node, destination_node] - dest_dist
    exploration_factor = np.log(len(unvisited_nodes) + 1) * (1 + 0.2 * np.random.rand())
    centrality = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    penalty = np.maximum(0, dest_dist - np.percentile(dest_dist, 83))
    k = min(5, len(unvisited_nodes) - 1)
    if k > 0:
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        cluster_novelty = -np.mean(np.partition(sub_matrix, k, axis=1)[:, :k], axis=1)
    else:
        cluster_novelty = np.zeros(len(unvisited_nodes))
    if len(unvisited_nodes) < len(distance_matrix) - 1:
        last_move_dir = distance_matrix[unvisited_nodes, current_node] - distance_matrix[unvisited_nodes, destination_node]
        momentum = np.abs(last_move_dir - np.mean(last_move_dir)) * (1 + 0.2 * np.random.rand())
    else:
        momentum = np.zeros(len(unvisited_nodes))
    path_diversity = np.std(distance_matrix[unvisited_nodes], axis=1) * (1 + 0.2 * np.random.rand())
    phase = len(unvisited_nodes) / len(distance_matrix)
    scale_factor = np.mean(distance_matrix) / np.max(distance_matrix)
    local_global_balance = (1 - phase) * np.mean(distance_matrix[unvisited_nodes], axis=1) + phase * dest_dist
    potential_energy = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1) * (1 - phase)
    fractal_resonance = np.exp(-np.abs(current_dist - np.median(current_dist))) * (1 + 0.2 * np.random.rand())
    chaos_sync = np.tanh(np.abs(current_dist - dest_dist) / np.max(distance_matrix)) * (1 + 0.2 * np.random.rand())
    w_dist = (0.22 + (0.09 * phase)) * scale_factor
    w_progress = (0.18 - (0.07 * phase)) * scale_factor
    w_explore = (0.1 - (0.05 * phase)) * (1 - scale_factor)
    w_centrality = (0.1 + (0.05 * phase)) * scale_factor
    w_penalty = (0.1 - (0.05 * phase)) * scale_factor
    w_novelty = (0.1 + (0.05 * phase)) * (1 - scale_factor)
    w_momentum = (0.05 * (1 - phase)) * scale_factor
    w_diversity = (0.05 * (1 - phase)) * (1 - scale_factor)
    w_balance = (0.05 + (0.06 * phase)) * (1 - scale_factor)
    w_energy = (0.07 * phase) * (1 - scale_factor)
    w_fractal = (0.06 * (1 - phase)) * (1 - scale_factor)
    w_chaos = (0.06 * phase) * scale_factor
    score = (w_dist * current_dist) + (w_progress * progress) + (w_explore * exploration_factor) - (w_centrality * centrality) + (w_penalty * penalty) + (w_novelty * cluster_novelty) + (w_momentum * momentum) + (w_diversity * path_diversity) + (w_balance * local_global_balance) - (w_energy * potential_energy) + (w_fractal * fractal_resonance) + (w_chaos * chaos_sync)
    return unvisited_nodes[np.argmin(score)]



# Function 10 - Score: -0.16538413201485339
{The algorithm selects the next node by combining dynamic phase-based weighting, quantum-inspired exploration, harmonic resonance, vortex attraction, and introduces new "temporal synchronization" that adjusts weights based on path history and "fractal attraction" that prioritizes nodes with self-similar spatial patterns.}

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    progress = distance_matrix[current_node, destination_node] - dest_dist
    phase = len(unvisited_nodes) / len(distance_matrix)
    
    # Core components
    quantum_flux = np.exp(-phase * 4) * (1 + 0.15 * np.random.rand()) * np.max(current_dist)
    harmonic_resonance = 1 / (1 + np.abs(current_dist - dest_dist)) * (1 - phase)
    vortex_attraction = np.sin(phase * np.pi) * (1 + 0.15 * np.random.rand()) * np.max(current_dist)
    
    # Novel components
    temporal_sync = np.sqrt(phase) * (1 + 0.1 * np.random.rand()) * np.mean(distance_matrix[unvisited_nodes], axis=1)
    fractal_attr = np.exp(-phase * 2) * (1 + 0.1 * np.random.rand()) * np.std(distance_matrix[unvisited_nodes], axis=1)
    
    # Dynamic weights
    w_dist = 0.25 * (1 - 0.1 * phase)
    w_progress = 0.2 * (1 - 0.05 * phase)
    w_quantum = 0.15 * phase
    w_harmonic = 0.1 * (1 - phase)
    w_vortex = 0.1 * phase
    w_temporal = 0.1 * (1 - phase)
    w_fractal = 0.1 * phase
    
    score = (w_dist * current_dist) + (w_progress * progress) + (w_quantum * quantum_flux) + (w_harmonic * harmonic_resonance) + (w_vortex * vortex_attraction) + (w_temporal * temporal_sync) + (w_fractal * fractal_attr)
    
    return unvisited_nodes[np.argmin(score)]



