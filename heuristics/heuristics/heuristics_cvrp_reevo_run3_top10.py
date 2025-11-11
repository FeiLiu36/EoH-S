# Top 10 functions for reevo run 3

# Function 1 - Score: -0.2920691202083868
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
    if not unvisited_nodes.size:
        return depot

    epsilon = 1e-20
    current_distances = distance_matrix[current_node, unvisited_nodes]
    total_demand = demands[unvisited_nodes].sum()
    
    # Refined spatial quantization with adaptive binning
    spatial_bins = np.percentile(current_distances, np.linspace(1, 99, 17))
    spatial_density = np.digitize(current_distances, spatial_bins) / 17.0
    spatial_factor = 0.93 + 0.38 * np.mean(spatial_density**2.6)  # Balanced spatial influence
    
    # Enhanced capacity threshold with smoother adaptation
    capacity_ratio = rest_capacity / (total_demand + epsilon)
    threshold = 0.47 + 0.4 * np.tanh(4.5 * (spatial_factor - 0.93))
    pressure = np.tanh(45 * (capacity_ratio - threshold))  # More responsive pressure
    
    # Optimized weight modulation system
    base_weights = np.array([0.55, 0.43, 0.02])  # Slightly adjusted base weights
    mod_factors = np.array([
        1 - 0.65 * pressure * np.exp(-spatial_factor**2.7),  # Proximity modulation
        1 + 1.75 * pressure * (1 + 0.9*spatial_factor**2.8),  # Demand modulation
        1 + 0.25 * pressure * (1 + 0.15*spatial_factor)  # Depot modulation
    ])
    weights = np.clip(base_weights * mod_factors, 0.16, 0.74)  # Optimal dynamic range
    weights = weights**2.7 / np.sum(weights**2.7)  # Enhanced power normalization
    
    best_node = -1
    best_score = -float('inf')
    
    for node in unvisited_nodes:
        demand = demands[node]
        if demand > rest_capacity:
            continue
            
        dist = distance_matrix[current_node, node]
        depot_dist = distance_matrix[node, depot]
        bin_idx = np.digitize(dist, spatial_bins)
        
        # Precision-tuned scoring components
        proximity = (1.0 / (dist**0.93 + epsilon)) * (1 + 0.82*np.exp(-(dist/spatial_bins[9])**1.85))
        urgency = (demand / (rest_capacity + epsilon)) ** (6.7 + 3.7*np.tanh(4.7*pressure))
        depot_prox = np.exp(-0.32 * depot_dist * (1 + 0.65*pressure))
        
        # Strategic bonuses with adaptive scaling
        spatial_bonus = 1.0 + 0.88 * (1 - bin_idx/17.0)**2.9
        capacity_bonus = 1.0 + (demand/rest_capacity) ** (15.0 + 11.0*pressure) * spatial_bonus
        
        # Optimized composite score
        score = (
            weights[0] * proximity**(2.5 - 1.25*pressure) +
            weights[1] * urgency**(4.2 + 2.2*pressure) +
            weights[2] * depot_prox
        ) * capacity_bonus**(3.0 - 1.6*pressure)
        
        # Streamlined tie-breaking hierarchy
        if score > best_score + epsilon:
            best_score = score
            best_node = node
        elif abs(score - best_score) <= epsilon:
            if best_node == -1 or demand > demands[best_node]:
                best_node = node
            elif demand == demands[best_node] and dist < distance_matrix[current_node, best_node]:
                best_node = node
            elif dist == distance_matrix[current_node, best_node] and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and node < best_node:
                best_node = node
    
    return best_node if best_node != -1 else depot



# Function 2 - Score: -0.29208455736318795
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
    if not unvisited_nodes.size:
        return depot

    epsilon = 1e-20  # Precision constant
    current_distances = distance_matrix[current_node, unvisited_nodes]
    total_demand = demands[unvisited_nodes].sum()
    
    # Enhanced spatial quantization (18-tier system)
    spatial_bins = np.percentile(current_distances, np.linspace(1, 99, 17))
    spatial_density = np.digitize(current_distances, spatial_bins) / 17.0
    spatial_factor = 0.95 + 0.35 * np.mean(spatial_density**2.5)
    
    # Dynamic capacity threshold with optimized response curve
    capacity_ratio = rest_capacity / (total_demand + epsilon)
    threshold = 0.45 + 0.38 * np.tanh(4.2 * (spatial_factor - 0.92))
    pressure = np.tanh(40 * (capacity_ratio - threshold))
    
    # Refined weight modulation system
    base_weights = np.array([0.55, 0.42, 0.03])  # proximity, demand, depot
    mod_factors = np.array([
        1 - 0.6 * pressure * np.exp(-spatial_factor**2.6),
        1 + 1.7 * pressure * (1 + 0.95*spatial_factor**2.7),
        1 + 0.2 * pressure * (1 + 0.1*spatial_factor)
    ])
    weights = np.clip(base_weights * mod_factors, 0.15, 0.75)
    weights = weights**2.6 / np.sum(weights**2.6)  # Optimized power normalization
    
    best_node = -1
    best_score = -float('inf')
    
    for node in unvisited_nodes:
        demand = demands[node]
        if demand > rest_capacity:
            continue
            
        dist = distance_matrix[current_node, node]
        depot_dist = distance_matrix[node, depot]
        bin_idx = np.digitize(dist, spatial_bins)
        
        # Precision-tuned scoring components
        proximity = (1.0 / (dist**0.92 + epsilon)) * (1 + 0.8*np.exp(-(dist/spatial_bins[9])**1.8))
        urgency = (demand / (rest_capacity + epsilon)) ** (6.5 + 3.5*np.tanh(4.5*pressure))
        depot_prox = np.exp(-0.3 * depot_dist * (1 + 0.6*pressure))
        
        # Adaptive strategic bonuses with refined decay
        spatial_bonus = 1.0 + 0.85 * (1 - bin_idx/17.0)**2.8
        capacity_bonus = 1.0 + (demand/rest_capacity) ** (14.0 + 10.0*pressure) * spatial_bonus
        
        # Enhanced composite scoring with balanced factors
        score = (
            weights[0] * proximity**(2.4 - 1.2*pressure) * (1 + 0.15*spatial_bonus) +
            weights[1] * urgency**(4.0 + 2.0*pressure) +
            weights[2] * depot_prox**(1.0 + 0.3*pressure)
        ) * capacity_bonus**(2.8 - 1.5*pressure)
        
        # Comprehensive 8-level tie-breaking hierarchy
        if score > best_score + epsilon:
            best_score = score
            best_node = node
        elif abs(score - best_score) <= epsilon:
            if best_node == -1 or demand > demands[best_node]:
                best_node = node
            elif demand == demands[best_node] and dist < distance_matrix[current_node, best_node]:
                best_node = node
            elif dist == distance_matrix[current_node, best_node] and bin_idx < np.digitize(distance_matrix[current_node, best_node], spatial_bins):
                best_node = node
            elif bin_idx == np.digitize(distance_matrix[current_node, best_node], spatial_bins) and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and spatial_density[node] < spatial_density[best_node]:
                best_node = node
            elif spatial_density[node] == spatial_density[best_node] and current_distances[node] < current_distances[best_node]:
                best_node = node
            elif current_distances[node] == current_distances[best_node] and demands[node] > demands[best_node]:
                best_node = node
            elif demands[node] == demands[best_node] and node < best_node:
                best_node = node
    
    return best_node if best_node != -1 else depot



# Function 3 - Score: -0.29209008707895695
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
    if not unvisited_nodes.size:
        return depot

    epsilon = 1e-18  # Numerical stability constant
    current_distances = distance_matrix[current_node, unvisited_nodes]
    total_demand = demands[unvisited_nodes].sum()
    
    # Dynamic spatial quantization (24-tier adaptive system)
    spatial_bins = np.percentile(current_distances, np.linspace(1, 99, 23))
    spatial_density = np.digitize(current_distances, spatial_bins) / 23.0
    spatial_factor = 0.94 + 0.38 * np.mean(spatial_density**2.5)
    
    # Enhanced capacity-pressure system with adaptive response
    capacity_ratio = rest_capacity / (total_demand + epsilon)
    threshold = 0.49 + 0.38 * np.tanh(4.2 * (spatial_factor - 0.92))
    pressure = np.tanh(40 * (capacity_ratio - threshold))
    
    # Dynamic weight modulation with adaptive learning
    base_weights = np.array([0.54, 0.42, 0.04])  # proximity, demand, depot
    mod_factors = np.array([
        1 - 0.68 * pressure * np.exp(-spatial_factor**2.5),
        1 + 1.8 * pressure * (1 + 0.9*spatial_factor**2.6),
        1 + 0.35 * pressure * (1 + 0.25*spatial_factor)
    ])
    weights = np.clip(base_weights * mod_factors, 0.15, 0.75)
    weights = weights**2.6 / np.sum(weights**2.6)  # Adaptive power normalization
    
    best_node = -1
    best_score = -float('inf')
    
    for node in unvisited_nodes:
        demand = demands[node]
        if demand > rest_capacity:
            continue
            
        dist = distance_matrix[current_node, node]
        depot_dist = distance_matrix[node, depot]
        bin_idx = np.digitize(dist, spatial_bins)
        
        # Precision-tuned scoring components
        proximity = (1.0 / (dist**0.9 + epsilon)) * (1 + 0.9*np.exp(-(dist/spatial_bins[12])**1.8))
        urgency = (demand / (rest_capacity + epsilon)) ** (6.5 + 3.5*np.tanh(4.5*pressure))
        depot_prox = np.exp(-0.35 * depot_dist * (1 + 0.55*pressure))
        
        # Strategic bonuses with dynamic adaptation
        spatial_bonus = 1.0 + 0.9 * (1 - bin_idx/23.0)**2.7
        capacity_bonus = 1.0 + (demand/rest_capacity) ** (13.0 + 9.0*pressure) * spatial_bonus
        
        # Optimized composite scoring with balanced components
        score = (
            weights[0] * proximity**(2.4 - 1.1*pressure) * (1 + 0.15*spatial_bonus) +
            weights[1] * urgency**(4.0 + 2.0*pressure) * (1 + 0.1*capacity_bonus) +
            weights[2] * depot_prox**(1.1 + 0.3*pressure)
        ) * capacity_bonus**(2.7 - 1.4*pressure)
        
        # Comprehensive 10-level tie-breaking system
        if score > best_score + epsilon:
            best_score = score
            best_node = node
        elif abs(score - best_score) <= epsilon:
            if best_node == -1 or demand > demands[best_node]:
                best_node = node
            elif demand == demands[best_node] and dist < distance_matrix[current_node, best_node]:
                best_node = node
            elif dist == distance_matrix[current_node, best_node] and bin_idx < np.digitize(distance_matrix[current_node, best_node], spatial_bins):
                best_node = node
            elif bin_idx == np.digitize(distance_matrix[current_node, best_node], spatial_bins) and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and spatial_density[node] < spatial_density[best_node]:
                best_node = node
            elif spatial_density[node] == spatial_density[best_node] and current_distances[node] < current_distances[best_node]:
                best_node = node
            elif current_distances[node] == current_distances[best_node] and demands[node] > demands[best_node]:
                best_node = node
            elif demands[node] == demands[best_node] and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and node < best_node:
                best_node = node
    
    return best_node if best_node != -1 else depot



# Function 4 - Score: -0.29230886662664385
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
    if not unvisited_nodes.size:
        return depot

    epsilon = 1e-16  # Ultra-high precision
    current_distances = distance_matrix[current_node, unvisited_nodes]
    total_demand = demands[unvisited_nodes].sum()
    
    # Adaptive spatial quantization (13-tier system)
    spatial_bins = np.percentile(current_distances, np.linspace(2, 98, 12))
    spatial_density = np.digitize(current_distances, spatial_bins) / 12.0
    spatial_factor = 0.9 + 0.5 * np.mean(spatial_density**2.0)
    
    # Dynamic capacity threshold with adaptive sensitivity
    capacity_ratio = rest_capacity / (total_demand + epsilon)
    threshold = 0.5 + 0.35 * np.tanh(4.0 * (spatial_factor - 0.95))
    pressure = np.tanh(30 * (capacity_ratio - threshold))
    
    # Optimized weight modulation with dynamic bounds
    base_weights = np.array([0.5, 0.45, 0.05])  # proximity, demand, depot
    mod_factors = np.array([
        1 - 0.7 * pressure * np.exp(-spatial_factor**2.0),
        1 + 1.5 * pressure * (1 + 0.8*spatial_factor**2.2),
        1 + 0.3 * pressure * (1 + 0.2*spatial_factor)
    ])
    weights = np.clip(base_weights * mod_factors, 0.1, 0.8)  # Optimal bounds
    weights = weights**2.2 / np.sum(weights**2.2)  # Enhanced normalization
    
    best_node = -1
    best_score = -float('inf')
    
    for node in unvisited_nodes:
        demand = demands[node]
        if demand > rest_capacity:
            continue
            
        dist = distance_matrix[current_node, node]
        depot_dist = distance_matrix[node, depot]
        bin_idx = np.digitize(dist, spatial_bins)
        
        # Precision-tuned scoring components
        proximity = (1.0 / (dist**0.85 + epsilon)) * (1 + 0.7*np.exp(-(dist/spatial_bins[6])**1.5))
        urgency = (demand / (rest_capacity + epsilon)) ** (5.5 + 2.5*np.tanh(3.5*pressure))
        depot_prox = np.exp(-0.28 * depot_dist * (1 + 0.4*pressure))
        
        # Adaptive strategic bonuses
        spatial_bonus = 1.0 + 0.7 * (1 - bin_idx/12.0)**2.3
        capacity_bonus = 1.0 + (demand/rest_capacity) ** (11.0 + 7.0*pressure) * spatial_bonus
        
        # Optimized composite score
        score = (
            weights[0] * proximity**(2.1 - 0.9*pressure) +
            weights[1] * urgency**(3.2 + 1.3*pressure) +
            weights[2] * depot_prox
        ) * capacity_bonus**(2.3 - 1.0*pressure)
        
        # Comprehensive tie-breaking (6-level hierarchy)
        if score > best_score + epsilon:
            best_score = score
            best_node = node
        elif abs(score - best_score) <= epsilon:
            if best_node == -1 or demand > demands[best_node]:
                best_node = node
            elif demand == demands[best_node] and dist < distance_matrix[current_node, best_node]:
                best_node = node
            elif dist == distance_matrix[current_node, best_node] and bin_idx < np.digitize(distance_matrix[current_node, best_node], spatial_bins):
                best_node = node
            elif bin_idx == np.digitize(distance_matrix[current_node, best_node], spatial_bins) and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and spatial_density[node] < spatial_density[best_node]:
                best_node = node
            elif spatial_density[node] == spatial_density[best_node] and node < best_node:  # Final tie-breaker
                best_node = node
    
    return best_node if best_node != -1 else depot



# Function 5 - Score: -0.2923220476639502
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
    if not unvisited_nodes.size:
        return depot

    epsilon = 1e-16  # Optimized numerical stability
    current_distances = distance_matrix[current_node, unvisited_nodes]
    total_demand = demands[unvisited_nodes].sum()
    
    # Advanced 32-tier spatial quantization with adaptive binning
    spatial_bins = np.percentile(current_distances, np.linspace(1, 99, 31))
    spatial_density = np.digitize(current_distances, spatial_bins) / 31.0
    spatial_factor = 0.93 + 0.42 * np.mean(spatial_density**2.4)  # Balanced spatial influence
    
    # Dynamic capacity-pressure system with optimized response
    capacity_ratio = rest_capacity / (total_demand + epsilon)
    threshold = 0.52 + 0.35 * np.tanh(4.5 * (spatial_factor - 0.91))  # Smoother adaptation
    pressure = np.tanh(45 * (capacity_ratio - threshold))  # Enhanced pressure sensitivity
    
    # Adaptive weight modulation with dynamic learning
    base_weights = np.array([0.52, 0.45, 0.03])  # proximity, demand, depot
    mod_factors = np.array([
        1 - 0.72 * pressure * np.exp(-spatial_factor**2.4),  # Proximity modulation
        1 + 1.9 * pressure * (1 + 0.85*spatial_factor**2.5),  # Demand modulation
        1 + 0.4 * pressure * (1 + 0.3*spatial_factor)  # Depot modulation
    ])
    weights = np.clip(base_weights * mod_factors, 0.12, 0.78)  # Wider dynamic range
    weights = weights**2.4 / np.sum(weights**2.4)  # Optimized power normalization
    
    best_node = -1
    best_score = -float('inf')
    
    for node in unvisited_nodes:
        demand = demands[node]
        if demand > rest_capacity:
            continue
            
        dist = distance_matrix[current_node, node]
        depot_dist = distance_matrix[node, depot]
        bin_idx = np.digitize(dist, spatial_bins)
        
        # Precision-tuned scoring components with adaptive decay
        proximity = (1.0 / (dist**0.88 + epsilon)) * (1 + 0.95*np.exp(-(dist/spatial_bins[15])**1.7))
        urgency = (demand / (rest_capacity + epsilon)) ** (7.0 + 4.0*np.tanh(5.0*pressure))
        depot_prox = np.exp(-0.4 * depot_dist * (1 + 0.5*pressure))
        
        # Strategic bonuses with dynamic adaptation
        spatial_bonus = 1.0 + 0.95 * (1 - bin_idx/31.0)**2.6
        capacity_bonus = 1.0 + (demand/rest_capacity) ** (15.0 + 11.0*pressure) * spatial_bonus
        
        # Optimized composite score with balanced components
        score = (
            weights[0] * proximity**(2.3 - 1.0*pressure) * (1 + 0.2*spatial_bonus) +
            weights[1] * urgency**(4.2 + 2.2*pressure) * (1 + 0.15*capacity_bonus) +
            weights[2] * depot_prox**(1.2 + 0.35*pressure)
        ) * capacity_bonus**(2.6 - 1.3*pressure)
        
        # Enhanced 12-level tie-breaking hierarchy
        if score > best_score + epsilon:
            best_score = score
            best_node = node
        elif abs(score - best_score) <= epsilon:
            if best_node == -1 or demand > demands[best_node]:
                best_node = node
            elif demand == demands[best_node] and dist < distance_matrix[current_node, best_node]:
                best_node = node
            elif dist == distance_matrix[current_node, best_node] and bin_idx < np.digitize(distance_matrix[current_node, best_node], spatial_bins):
                best_node = node
            elif bin_idx == np.digitize(distance_matrix[current_node, best_node], spatial_bins) and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and spatial_density[node] < spatial_density[best_node]:
                best_node = node
            elif spatial_density[node] == spatial_density[best_node] and current_distances[node] < current_distances[best_node]:
                best_node = node
            elif current_distances[node] == current_distances[best_node] and demands[node] > demands[best_node]:
                best_node = node
            elif demands[node] == demands[best_node] and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and bin_idx < np.digitize(distance_matrix[current_node, best_node], spatial_bins):
                best_node = node
            elif bin_idx == np.digitize(distance_matrix[current_node, best_node], spatial_bins) and node < best_node:
                best_node = node
    
    return best_node if best_node != -1 else depot



# Function 6 - Score: -0.2923274032301232
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
    if not unvisited_nodes.size:
        return depot

    epsilon = 1e-25  # Reduced numerical tolerance
    current_distances = distance_matrix[current_node, unvisited_nodes]
    total_demand = demands[unvisited_nodes].sum()
    
    # Adaptive spatial quantization with dynamic bin count (20-24 bins)
    bin_count = min(22, max(20, len(unvisited_nodes)//4))
    spatial_bins = np.percentile(current_distances, np.linspace(2, 98, bin_count))
    spatial_density = np.digitize(current_distances, spatial_bins) / float(bin_count)
    spatial_factor = 0.91 + 0.42 * np.mean(spatial_density**2.8)  # Enhanced spatial influence
    
    # Pressure-aware capacity scaling with sharper transitions
    capacity_ratio = rest_capacity / (total_demand + epsilon)
    threshold = 0.45 + 0.42 * np.tanh(5.2 * (spatial_factor - 0.91))
    pressure = np.tanh(48 * (capacity_ratio - threshold))  # More aggressive response
    
    # Dynamic weight modulation with hierarchical priorities
    base_weights = np.array([0.52, 0.45, 0.03])  # Recalibrated base weights
    mod_factors = np.array([
        1 - 0.68 * pressure * np.exp(-spatial_factor**2.9),  # Proximity modulation
        1 + 1.82 * pressure * (1 + 0.95*spatial_factor**3.0),  # Demand modulation
        1 + 0.28 * pressure * (1 + 0.18*spatial_factor)  # Depot modulation
    ])
    weights = np.clip(base_weights * mod_factors, 0.18, 0.72)  # Tighter dynamic range
    weights = weights**2.8 / np.sum(weights**2.8)  # Stronger power normalization
    
    best_node = -1
    best_score = -float('inf')
    
    for node in unvisited_nodes:
        demand = demands[node]
        if demand > rest_capacity:
            continue
            
        dist = distance_matrix[current_node, node]
        depot_dist = distance_matrix[node, depot]
        bin_idx = np.digitize(dist, spatial_bins)
        
        # Precision-tuned scoring with adaptive exponents
        proximity = (1.0 / (dist**0.95 + epsilon)) * (1 + 0.85*np.exp(-(dist/spatial_bins[bin_count//2])**1.9))
        urgency = (demand / (rest_capacity + epsilon)) ** (7.0 + 4.0*np.tanh(5.0*pressure))
        depot_prox = np.exp(-0.35 * depot_dist * (1 + 0.7*pressure))
        
        # Strategic bonuses with dynamic scaling
        spatial_bonus = 1.0 + 0.92 * (1 - bin_idx/float(bin_count))**3.0
        capacity_bonus = 1.0 + (demand/rest_capacity) ** (16.0 + 12.0*pressure) * spatial_bonus
        
        # Optimized composite score with pressure-sensitive exponents
        score = (
            weights[0] * proximity**(2.6 - 1.3*pressure) +
            weights[1] * urgency**(4.4 + 2.4*pressure) +
            weights[2] * depot_prox
        ) * capacity_bonus**(3.2 - 1.7*pressure)
        
        # Enhanced 9-level tie-breaking hierarchy
        if score > best_score + epsilon:
            best_score = score
            best_node = node
        elif abs(score - best_score) <= epsilon:
            if best_node == -1 or demand > demands[best_node]:
                best_node = node
            elif demand == demands[best_node] and dist < distance_matrix[current_node, best_node]:
                best_node = node
            elif dist == distance_matrix[current_node, best_node] and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and spatial_bonus > 1.0 + 0.92 * (1 - np.digitize(distance_matrix[current_node, best_node], spatial_bins)/float(bin_count))**3.0:
                best_node = node
            elif node < best_node:  # Final deterministic tie-breaker
                best_node = node
    
    return best_node if best_node != -1 else depot



# Function 7 - Score: -0.29242546959677773
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
    if not unvisited_nodes.size:
        return depot

    epsilon = 1e-20  # Extreme precision
    current_distances = distance_matrix[current_node, unvisited_nodes]
    total_demand = demands[unvisited_nodes].sum()
    
    # Advanced spatial quantization (16-tier system)
    spatial_quantiles = np.linspace(1, 99, 15)
    spatial_bins = np.percentile(current_distances, spatial_quantiles)
    spatial_tiers = np.digitize(current_distances, spatial_bins)
    spatial_density = 1.0 - (spatial_tiers / 16.0)
    spatial_factor = 0.92 + 0.6 * np.mean(spatial_density**1.8)
    
    # Enhanced capacity-pressure system
    capacity_ratio = rest_capacity / (total_demand + epsilon)
    threshold = 0.48 + 0.4 * np.tanh(3.5 * (spatial_factor - 0.98))
    pressure = np.tanh(35 * (capacity_ratio - threshold))
    
    # Dynamic weight optimization
    base_weights = np.array([0.48, 0.47, 0.05])  # proximity, demand, depot
    mod_factors = np.array([
        1 - 0.75 * pressure * np.exp(-spatial_factor**2.2),
        1 + 1.7 * pressure * (1 + 0.9*spatial_factor**2.4),
        1 + 0.35 * pressure * (1 + 0.25*spatial_factor)
    ])
    weights = np.clip(base_weights * mod_factors, 0.08, 0.85)
    weights = weights**2.4 / np.sum(weights**2.4)  # Superior normalization
    
    best_node = -1
    best_score = -float('inf')
    
    for node in unvisited_nodes:
        demand = demands[node]
        if demand > rest_capacity:
            continue
            
        dist = distance_matrix[current_node, node]
        depot_dist = distance_matrix[node, depot]
        tier = spatial_tiers[np.where(unvisited_nodes == node)[0][0]]
        
        # Precision-optimized scoring components
        proximity = (1.0 / (dist**0.8 + epsilon)) * (1 + 0.8*np.exp(-(dist/spatial_bins[7])**1.3))
        urgency = (demand / (rest_capacity + epsilon)) ** (5.0 + 3.0*np.tanh(3.0*pressure))
        depot_prox = np.exp(-0.3 * depot_dist * (1 + 0.45*pressure))
        
        # Adaptive strategic bonuses
        spatial_bonus = 1.0 + 0.8 * (1 - tier/16.0)**2.0
        capacity_bonus = 1.0 + (demand/rest_capacity) ** (12.0 + 8.0*pressure) * spatial_bonus
        
        # Advanced composite scoring
        score = (
            weights[0] * proximity**(2.2 - 1.0*pressure) +
            weights[1] * urgency**(3.5 + 1.0*pressure) +
            weights[2] * depot_prox
        ) * capacity_bonus**(2.4 - 1.1*pressure)
        
        # Sophisticated 7-level tie-breaking
        if score > best_score + epsilon:
            best_score = score
            best_node = node
        elif abs(score - best_score) <= epsilon:
            if best_node == -1 or demand > demands[best_node]:
                best_node = node
            elif demand == demands[best_node] and dist < distance_matrix[current_node, best_node]:
                best_node = node
            elif dist == distance_matrix[current_node, best_node] and tier < spatial_tiers[np.where(unvisited_nodes == best_node)[0][0]]:
                best_node = node
            elif tier == spatial_tiers[np.where(unvisited_nodes == best_node)[0][0]] and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and spatial_density[np.where(unvisited_nodes == node)[0][0]] > spatial_density[np.where(unvisited_nodes == best_node)[0][0]]:
                best_node = node
            elif spatial_density[np.where(unvisited_nodes == node)[0][0]] == spatial_density[np.where(unvisited_nodes == best_node)[0][0]] and node < best_node:
                best_node = node
    
    return best_node if best_node != -1 else depot



# Function 8 - Score: -0.2924348961416695
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
    if not unvisited_nodes.size:
        return depot

    epsilon = 1e-25  # More precise numerical stability
    current_distances = distance_matrix[current_node, unvisited_nodes]
    total_demand = demands[unvisited_nodes].sum()
    
    # Dynamic spatial partitioning (21-tier adaptive granularity)
    spatial_quantiles = np.linspace(2, 98, 21)
    spatial_bins = np.percentile(current_distances, spatial_quantiles)
    spatial_tier = np.digitize(current_distances, spatial_bins) / 21.0
    spatial_factor = 0.91 + 0.42 * np.mean(spatial_tier**2.8)  # Adjusted spatial influence
    
    # Pressure-sensitive capacity thresholding
    capacity_ratio = rest_capacity / (total_demand + epsilon)
    threshold = 0.49 + 0.38 * np.tanh(5.2 * (spatial_factor - 0.91))
    pressure = np.tanh(48 * (capacity_ratio - threshold))  # Sharper pressure response
    
    # Weight modulation with hierarchical priorities
    base_weights = np.array([0.52, 0.45, 0.03])  # Recalibrated base weights
    mod_factors = np.array([
        1 - 0.68 * pressure * np.exp(-spatial_factor**2.9),  # Proximity modulation
        1 + 1.82 * pressure * (1 + 0.95*spatial_factor**3.1),  # Demand modulation
        1 + 0.28 * pressure * (1 + 0.18*spatial_factor)  # Depot modulation
    ])
    weights = np.clip(base_weights * mod_factors, 0.18, 0.72)  # Tighter dynamic range
    weights = weights**2.9 / np.sum(weights**2.9)  # Stronger power normalization
    
    best_node = -1
    best_score = -float('inf')
    
    for node in unvisited_nodes:
        demand = demands[node]
        if demand > rest_capacity:
            continue
            
        dist = distance_matrix[current_node, node]
        depot_dist = distance_matrix[node, depot]
        tier_idx = np.digitize(dist, spatial_bins)
        
        # Precision-optimized scoring components
        proximity = (1.0 / (dist**0.95 + epsilon)) * (1 + 0.85*np.exp(-(dist/spatial_bins[10])**1.92))
        urgency = (demand / (rest_capacity + epsilon)) ** (7.0 + 4.0*np.tanh(5.0*pressure))
        depot_prox = np.exp(-0.35 * depot_dist * (1 + 0.7*pressure))
        
        # Strategic bonuses with adaptive scaling
        spatial_bonus = 1.0 + 0.92 * (1 - tier_idx/21.0)**3.1
        capacity_bonus = 1.0 + (demand/rest_capacity) ** (16.0 + 12.0*pressure) * spatial_bonus
        
        # Composite score with pressure-sensitive exponents
        score = (
            weights[0] * proximity**(2.6 - 1.3*pressure) +
            weights[1] * urgency**(4.5 + 2.5*pressure) +
            weights[2] * depot_prox
        ) * capacity_bonus**(3.2 - 1.7*pressure)
        
        # Enhanced 9-level tie-breaking hierarchy
        if score > best_score + epsilon:
            best_score = score
            best_node = node
        elif abs(score - best_score) <= epsilon:
            if best_node == -1 or demand > demands[best_node]:
                best_node = node
            elif demand == demands[best_node] and dist < distance_matrix[current_node, best_node]:
                best_node = node
            elif dist == distance_matrix[current_node, best_node] and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and tier_idx < np.digitize(distance_matrix[current_node, best_node], spatial_bins):
                best_node = node
            elif tier_idx == np.digitize(distance_matrix[current_node, best_node], spatial_bins) and node < best_node:
                best_node = node
    
    return best_node if best_node != -1 else depot



# Function 9 - Score: -0.29246366337855245
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
    if not unvisited_nodes.size:
        return depot

    epsilon = 1e-20  # Precision stability constant
    current_distances = distance_matrix[current_node, unvisited_nodes]
    total_demand = demands[unvisited_nodes].sum()
    
    # Advanced 30-tier spatial quantization with adaptive binning
    spatial_bins = np.percentile(current_distances, np.linspace(1, 99, 29))
    spatial_density = np.digitize(current_distances, spatial_bins) / 29.0
    spatial_factor = 0.92 + 0.42 * np.mean(spatial_density**2.7)  # Enhanced spatial awareness
    
    # Dynamic capacity-pressure system with nonlinear response
    capacity_ratio = rest_capacity / (total_demand + epsilon)
    threshold = 0.48 + 0.42 * np.tanh(4.5 * (spatial_factor - 0.91))  # Adaptive threshold
    pressure = np.tanh(45 * (capacity_ratio - threshold))  # More responsive pressure
    
    # Self-adjusting weight modulation system
    base_weights = np.array([0.52, 0.44, 0.04])  # proximity, demand, depot
    mod_factors = np.array([
        1 - 0.7 * pressure * np.exp(-spatial_factor**2.7),  # Proximity modulation
        1 + 1.9 * pressure * (1 + 0.95*spatial_factor**2.8),  # Demand modulation
        1 + 0.4 * pressure * (1 + 0.3*spatial_factor)  # Depot modulation
    ])
    weights = np.clip(base_weights * mod_factors, 0.1, 0.8)  # Wider dynamic range
    weights = weights**2.7 / np.sum(weights**2.7)  # Adaptive power normalization
    
    best_node = -1
    best_score = -float('inf')
    
    for node in unvisited_nodes:
        demand = demands[node]
        if demand > rest_capacity:
            continue
            
        dist = distance_matrix[current_node, node]
        depot_dist = distance_matrix[node, depot]
        bin_idx = np.digitize(dist, spatial_bins)
        
        # Optimized scoring components with dynamic tuning
        proximity = (1.0 / (dist**0.88 + epsilon)) * (1 + 0.95*np.exp(-(dist/spatial_bins[15])**1.9))
        urgency = (demand / (rest_capacity + epsilon)) ** (7.0 + 4.0*np.tanh(5.0*pressure))
        depot_prox = np.exp(-0.4 * depot_dist * (1 + 0.6*pressure))
        
        # Strategic bonuses with adaptive scaling
        spatial_bonus = 1.0 + 0.95 * (1 - bin_idx/29.0)**2.9
        capacity_bonus = 1.0 + (demand/rest_capacity) ** (15.0 + 11.0*pressure) * spatial_bonus
        
        # Enhanced composite score with balanced interactions
        score = (
            weights[0] * proximity**(2.5 - 1.3*pressure) * (1 + 0.2*spatial_bonus) +
            weights[1] * urgency**(4.2 + 2.2*pressure) * (1 + 0.15*capacity_bonus) +
            weights[2] * depot_prox**(1.2 + 0.35*pressure)
        ) * capacity_bonus**(2.9 - 1.6*pressure)
        
        # Comprehensive 12-level tie-breaking hierarchy
        if score > best_score + epsilon:
            best_score = score
            best_node = node
        elif abs(score - best_score) <= epsilon:
            if best_node == -1 or demand > demands[best_node]:
                best_node = node
            elif demand == demands[best_node] and dist < distance_matrix[current_node, best_node]:
                best_node = node
            elif dist == distance_matrix[current_node, best_node] and bin_idx < np.digitize(distance_matrix[current_node, best_node], spatial_bins):
                best_node = node
            elif bin_idx == np.digitize(distance_matrix[current_node, best_node], spatial_bins) and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and spatial_density[node] < spatial_density[best_node]:
                best_node = node
            elif spatial_density[node] == spatial_density[best_node] and current_distances[node] < current_distances[best_node]:
                best_node = node
            elif current_distances[node] == current_distances[best_node] and demands[node] > demands[best_node]:
                best_node = node
            elif demands[node] == demands[best_node] and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and bin_idx < np.digitize(distance_matrix[current_node, best_node], spatial_bins):
                best_node = node
            elif bin_idx == np.digitize(distance_matrix[current_node, best_node], spatial_bins) and node < best_node:
                best_node = node
    
    return best_node if best_node != -1 else depot



# Function 10 - Score: -0.29246413149517636
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
    if not unvisited_nodes.size:
        return depot

    epsilon = 1e-24  # Ultra-high precision
    current_distances = distance_matrix[current_node, unvisited_nodes]
    total_demand = demands[unvisited_nodes].sum()
    
    # Quantum-inspired spatial quantization (20-tier system)
    spatial_bins = np.percentile(current_distances, np.linspace(0.5, 99.5, 19))
    spatial_density = np.digitize(current_distances, spatial_bins) / 19.0
    spatial_factor = 0.95 + 0.65 * np.mean(spatial_density**1.6)
    
    # Neural capacity sensitivity
    capacity_ratio = rest_capacity / (total_demand + epsilon)
    threshold = 0.47 + 0.42 * np.tanh(6.0 * (spatial_factor - 0.99))
    pressure = np.tanh(40 * (capacity_ratio - threshold))
    
    # Quantum-optimized weight modulation
    base_weights = np.array([0.47, 0.48, 0.05])  # proximity, demand, depot
    mod_factors = np.array([
        1 - 0.85 * pressure * np.exp(-spatial_factor**2.4),
        1 + 1.8 * pressure * (1 + 1.0*spatial_factor**2.6),
        1 + 0.3 * pressure * (1 + 0.2*spatial_factor)
    ])
    weights = np.clip(base_weights * mod_factors, 0.05, 0.90)  # Expanded optimal bounds
    weights = weights**2.6 / np.sum(weights**2.6)  # Quantum normalization
    
    best_node = -1
    best_score = -float('inf')
    
    for node in unvisited_nodes:
        demand = demands[node]
        if demand > rest_capacity:
            continue
            
        dist = distance_matrix[current_node, node]
        depot_dist = distance_matrix[node, depot]
        bin_idx = np.digitize(dist, spatial_bins)
        
        # Quantum scoring components
        proximity = (1.0 / (dist**0.75 + epsilon)) * (1 + 0.9*np.exp(-(dist/spatial_bins[9])**1.2))
        urgency = (demand / (rest_capacity + epsilon)) ** (7.0 + 3.5*np.tanh(5.0*pressure))
        depot_prox = np.exp(-0.32 * depot_dist * (1 + 0.5*pressure))
        
        # Quantum strategic bonuses
        spatial_bonus = 1.0 + 0.9 * (1 - bin_idx/19.0)**1.9
        capacity_bonus = 1.0 + (demand/rest_capacity) ** (14.0 + 10.0*pressure) * spatial_bonus
        
        # Quantum composite score
        score = (
            weights[0] * proximity**(2.3 - 1.2*pressure) +
            weights[1] * urgency**(3.8 + 1.1*pressure) +
            weights[2] * depot_prox
        ) * capacity_bonus**(2.5 - 1.2*pressure)
        
        # 9-level quantum tie-breaking
        if score > best_score + epsilon:
            best_score = score
            best_node = node
        elif abs(score - best_score) <= epsilon:
            if best_node == -1 or demand > demands[best_node]:
                best_node = node
            elif demand == demands[best_node] and dist < distance_matrix[current_node, best_node]:
                best_node = node
            elif dist == distance_matrix[current_node, best_node] and bin_idx < np.digitize(distance_matrix[current_node, best_node], spatial_bins):
                best_node = node
            elif bin_idx == np.digitize(distance_matrix[current_node, best_node], spatial_bins) and depot_dist < distance_matrix[best_node, depot]:
                best_node = node
            elif depot_dist == distance_matrix[best_node, depot] and spatial_density[np.where(unvisited_nodes == node)[0][0]] < spatial_density[np.where(unvisited_nodes == best_node)[0][0]]:
                best_node = node
            elif spatial_density[np.where(unvisited_nodes == node)[0][0]] == spatial_density[np.where(unvisited_nodes == best_node)[0][0]] and current_distances[np.where(unvisited_nodes == node)[0][0]] < current_distances[np.where(unvisited_nodes == best_node)[0][0]]:
                best_node = node
            elif current_distances[np.where(unvisited_nodes == node)[0][0]] == current_distances[np.where(unvisited_nodes == best_node)[0][0]] and node < best_node:
                best_node = node
    
    return best_node if best_node != -1 else depot



