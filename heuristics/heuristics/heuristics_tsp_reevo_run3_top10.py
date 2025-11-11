# Top 10 functions for reevo run 3

# Function 1 - Score: -0.13961830412787868
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
    
    # Three distinct phases with custom transition logic
    completion = 1 - len(unvisited_nodes) / distance_matrix.shape[0]
    if completion < 0.4:  # Exploration phase
        progress = completion**0.5
        phase_coeff = 0.0
    elif completion < 0.8:  # Transition phase
        progress = 0.4 + 0.5*(completion - 0.4)/0.4
        phase_coeff = (completion - 0.4)/0.4
    else:  # Exploitation phase
        progress = 0.9 + 0.1*np.tanh(10*(completion - 0.9))
        phase_coeff = 1.0
    
    # Dynamic metric normalization using adaptive percentiles
    dist_to_nodes = distance_matrix[current_node, unvisited_nodes]
    p10, p50, p90 = np.percentile(dist_to_nodes, [10, 50, 90])
    
    # Phase-adaptive proximity with noise resilience
    proximity = 1 / (dist_to_nodes + 0.1*p10 + 0.05*p50)
    
    # Dual closure strategy (immediate + global)
    closure_diff = distance_matrix[unvisited_nodes, destination_node] - distance_matrix[current_node, destination_node]
    immediate_closure = np.tanh(2.5 * closure_diff/(p50 + 1e-6))
    global_closure = np.exp(-0.7 * distance_matrix[unvisited_nodes, destination_node]/p90)
    closure_boost = (immediate_closure * (1.3 - 0.6*phase_coeff) + 
                    global_closure * (0.3 + 0.7*phase_coeff))
    
    # Cluster density with adaptive scaling
    if len(unvisited_nodes) > 2:
        local_dists = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        density = np.sum(np.exp(-local_dists/(0.3*p50)), axis=1)
        density_factor = 0.4 + 1.6*(1 - density/np.max(density))**3  # Cubic penalty
    else:
        density_factor = 1.0
    
    # Phase-dependent exploration
    exploration = (0.9 * np.exp(-5*progress) * 
                  np.random.rand(len(unvisited_nodes))**(0.2 + 0.8*(1 - phase_coeff)))
    
    # Non-linear dynamic weighting
    proximity_weight = 1.0 - 0.5*phase_coeff**1.5
    closure_weight = 0.1 + 0.9*phase_coeff**2
    density_weight = 1.5 - phase_coeff**0.8
    
    # Combined scoring with phase-aware composition
    score = (proximity * proximity_weight * 
            (1 + closure_weight * closure_boost) * 
            density_factor**density_weight + 
            exploration)
    
    return unvisited_nodes[np.argmax(score)]



# Function 2 - Score: -0.1407640706217415
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
    
    # Three-phase progress tracking with smooth transitions
    completion = 1 - len(unvisited_nodes) / distance_matrix.shape[0]
    if completion < 0.3:
        progress = 0.5 * (1 - np.cos(3*np.pi*completion))  # Strong early exploration
    elif completion < 0.7:
        progress = 0.5 + 0.4*(completion - 0.3)/0.4  # Balanced phase
    else:
        progress = 0.9 + 0.1/(1 + np.exp(-20*(completion - 0.85)))  # Sharp final convergence
    
    # Adaptive proximity with dynamic normalization
    dist_to_nodes = distance_matrix[current_node, unvisited_nodes]
    p25, p50 = np.percentile(dist_to_nodes, [25, 50])
    proximity = 1 / (dist_to_nodes + 0.15*p25 + 0.05*p50)  # Simplified scaling
    
    # Smart closure strategy with progress adaptation
    closure_diff = distance_matrix[unvisited_nodes, destination_node] - distance_matrix[current_node, destination_node]
    closure_factor = (np.tanh(2.0 * closure_diff/(p50 + 1e-6)) * (1.2 - 0.5*progress) + 
                     np.exp(-0.5 * np.abs(closure_diff)/p50))
    
    # Cluster-aware density factor
    if len(unvisited_nodes) > 3:
        local_dists = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        density = np.sum(np.exp(-local_dists/(0.4*p50)), axis=1)
        density_factor = 0.5 + 1.5*(1 - density/np.max(density))**2  # Quadratic scaling
    else:
        density_factor = 1.0
    
    # Progress-driven exploration
    exploration = (0.8 * np.exp(-4*progress) * 
                  (np.random.rand(len(unvisited_nodes))**(0.3 + 0.7*progress) - 
                   0.4*(1 - progress)))
    
    # Dynamic multi-criteria weighting
    proximity_weight = 0.9 - 0.4*progress
    closure_weight = 0.2 + 0.6*progress**0.7
    density_weight = 1.2 - 0.7*progress**1.3
    
    score = (proximity * (proximity_weight + 0.1*density_weight) *
            (1 + closure_weight * closure_factor) *
            density_factor**density_weight + 
            exploration)
    
    return unvisited_nodes[np.argmax(score)]



# Function 3 - Score: -0.14120432034498376
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
    
    # Two-phase progress tracking with smooth transitions
    completion = 1 - len(unvisited_nodes) / distance_matrix.shape[0]
    if completion < 0.6:
        progress = 0.6 * (1 - np.cos(2.5*np.pi*completion))  # Exploration-dominant phase
    else:
        progress = 0.6 + 0.4 * (1 + np.tanh(15*(completion - 0.75)))  # Exploitation-focused phase
    
    # Adaptive proximity with single-percentile normalization
    dist_to_nodes = distance_matrix[current_node, unvisited_nodes]
    p50 = np.percentile(dist_to_nodes, 50)
    proximity = 1 / (dist_to_nodes + 0.2*p50)  # Simplified normalization
    
    # Progress-adaptive closure strategy
    closure_diff = distance_matrix[unvisited_nodes, destination_node] - distance_matrix[current_node, destination_node]
    closure_factor = np.tanh(1.8 * closure_diff/(p50 + 1e-6)) * (1.3 - 0.6*progress)  # Progress-scaled response
    
    # Lightweight density awareness
    if len(unvisited_nodes) > 2:
        local_dists = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        density = np.sum(np.exp(-local_dists/(0.5*p50)), axis=1)
        density_factor = 0.6 + 1.4*(1 - density/np.max(density))  # Linear scaling
    else:
        density_factor = 1.0
    
    # Balanced exploration with progress decay
    exploration = 0.7 * np.exp(-3.5*progress) * np.random.rand(len(unvisited_nodes))**(0.5 + 0.5*progress)
    
    # Dynamic weighting with smooth transitions
    proximity_weight = 0.85 - 0.35*progress**0.9
    closure_weight = 0.1 + 0.5*progress**1.2
    
    score = (proximity * proximity_weight * 
            (1 + closure_weight * closure_factor) * 
            density_factor + 
            exploration)
    
    return unvisited_nodes[np.argmax(score)]



# Function 4 - Score: -0.1415880925687996
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
    
    # Five-phase progress tracking with smooth transitions
    completion = 1 - len(unvisited_nodes)/distance_matrix.shape[0]
    if completion < 0.15:
        progress = 0.4 * (1 - np.cos(5*np.pi*completion))  # Initial exploration boost
    elif completion < 0.35:
        progress = 0.4 + 0.3*(completion - 0.15)/0.2  # Gradual transition
    elif completion < 0.65:
        progress = 0.7 + 0.2*(completion - 0.35)/0.3  # Balanced phase
    elif completion < 0.85:
        progress = 0.9 + 0.08*(completion - 0.65)/0.2  # Focused optimization
    else:
        progress = 0.98 + 0.02/(1 + np.exp(-40*(completion - 0.92)))  # Final precision
    
    # Adaptive multi-quantile proximity scaling
    dist_to_nodes = distance_matrix[current_node, unvisited_nodes]
    q1, med, q3 = np.percentile(dist_to_nodes, [15, 50, 85])
    proximity = 1 / (dist_to_nodes + 0.1*q1 + 0.15*med + 0.05*q3)  # Triple-quantile normalization
    
    # Triple-component closure strategy
    closure_diff = distance_matrix[unvisited_nodes, destination_node] - distance_matrix[current_node, destination_node]
    relative_closure = closure_diff / (0.2*q1 + 0.3*med + 0.5*q3 + 1e-6)
    directional = np.tanh(3.0 * relative_closure) * (1.4 - 0.7*progress)
    absolute = np.exp(-0.7 * distance_matrix[unvisited_nodes, destination_node]/q3)
    intermediate = np.exp(-0.3 * np.abs(closure_diff)/med)
    closure_factor = 0.6*directional + 0.2*absolute + 0.2*intermediate
    
    # Hierarchical density awareness
    if len(unvisited_nodes) > 3:
        local_dists = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Three-level density analysis
        immediate = np.sum(np.exp(-local_dists/(0.25*med)), axis=1)
        neighborhood = np.sum(np.exp(-local_dists/(0.75*med)), axis=1)
        regional = np.sum(np.exp(-local_dists/(2.0*med)), axis=1)
        density_factor = 0.5 + 1.5*(1 - (0.5*immediate + 0.3*neighborhood + 0.2*regional) / 
                                   np.max(0.5*immediate + 0.3*neighborhood + 0.2*regional))
    else:
        density_factor = 1.0
    
    # Adaptive exploration with phase-dependent parameters
    exploration_magnitude = 0.8 * np.exp(-6*progress)
    exploration_shape = 0.15 + 0.85*progress  # Shifts from uniform to peaky
    exploration_bias = 0.4 - 0.35*progress  # Shifts from positive to neutral
    exploration = exploration_magnitude * (
        np.random.rand(len(unvisited_nodes))**exploration_shape - exploration_bias)
    
    # Dynamic multi-objective scoring
    proximity_weight = 0.9 - 0.5*progress**1.5  # Non-linear decay
    closure_weight = 0.25 + 0.6*progress**0.8  # Sub-linear growth
    density_weight = 1.3 - 0.8*progress**1.2  # Curved adjustment
    
    score = (
        proximity_weight * proximity * 
        (1 + closure_weight * closure_factor) * 
        (density_factor**density_weight) + 
        (1 - proximity_weight) * exploration
    )
    
    return unvisited_nodes[np.argmax(score)]



# Function 5 - Score: -0.14198613705472785
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
    
    # Six-phase progress tracking with enhanced transitions
    completion = 1 - len(unvisited_nodes)/distance_matrix.shape[0]
    if completion < 0.1:
        progress = 0.3 * (1 - np.cos(6*np.pi*completion))  # Strong initial exploration
    elif completion < 0.25:
        progress = 0.3 + 0.25*(completion - 0.1)/0.15  # Smooth ramp-up
    elif completion < 0.45:
        progress = 0.55 + 0.25*(completion - 0.25)/0.2  # Balanced growth
    elif completion < 0.7:
        progress = 0.8 + 0.15*(completion - 0.45)/0.25  # Focused optimization
    elif completion < 0.9:
        progress = 0.95 + 0.04*(completion - 0.7)/0.2  # Precision tuning
    else:
        progress = 0.99 + 0.01/(1 + np.exp(-50*(completion - 0.93)))  # Final refinement
    
    # Enhanced proximity with quadruple-quantile scaling
    dist_to_nodes = distance_matrix[current_node, unvisited_nodes]
    q1, q2, med, q3, q4 = np.percentile(dist_to_nodes, [10, 25, 50, 75, 90])
    proximity = 1 / (dist_to_nodes + 0.05*q1 + 0.1*q2 + 0.1*med + 0.05*q3 + 0.02*q4)
    
    # Quad-component closure strategy
    closure_diff = distance_matrix[unvisited_nodes, destination_node] - distance_matrix[current_node, destination_node]
    relative_closure = closure_diff / (0.15*q1 + 0.2*q2 + 0.3*med + 0.35*q3 + 1e-6)
    directional = np.tanh(3.5 * relative_closure) * (1.5 - 0.8*progress)
    absolute = np.exp(-0.8 * distance_matrix[unvisited_nodes, destination_node]/q3)
    intermediate = np.exp(-0.4 * np.abs(closure_diff)/med)
    global_adjust = 1.2 - 0.4*np.tanh(2.5*distance_matrix[unvisited_nodes, destination_node]/q4)
    closure_factor = 0.5*directional + 0.2*absolute + 0.15*intermediate + 0.15*global_adjust
    
    # Multi-scale density analysis
    if len(unvisited_nodes) > 2:
        local_dists = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Four-level density assessment
        core = np.sum(np.exp(-local_dists/(0.2*med)), axis=1)
        local = np.sum(np.exp(-local_dists/(0.5*med)), axis=1)
        regional = np.sum(np.exp(-local_dists/(1.2*med)), axis=1)
        global_scale = np.sum(np.exp(-local_dists/(3.0*med)), axis=1)
        density_factor = 0.4 + 1.6*(1 - (0.4*core + 0.3*local + 0.2*regional + 0.1*global_scale) / 
                                   np.max(0.4*core + 0.3*local + 0.2*regional + 0.1*global_scale))
    else:
        density_factor = 1.0
    
    # Smart exploration with dual adaptation
    exploration_magnitude = 0.7 * np.exp(-7*progress**1.2)
    exploration_shape = 0.1 + 0.9*progress**0.7  # Dynamic shape control
    exploration_bias = 0.5 - 0.45*progress**0.9  # Progress-dependent bias
    exploration = exploration_magnitude * (
        np.random.rand(len(unvisited_nodes))**exploration_shape - exploration_bias)
    
    # Advanced dynamic weighting
    proximity_weight = 0.95 - 0.6*progress**1.8  # Strong initial preference
    closure_weight = 0.2 + 0.7*progress**0.7  # Gradual importance increase
    density_weight = 1.4 - 0.9*progress**1.4  # Curved adjustment
    
    # Final scoring with non-linear combination
    score = (
        proximity_weight * proximity * 
        (1 + closure_weight * closure_factor) * 
        (density_factor**density_weight) + 
        (1 - proximity_weight) * exploration
    )
    
    return unvisited_nodes[np.argmax(score)]



# Function 6 - Score: -0.1421659703338611
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
    
    # Four-phase adaptive progress tracking
    completion = 1 - len(unvisited_nodes)/distance_matrix.shape[0]
    if completion < 0.2:
        progress = 0.5 * (1 - np.cos(4*np.pi*completion))  # Aggressive early exploration
    elif completion < 0.5:
        progress = 0.2 + 0.6*(completion - 0.2)/0.3  # Balanced middle phase
    elif completion < 0.8:
        progress = 0.8 + 0.15*(completion - 0.5)/0.3  # Focused convergence
    else:
        progress = 1 / (1 + np.exp(-25*(completion - 0.9)))  # Sharp final optimization
    
    # Dynamic multi-scale proximity
    dist_to_nodes = distance_matrix[current_node, unvisited_nodes]
    q1, q3 = np.percentile(dist_to_nodes, [20, 80])
    proximity = 1 / (dist_to_nodes + 0.1*q1 + 0.2*q3)  # Multi-quantile normalization
    
    # Adaptive closure strategy with dual components
    closure_diff = distance_matrix[unvisited_nodes, destination_node] - distance_matrix[current_node, destination_node]
    relative_closure = closure_diff / (0.3*q1 + 0.7*q3 + 1e-6)
    directional_boost = np.tanh(2.5 * relative_closure) * (1.3 - 0.6*progress)
    absolute_boost = np.exp(-0.5 * distance_matrix[unvisited_nodes, destination_node]/q3)
    closure_factor = 0.7*directional_boost + 0.3*absolute_boost
    
    # Multi-resolution density awareness
    if len(unvisited_nodes) > 4:
        local_dists = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        short_range = np.sum(np.exp(-local_dists/(0.3*np.median(local_dists))), axis=1)
        long_range = np.sum(np.exp(-local_dists/(1.5*np.median(local_dists))), axis=1)
        density_factor = 0.4 + 1.6*(1 - (0.6*short_range + 0.4*long_range)/np.max(0.6*short_range + 0.4*long_range))
    else:
        density_factor = 1.0
    
    # Smart exploration with phase-dependent characteristics
    exploration_magnitude = 0.9 * np.exp(-5*progress)
    exploration_shape = 0.2 + 0.8*progress  # Shifts exploration distribution
    exploration = exploration_magnitude * (np.random.rand(len(unvisited_nodes))**exploration_shape - 0.5)
    
    # Phase-aware multi-objective scoring
    proximity_weight = 0.85 - 0.4*progress
    closure_weight = 0.3 + 0.5*progress
    density_weight = 1.2 - 0.7*progress
    
    score = (
        proximity_weight * proximity * 
        (1 + closure_weight * closure_factor) * 
        (density_factor**density_weight) + 
        (1 - proximity_weight) * exploration
    )
    
    return unvisited_nodes[np.argmax(score)]



# Function 7 - Score: -0.14293566708642974
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
    
    # Seven-phase progress tracking with exponential transitions
    completion = 1 - len(unvisited_nodes)/distance_matrix.shape[0]
    if completion < 0.1:
        progress = 0.3 * (1 - np.cos(7*np.pi*completion))  # Initial random exploration
    elif completion < 0.25:
        progress = 0.3 + 0.25*(1 - np.exp(-10*(completion - 0.1)))  # Early exploration
    elif completion < 0.45:
        progress = 0.55 + 0.2*(completion - 0.25)/0.2  # Transition to balanced
    elif completion < 0.65:
        progress = 0.75 + 0.15*(1 - np.cos(np.pi*(completion - 0.45)/0.2))  # Balanced optimization
    elif completion < 0.8:
        progress = 0.9 + 0.07*(completion - 0.65)/0.15  # Focused exploitation
    elif completion < 0.95:
        progress = 0.97 + 0.02*np.tanh(20*(completion - 0.875))  # Precision tuning
    else:
        progress = 0.99 + 0.01/(1 + np.exp(-50*(completion - 0.96)))  # Final convergence
    
    # Multi-quantile distance normalization (5-point)
    dist_to_nodes = distance_matrix[current_node, unvisited_nodes]
    q = np.percentile(dist_to_nodes, [5, 25, 50, 75, 95])
    proximity = 1 / (dist_to_nodes + 0.05*q[0] + 0.1*q[1] + 0.1*q[2] + 0.05*q[3] + 0.02*q[4])
    
    # Quad-component closure strategy
    closure_diff = distance_matrix[unvisited_nodes, destination_node] - distance_matrix[current_node, destination_node]
    relative_closure = closure_diff / (0.1*q[0] + 0.2*q[1] + 0.3*q[2] + 0.3*q[3] + 0.1*q[4] + 1e-6)
    directional = np.tanh(3.5 * relative_closure) * (1.5 - 0.8*progress)
    absolute = np.exp(-0.8 * distance_matrix[unvisited_nodes, destination_node]/q[3])
    intermediate = np.exp(-0.4 * np.abs(closure_diff)/q[2])
    momentum = np.exp(-0.2 * np.abs(closure_diff - np.mean(closure_diff))/q[2])
    closure_factor = 0.5*directional + 0.2*absolute + 0.15*intermediate + 0.15*momentum
    
    # Multi-scale density analysis (4 levels)
    if len(unvisited_nodes) > 4:
        local_dists = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        density_weights = [0.4, 0.3, 0.2, 0.1]
        density_scales = [0.2*q[2], 0.5*q[2], 1.2*q[2], 2.5*q[2]]
        density_components = [
            np.sum(np.exp(-local_dists/scale), axis=1)
            for scale in density_scales
        ]
        combined_density = sum(w*c for w, c in zip(density_weights, density_components))
        density_factor = 0.4 + 1.6*(1 - combined_density/np.max(combined_density))
    else:
        density_factor = 1.0
    
    # Phase-aware exploration
    exploration_magnitude = 0.9 * np.exp(-8*progress**1.2)
    exploration_shape = 0.1 + 0.9*progress**1.5
    exploration_bias = 0.5 - 0.45*progress**0.7
    exploration = exploration_magnitude * (
        np.random.rand(len(unvisited_nodes))**exploration_shape - exploration_bias)
    
    # Dynamic objective balancing with non-linear mixing
    proximity_weight = 0.95 - 0.6*progress**1.8
    closure_weight = 0.2 + 0.7*progress**0.9
    density_weight = 1.4 - 0.9*progress**1.4
    
    score = (
        proximity_weight * proximity *
        (1 + closure_weight * closure_factor) *
        (density_factor**density_weight) +
        (1 - proximity_weight**0.7) * exploration
    )
    
    return unvisited_nodes[np.argmax(score)]



# Function 8 - Score: -0.14403781716289427
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
    
    # Adaptive phase detection with dynamic thresholds
    remaining_ratio = len(unvisited_nodes) / distance_matrix.shape[0]
    if remaining_ratio > 0.6:  # Early exploration
        progress = 0.3 * (1 - np.cos(np.pi * (1 - remaining_ratio)/0.4))
    elif remaining_ratio > 0.2:  # Balanced phase
        progress = 0.3 + 0.5 * ((0.6 - remaining_ratio)/0.4)**0.7
    else:  # Final exploitation
        progress = 0.8 + 0.2 * np.tanh(8 * (0.2 - remaining_ratio))
    
    # Smart proximity with adaptive scaling
    dist_to_nodes = distance_matrix[current_node, unvisited_nodes]
    p25, p75 = np.percentile(dist_to_nodes, [25, 75])
    proximity = 1 / (dist_to_nodes + 0.15*p25 + 0.1*p75)
    
    # Enhanced closure strategy
    closure_diff = distance_matrix[unvisited_nodes, destination_node] - distance_matrix[current_node, destination_node]
    closure_factor = (
        np.tanh(2.0 * closure_diff / (p75 + 1e-6)) * (1.2 - 0.4*progress) +
        np.exp(-0.8 * np.abs(closure_diff) / (p75 + 1e-6))
    )
    
    # Dynamic cluster awareness
    if len(unvisited_nodes) > 3:
        local_dists = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        density = np.sum(np.exp(-local_dists / (0.4*p75)), axis=1)
        density_factor = 0.7 + 1.3 * (1 - density/np.max(density))**1.3
    else:
        density_factor = 1.0
    
    # Controlled stochastic exploration
    exploration = (
        0.6 * np.exp(-4*progress) *
        (np.random.rand(len(unvisited_nodes))**(0.3 + 0.7*progress) -
         0.4*(1 - progress)**1.5)
    )
    
    # Optimized dynamic weights
    proximity_weight = 1.0 - 0.6*progress**1.3
    closure_weight = 0.2 + 0.6*progress**1.6
    density_weight = 1.2 - 0.7*progress**1.1
    
    # Final scoring with balanced factors
    score = (
        proximity**proximity_weight *
        (1 + closure_weight * closure_factor) *
        density_factor**density_weight +
        exploration
    )
    
    return unvisited_nodes[np.argmax(score)]



# Function 9 - Score: -0.1442995423547878
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
    
    # Three-phase progress tracking
    completion = 1 - len(unvisited_nodes)/distance_matrix.shape[0]
    if completion < 0.3:
        progress = 0.6 * (1 - np.cos(3*np.pi*completion))  # Exploration phase
    elif completion < 0.7:
        progress = 0.3 + 0.5*(completion - 0.3)/0.4  # Balanced phase
    else:
        progress = 0.8 + 0.2*(1 - np.exp(-10*(completion - 0.7)))  # Exploitation phase
    
    # Dual-quantile proximity scaling
    dist_to_nodes = distance_matrix[current_node, unvisited_nodes]
    q1, q3 = np.percentile(dist_to_nodes, [25, 75])
    proximity = 1 / (dist_to_nodes + 0.15*q1 + 0.1*q3)
    
    # Direction-focused closure strategy
    closure_diff = distance_matrix[unvisited_nodes, destination_node] - distance_matrix[current_node, destination_node]
    relative_closure = closure_diff / (0.4*q1 + 0.6*q3 + 1e-6)
    closure_factor = np.tanh(2.0 * relative_closure) * (1.2 - 0.5*progress)
    
    # Dual-scale density awareness
    if len(unvisited_nodes) > 3:
        local_dists = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        density = (
            0.7 * np.sum(np.exp(-local_dists/(0.4*np.median(local_dists))), axis=1) +
            0.3 * np.sum(np.exp(-local_dists/(1.2*np.median(local_dists))), axis=1)
        )
        density_factor = 0.5 + 1.5*(1 - density/np.max(density))
    else:
        density_factor = 1.0
    
    # Progress-adaptive exploration
    exploration = (0.8 - 0.7*progress) * (np.random.rand(len(unvisited_nodes))**(0.3 + 0.7*progress) - 0.3)
    
    # Dynamic objective balancing
    proximity_weight = 0.9 - 0.5*progress
    closure_weight = 0.2 + 0.6*progress
    
    score = (
        proximity_weight * proximity * 
        (1 + closure_weight * closure_factor) * 
        density_factor + 
        (1 - proximity_weight) * exploration
    )
    
    return unvisited_nodes[np.argmax(score)]



# Function 10 - Score: -0.14482071424205892
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
    
    # Phase detection with cubic interpolation for smooth transitions
    completion = 1 - len(unvisited_nodes) / distance_matrix.shape[0]
    if completion < 0.4:  # Exploration phase
        progress = 0.5 * (1 - np.cos(np.pi * completion / 0.4))
    elif completion < 0.8:  # Balanced phase
        progress = 0.5 + 0.4 * ((completion - 0.4) / 0.4)**0.8
    else:  # Exploitation phase
        progress = 0.9 + 0.1 * np.tanh(10 * (completion - 0.9))
    
    # Dynamic proximity with adaptive percentile scaling
    dist_to_nodes = distance_matrix[current_node, unvisited_nodes]
    p10, p50, p90 = np.percentile(dist_to_nodes, [10, 50, 90])
    proximity = 1 / (dist_to_nodes + 0.1*p10 + 0.05*p50)
    
    # Dual-mode closure strategy
    closure_diff = distance_matrix[unvisited_nodes, destination_node] - distance_matrix[current_node, destination_node]
    closure_factor = (
        np.tanh(1.8 * closure_diff / (p50 + 1e-6)) * (1.4 - 0.6*progress) + 
        np.exp(-0.7 * np.abs(closure_diff) / (p90 + 1e-6))
    )
    
    # Adaptive cluster awareness
    if len(unvisited_nodes) > 4:
        local_dists = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        density = np.sum(np.exp(-local_dists / (0.35*p50)), axis=1)
        density_factor = 0.6 + 1.4 * (1 - density/np.max(density))**1.5
    else:
        density_factor = 1.0
    
    # Phase-optimized exploration
    exploration = (
        0.7 * np.exp(-5*progress) * 
        (np.random.rand(len(unvisited_nodes))**(0.2 + 0.8*progress) - 
         0.5*(1 - progress)**2)
    )
    
    # Non-linear dynamic weighting
    proximity_weight = 0.95 - 0.5*progress**1.2
    closure_weight = 0.15 + 0.7*progress**1.5
    density_weight = 1.3 - 0.8*progress**0.9
    
    # Final scoring with exponential blending
    score = (
        proximity**(proximity_weight + 0.05*density_weight) *
        (1 + closure_weight * closure_factor) *
        density_factor**density_weight + 
        exploration
    )
    
    return unvisited_nodes[np.argmax(score)]



