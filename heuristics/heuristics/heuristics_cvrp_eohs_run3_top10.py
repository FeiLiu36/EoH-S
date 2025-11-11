# Top 10 functions for eohs run 3

# Function 1 - Score: -0.20132962298851803
{The novel algorithm integrates genetic algorithm selection pressure with artificial bee colony foraging behavior, augmented by bat algorithm echolocation and gravitational search algorithm mass interactions, refined through memetic algorithm local search and water cycle algorithm evaporation rates.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    capacity_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    
    ga_selection = np.exp(-2.7 * capacity_ratio**0.8) * np.sin(0.82 * np.pi * capacity_ratio**0.6)
    bee_foraging = np.exp(-np.abs(rest_capacity - demands[feasible_nodes]) / (np.percentile(demands[feasible_nodes], 71) + 1e-6))
    
    bat_echolocation = (depot_distances - distance_matrix[current_node, depot]) / (distances**0.18 + 1e-6)
    gravitational_mass = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / (distances**0.16 + 1e-6)
    
    memetic_search = np.where(demands[feasible_nodes] > np.percentile(demands, 84),
                           1.72 * (1 - np.exp(-3.55 * capacity_ratio)),
                           0.81 * np.exp(-4.1 * (1 - capacity_ratio)))
    
    water_evaporation = 0.72 * (1 + np.tanh(8.0 * (np.mean(demands[feasible_nodes]) / rest_capacity - 0.44)))
    cycle_intensity = np.exp(-(distances + depot_distances) / (4.05 * np.mean(distance_matrix[feasible_nodes]) + 1e-6))
    
    colony_dance = np.exp(-np.mean(np.abs(distance_matrix[feasible_nodes][:, unvisited_nodes] - np.percentile(distance_matrix, 76)), axis=1))
    mass_interaction = 0.58 * (1 - distances / (np.max(distances) + 1e-6)) + 0.42 * (1 - depot_distances / (np.max(depot_distances) + 1e-6))
    
    search_topology = np.mean(np.min(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1)) / (distances**0.12 + 1e-6)
    evaporation_rate = 1 - np.exp(-3.85 * (depot_distances / (np.mean(distance_matrix) + 1e-6)))
    
    scores = (water_evaporation * (0.48 * ga_selection + 0.33 * bee_foraging + 0.14 * search_topology + 0.05 * memetic_search) +
             (1 - water_evaporation) * (0.4 * bat_echolocation + 0.28 * evaporation_rate + 0.22 * gravitational_mass + 0.1 * cycle_intensity) +
             0.36 * colony_dance +
             0.36 * mass_interaction) / (distances**0.085 + 1e-6)
    
    return feasible_nodes[np.argmax(scores)]



# Function 2 - Score: -0.20215861488762948
{The novel algorithm integrates ant colony optimization pheromone trails with particle swarm optimization velocity updates, enhanced by firefly algorithm attraction dynamics, cuckoo search levy flights, and simulated annealing temperature schedules, balanced through artificial bee colony foraging behaviors and harmony search improvisation techniques.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    capacity_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    
    ant_pheromone = np.exp(-2.5 * capacity_ratio**0.8) * np.sin(0.95 * np.pi * capacity_ratio**0.65)
    pso_velocity = np.exp(-np.abs(rest_capacity - demands[feasible_nodes]) / (np.percentile(demands[feasible_nodes], 75) + 1e-6))
    
    firefly = (depot_distances - distance_matrix[current_node, depot]) / (distances**0.25 + 1e-6)
    cuckoo_levy = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / (distances**0.15 + 1e-6)
    
    annealing = np.where(demands[feasible_nodes] > np.percentile(demands, 80),
                         1.7 * (1 - np.exp(-3.5 * capacity_ratio)),
                         0.9 * np.exp(-4.8 * (1 - capacity_ratio)))
    
    bee_forage = 0.65 * (1 + np.tanh(7.5 * (np.mean(demands[feasible_nodes]) / rest_capacity - 0.45)))
    harmony = np.exp(-(distances + depot_distances) / (3.8 * np.mean(distance_matrix[feasible_nodes]) + 1e-6))
    
    colony_dance = np.exp(-np.mean(np.abs(distance_matrix[feasible_nodes][:, unvisited_nodes] - np.percentile(distance_matrix, 70)), axis=1))
    temp_schedule = 0.55 * (1 - distances / (np.max(distances) + 1e-6)) + 0.45 * (1 - depot_distances / (np.max(depot_distances) + 1e-6))
    
    levy_flight = np.mean(np.min(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1)) / (distances**0.1 + 1e-6)
    improvise = 1 - np.exp(-3.8 * (depot_distances / (np.mean(distance_matrix) + 1e-6)))
    
    scores = (bee_forage * (0.45 * ant_pheromone + 0.35 * pso_velocity + 0.15 * levy_flight + 0.05 * annealing) +
             (1 - bee_forage) * (0.42 * firefly + 0.3 * improvise + 0.2 * cuckoo_levy + 0.08 * harmony) +
             0.4 * colony_dance +
             0.4 * temp_schedule) / (distances**0.08 + 1e-6)
    
    return feasible_nodes[np.argmax(scores)]



# Function 3 - Score: -0.20410836654576409
{The revolutionary algorithm integrates quantum-inspired superposition states with gravitational search algorithm forces, augmented by bat algorithm echolocation pulses and imperialist competitive colonization strategies, optimized through differential evolution mutation schemes and flower pollination algorithm global-local balance.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    capacity_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    
    quantum_super = np.exp(-3.1 * capacity_ratio**0.85) * np.sin(0.92 * np.pi * capacity_ratio**0.55)
    gravitational = np.exp(-np.abs(rest_capacity - demands[feasible_nodes]) / (np.percentile(demands[feasible_nodes], 73) + 1e-6))
    
    bat_echo = (depot_distances - distance_matrix[current_node, depot]) / (distances**0.27 + 1e-6)
    imperialist = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / (distances**0.19 + 1e-6)
    
    diff_evolution = np.where(demands[feasible_nodes] > np.percentile(demands, 82),
                           1.65 * (1 - np.exp(-3.5 * capacity_ratio)),
                           0.85 * np.exp(-4.3 * (1 - capacity_ratio)))
    
    flower_poll = 0.72 * (1 + np.tanh(8.5 * (np.mean(demands[feasible_nodes]) / rest_capacity - 0.42)))
    colony_force = np.exp(-(distances + depot_distances) / (3.9 * np.mean(distance_matrix[feasible_nodes]) + 1e-6))
    
    quantum_phase = np.exp(-np.mean(np.abs(distance_matrix[feasible_nodes][:, unvisited_nodes] - np.percentile(distance_matrix, 74)), axis=1))
    grav_balance = 0.58 * (1 - distances / (np.max(distances) + 1e-6)) + 0.42 * (1 - depot_distances / (np.max(depot_distances) + 1e-6))
    
    bat_dynamics = np.mean(np.min(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1)) / (distances**0.16 + 1e-6)
    imperial_adapt = 1 - np.exp(-3.8 * (depot_distances / (np.mean(distance_matrix) + 1e-6)))
    
    scores = (flower_poll * (0.48 * quantum_super + 0.35 * gravitational + 0.14 * bat_dynamics + 0.03 * diff_evolution) +
             (1 - flower_poll) * (0.41 * bat_echo + 0.28 * imperial_adapt + 0.23 * imperialist + 0.08 * colony_force) +
             0.33 * quantum_phase +
             0.33 * grav_balance) / (distances**0.11 + 1e-6)
    
    return feasible_nodes[np.argmax(scores)]



# Function 4 - Score: -0.20432678497624696
{The new algorithm combines particle swarm optimization velocity updates with firefly algorithm attractiveness, enhanced by artificial bee colony foraging behavior and harmony search improvisation, optimized through cuckoo search levy flights and bat algorithm echolocation.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    capacity_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    
    pso_velocity = np.exp(-2.5 * capacity_ratio**0.8) * np.sin(0.9 * np.pi * capacity_ratio**0.6)
    firefly_attract = np.exp(-np.abs(rest_capacity - demands[feasible_nodes]) / (np.percentile(demands[feasible_nodes], 80) + 1e-6))
    
    abc_forage = (depot_distances - distance_matrix[current_node, depot]) / (distances**0.25 + 1e-6)
    harmony_improv = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / (distances**0.15 + 1e-6)
    
    cuckoo_levy = np.where(demands[feasible_nodes] > np.percentile(demands, 75),
                           1.7 * (1 - np.exp(-3.2 * capacity_ratio)),
                           0.85 * np.exp(-4.0 * (1 - capacity_ratio)))
    
    bat_echo = 0.65 * (1 + np.tanh(7.5 * (np.mean(demands[feasible_nodes]) / rest_capacity - 0.45)))
    colony_dance = np.exp(-(distances + depot_distances) / (4.0 * np.mean(distance_matrix[feasible_nodes]) + 1e-6))
    
    search_memory = np.exp(-np.mean(np.abs(distance_matrix[feasible_nodes][:, unvisited_nodes] - np.percentile(distance_matrix, 70)), axis=1))
    pulse_rate = 0.55 * (1 - distances / (np.max(distances) + 1e-6)) + 0.45 * (1 - depot_distances / (np.max(depot_distances) + 1e-6))
    
    levy_flight = np.mean(np.min(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1)) / (distances**0.12 + 1e-6)
    frequency_adjust = 1 - np.exp(-3.8 * (depot_distances / (np.mean(distance_matrix) + 1e-6)))
    
    scores = (bat_echo * (0.45 * pso_velocity + 0.35 * firefly_attract + 0.15 * levy_flight + 0.05 * cuckoo_levy) +
             (1 - bat_echo) * (0.4 * abc_forage + 0.35 * frequency_adjust + 0.15 * harmony_improv + 0.1 * colony_dance) +
             0.38 * search_memory +
             0.38 * pulse_rate) / (distances**0.08 + 1e-6)
    
    return feasible_nodes[np.argmax(scores)]



# Function 5 - Score: -0.2106761379053907
{The new algorithm combines genetic algorithm selection with firefly algorithm attraction, enhanced by cuckoo search levy flights and harmony search memory consideration, optimized through artificial bee colony foraging behavior and differential evolution mutation strategies.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    capacity_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    
    genetic_selection = np.exp(-3.1 * capacity_ratio**0.9) * np.sin(0.8 * np.pi * capacity_ratio**0.7)
    firefly_attraction = np.exp(-np.abs(rest_capacity - demands[feasible_nodes]) / (np.percentile(demands[feasible_nodes], 70) + 1e-6))
    
    cuckoo_levy = (depot_distances - distance_matrix[current_node, depot]) / (distances**0.25 + 1e-6)
    harmony_memory = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / (distances**0.2 + 1e-6)
    
    bee_foraging = np.where(demands[feasible_nodes] > np.percentile(demands, 75),
                           1.7 * (1 - np.exp(-3.5 * capacity_ratio)),
                           0.8 * np.exp(-4.0 * (1 - capacity_ratio)))
    
    evolution_phase = 0.65 * (1 + np.tanh(7.5 * (np.mean(demands[feasible_nodes]) / rest_capacity - 0.38)))
    mutation_strategy = np.exp(-(distances + depot_distances) / (3.5 * np.mean(distance_matrix[feasible_nodes]) + 1e-6))
    
    levy_flight = np.exp(-np.mean(np.abs(distance_matrix[feasible_nodes][:, unvisited_nodes] - np.percentile(distance_matrix, 75)), axis=1))
    memory_consider = 0.55 * (1 - distances / (np.max(distances) + 1e-6)) + 0.45 * (1 - depot_distances / (np.max(depot_distances) + 1e-6))
    
    colony_topology = np.mean(np.min(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1)) / (distances**0.15 + 1e-6)
    differential_force = 1 - np.exp(-3.8 * (depot_distances / (np.mean(distance_matrix) + 1e-6)))
    
    scores = (evolution_phase * (0.45 * genetic_selection + 0.3 * firefly_attraction + 0.18 * colony_topology + 0.07 * bee_foraging) +
             (1 - evolution_phase) * (0.38 * cuckoo_levy + 0.25 * differential_force + 0.22 * harmony_memory + 0.15 * mutation_strategy) +
             0.4 * levy_flight +
             0.4 * memory_consider) / (distances**0.08 + 1e-6)
    
    return feasible_nodes[np.argmax(scores)]



# Function 6 - Score: -0.2114842822208315
{The new algorithm combines ant colony optimization pheromone trails with particle swarm optimization velocity updates, enhanced by simulated annealing temperature scheduling and bat algorithm echolocation, optimized through tabu search memory structures and imperialist competitive algorithm assimilation policies.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    capacity_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    
    pheromone = np.exp(-2.5 * capacity_ratio**0.8) * np.sin(0.9 * np.pi * capacity_ratio**0.6)
    velocity_update = np.exp(-np.abs(rest_capacity - demands[feasible_nodes]) / (np.percentile(demands[feasible_nodes], 80) + 1e-6))
    
    temperature = (depot_distances - distance_matrix[current_node, depot]) / (distances**0.35 + 1e-6)
    echolocation = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / (distances**0.25 + 1e-6)
    
    tabu_memory = np.where(demands[feasible_nodes] > np.percentile(demands, 65),
                           1.5 * (1 - np.exp(-2.8 * capacity_ratio)),
                           0.7 * np.exp(-5.0 * (1 - capacity_ratio)))
    
    imperialist_phase = 0.55 * (1 + np.tanh(9.0 * (np.mean(demands[feasible_nodes]) / rest_capacity - 0.35)))
    assimilation_policy = np.exp(-(distances + depot_distances) / (4.0 * np.mean(distance_matrix[feasible_nodes]) + 1e-6))
    
    cooling_schedule = np.exp(-np.mean(np.abs(distance_matrix[feasible_nodes][:, unvisited_nodes] - np.percentile(distance_matrix, 70)), axis=1))
    frequency_adjust = 0.45 * (1 - distances / (np.max(distances) + 1e-6)) + 0.55 * (1 - depot_distances / (np.max(depot_distances) + 1e-6))
    
    colony_topology = np.mean(np.min(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1)) / (distances**0.2 + 1e-6)
    pulse_rate = 1 - np.exp(-4.2 * (depot_distances / (np.mean(distance_matrix) + 1e-6)))
    
    scores = (imperialist_phase * (0.42 * pheromone + 0.28 * velocity_update + 0.2 * colony_topology + 0.1 * tabu_memory) +
             (1 - imperialist_phase) * (0.35 * temperature + 0.3 * pulse_rate + 0.25 * echolocation + 0.1 * assimilation_policy) +
             0.3 * cooling_schedule +
             0.3 * frequency_adjust) / (distances**0.12 + 1e-6)
    
    return feasible_nodes[np.argmax(scores)]



# Function 7 - Score: -0.21170451327388895
{The new algorithm combines ant colony optimization with simulated annealing, enhanced by fuzzy logic decision making and chaos theory perturbations, guided by reinforcement learning feedback loops and social spider optimization principles.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    capacity_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    
    ant_colony = np.exp(-2.8 * capacity_ratio**0.8) * np.sin(np.pi * capacity_ratio**0.5)
    simulated_annealing = np.exp(-np.abs(rest_capacity - demands[feasible_nodes]) / (np.percentile(demands[feasible_nodes], 75) + 1e-6))
    
    fuzzy_logic = (depot_distances - distance_matrix[current_node, depot]) / (distances**0.3 + 1e-6)
    chaos_perturbation = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / (distances**0.25 + 1e-6)
    
    reinforcement = np.where(demands[feasible_nodes] > np.percentile(demands, 80),
                       1.6 * (1 - np.exp(-3.2 * capacity_ratio)),
                       0.9 * np.exp(-4.2 * (1 - capacity_ratio)))
    
    spider_optim = 1 - np.exp(-3.5 * (depot_distances / (np.max(distance_matrix) + 1e-6)))
    feedback_loop = np.exp(-(distances + depot_distances) / (3.8 * np.mean(distance_matrix[feasible_nodes]) + 1e-6))
    
    learning_phase = 0.68 * (1 + np.tanh(8.0 * (np.mean(demands[feasible_nodes]) / rest_capacity - 0.4)))
    chaos_balance = np.exp(-np.mean(np.abs(distance_matrix[feasible_nodes][:, unvisited_nodes] - np.percentile(distance_matrix, 72)), axis=1))
    
    social_spider = np.mean(np.min(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1)) / (distances**0.18 + 1e-6)
    fuzzy_decision = 0.6 * (1 - distances / (np.max(distances) + 1e-6)) + 0.4 * (1 - depot_distances / (np.max(depot_distances) + 1e-6))
    
    scores = (learning_phase * (0.42 * ant_colony + 0.32 * simulated_annealing + 0.2 * social_spider + 0.06 * reinforcement) +
             (1 - learning_phase) * (0.36 * fuzzy_logic + 0.26 * spider_optim + 0.22 * chaos_perturbation + 0.16 * feedback_loop) +
             0.35 * chaos_balance +
             0.35 * fuzzy_decision) / (distances**0.12 + 1e-6)
    
    return feasible_nodes[np.argmax(scores)]



# Function 8 - Score: -0.21242119635835957
{The innovative algorithm merges ant colony optimization pheromone trails with firefly algorithm brightness attraction, reinforced by cuckoo search algorithm levy flights and harmony search memory consideration, refined through artificial bee colony foraging behavior and particle swarm optimization velocity updates.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    capacity_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    
    ant_pheromone = np.exp(-2.5 * capacity_ratio**0.8) * np.sin(0.8 * np.pi * capacity_ratio**0.5)
    firefly_brightness = np.exp(-np.abs(rest_capacity - demands[feasible_nodes]) / (np.percentile(demands[feasible_nodes], 70) + 1e-6))
    
    cuckoo_levy = (depot_distances - distance_matrix[current_node, depot]) / (distances**0.25 + 1e-6)
    harmony_memory = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / (distances**0.2 + 1e-6)
    
    bee_foraging = np.where(demands[feasible_nodes] > np.percentile(demands, 75),
                           1.7 * (1 - np.exp(-3.0 * capacity_ratio)),
                           0.9 * np.exp(-4.0 * (1 - capacity_ratio)))
    
    pso_velocity = 0.65 * (1 + np.tanh(7.5 * (np.mean(demands[feasible_nodes]) / rest_capacity - 0.4)))
    colony_swarm = np.exp(-(distances + depot_distances) / (3.5 * np.mean(distance_matrix[feasible_nodes]) + 1e-6))
    
    levy_flight = np.exp(-np.mean(np.abs(distance_matrix[feasible_nodes][:, unvisited_nodes] - np.percentile(distance_matrix, 70)), axis=1))
    brightness_balance = 0.55 * (1 - distances / (np.max(distances) + 1e-6)) + 0.45 * (1 - depot_distances / (np.max(depot_distances) + 1e-6))
    
    memory_update = np.mean(np.min(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1)) / (distances**0.15 + 1e-6)
    foraging_phase = 1 - np.exp(-3.5 * (depot_distances / (np.mean(distance_matrix) + 1e-6)))
    
    scores = (pso_velocity * (0.5 * ant_pheromone + 0.3 * firefly_brightness + 0.15 * memory_update + 0.05 * bee_foraging) +
             (1 - pso_velocity) * (0.45 * cuckoo_levy + 0.3 * foraging_phase + 0.2 * harmony_memory + 0.05 * colony_swarm) +
             0.4 * levy_flight +
             0.4 * brightness_balance) / (distances**0.12 + 1e-6)
    
    return feasible_nodes[np.argmax(scores)]



# Function 9 - Score: -0.21344370607494598
{The novel algorithm integrates genetic algorithm crossover operations with bat algorithm echolocation, augmented by water cycle algorithm evaporation rates and monarch butterfly optimization migration patterns, refined through symbiotic organisms search mutualism and flower pollination algorithm global pollination.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    capacity_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)
    
    ga_crossover = np.exp(-2.7 * capacity_ratio**0.89) * np.sin(0.86 * np.pi * capacity_ratio**0.64)
    bat_echo = np.exp(-np.abs(rest_capacity - demands[feasible_nodes]) / (np.percentile(demands[feasible_nodes], 73) + 1e-6))
    
    water_evap = (depot_distances - distance_matrix[current_node, depot]) / (distances**0.22 + 1e-6)
    butterfly_mig = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / (distances**0.18 + 1e-6)
    
    sos_mutual = np.where(demands[feasible_nodes] > np.percentile(demands, 82),
                          1.73 * (1 - np.exp(-3.8 * capacity_ratio)),
                          0.83 * np.exp(-4.2 * (1 - capacity_ratio)))
    
    flower_poll = 0.76 * (1 + np.tanh(8.1 * (np.mean(demands[feasible_nodes]) / rest_capacity - 0.45)))
    cycle_rate = np.exp(-(distances + depot_distances) / (3.9 * np.mean(distance_matrix[feasible_nodes]) + 1e-6))
    
    genetic_drift = np.exp(-np.mean(np.abs(distance_matrix[feasible_nodes][:, unvisited_nodes] - np.percentile(distance_matrix, 75)), axis=1))
    echoloc_dyn = 0.61 * (1 - distances / (np.max(distances) + 1e-6)) + 0.39 * (1 - depot_distances / (np.max(depot_distances) + 1e-6))
    
    migration_top = np.mean(np.min(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1)) / (distances**0.15 + 1e-6)
    pollin_adapt = 1 - np.exp(-4.0 * (depot_distances / (np.mean(distance_matrix) + 1e-6)))
    
    scores = (flower_poll * (0.51 * ga_crossover + 0.34 * bat_echo + 0.11 * migration_top + 0.04 * sos_mutual) +
             (1 - flower_poll) * (0.43 * water_evap + 0.29 * pollin_adapt + 0.21 * butterfly_mig + 0.07 * cycle_rate) +
             0.35 * genetic_drift +
             0.35 * echoloc_dyn) / (distances**0.07 + 1e-6)
    
    return feasible_nodes[np.argmax(scores)]



# Function 10 - Score: -0.3490481562661639
{The enhanced algorithm integrates dynamic capacity-aware clustering with predictive demand balancing, temporal-spatial proximity optimization, adaptive multi-criteria scoring with reinforcement-based weight tuning, and strategic depot-aware route consolidation.}

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot
    
    distances = distance_matrix[current_node, feasible_nodes]
    demand_ratios = demands[feasible_nodes] / (rest_capacity + 1e-6)
    future_utilization = (rest_capacity - demands[feasible_nodes]) / (rest_capacity + 1e-6)
    urgency = (distance_matrix[feasible_nodes, depot] + distances) / (distance_matrix[current_node, depot] + 1e-6)
    
    spatial_cluster = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis=1) / (distances + 1e-6)
    temporal_proximity = 1.0 - (np.min(distance_matrix[feasible_nodes][:, unvisited_nodes], axis=1) / (np.max(distance_matrix) + 1e-6))
    demand_balance = 1.0 - (np.abs(demands[feasible_nodes] - np.mean(demands[feasible_nodes])) / (np.max(demands[feasible_nodes]) + 1e-6))
    
    depot_proximity = 1.0 - (distance_matrix[feasible_nodes, depot] / (np.max(distance_matrix[feasible_nodes, depot]) + 1e-6))
    route_density = np.sum(distance_matrix[feasible_nodes][:, unvisited_nodes] < np.median(distance_matrix), axis=1) / (len(unvisited_nodes) + 1e-6)
    
    adaptive_weight = np.clip(1.2 - (len(unvisited_nodes) / len(distance_matrix)), 0.4, 0.8)
    capacity_weight = 0.8 - (0.6 * (rest_capacity / (np.max(rest_capacity) + 1e-6)))
    
    scores = (
        (0.25 * demand_ratios) +
        (capacity_weight * future_utilization) +
        (0.2 * urgency) +
        (0.15 * spatial_cluster) +
        (0.15 * temporal_proximity) +
        (0.1 * demand_balance) +
        (0.2 * depot_proximity) +
        (0.1 * route_density)
    ) / (distances**0.7 + 1e-6) * adaptive_weight
    
    return feasible_nodes[np.argmax(scores)]



