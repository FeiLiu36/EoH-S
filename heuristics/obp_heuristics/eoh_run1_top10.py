# Top 10 functions for eoh run 1

# Function 1 - Score: -0.0397415458022838
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    ideal = 0.3
    deviation = np.abs(remaining_capacity - ideal)
    max_deviation = 1.0  # Adjust as needed
    priority_scores = np.where(
        valid_mask,
        np.maximum(0, 1 - deviation / max_deviation),
        0
    )
    return priority_scores



# Function 2 - Score: -0.0397415458022838
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    ideal = 0.3
    deviation = np.abs(remaining_capacity - ideal)
    priority_scores = np.where(
        valid_mask,
        np.maximum(1 - deviation, 0),
        0
    )
    return priority_scores



# Function 3 - Score: -0.0397415458022838
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    ideal = 0.4
    deviation = np.abs(remaining_capacity - ideal)
    beta = 1.5  # Slope parameter
    priority_scores = np.where(
        valid_mask,
        np.maximum(1 - beta * deviation, 0),
        0
    )
    return priority_scores



# Function 4 - Score: -0.0397415458022838
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    ideal = 0.4
    deviation = np.abs(remaining_capacity - ideal)
    beta = 1.5  # Slope parameter
    priority_scores = np.where(
        valid_mask,
        np.maximum(1 - beta * deviation, 0),
        0
    )
    return priority_scores



# Function 5 - Score: -0.0397415458022838
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    ideal = 0.3
    deviation = np.abs(remaining_capacity - ideal)
    max_deviation = 1.0  # Adjust as needed
    priority_scores = np.where(
        valid_mask,
        np.maximum(0, 1 - deviation / max_deviation),
        -np.inf
    )
    return priority_scores



# Function 6 - Score: -0.0397415458022838
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    ideal = 0.4
    deviation = np.abs(remaining_capacity - ideal)
    beta = 1.5  # Slope parameter
    priority_scores = np.where(
        valid_mask,
        np.maximum(1 - beta * deviation, 0),
        0
    )
    return priority_scores



# Function 7 - Score: -0.0397415458022838
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    ideal = 0.3
    distance = np.abs(remaining_capacity - ideal)
    max_distance = 1.0  # Maximum distance for full penalty
    priority_scores = np.where(
        valid_mask,
        np.maximum(0, 1 - (distance / max_distance)),
        0
    )
    return priority_scores



# Function 8 - Score: -0.0397415458022838
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    ideal = 0.3
    deviation = np.abs(remaining_capacity - ideal)
    max_deviation = 1.0  # Adjust as needed
    priority_scores = np.where(
        valid_mask,
        np.maximum(0, 1 - deviation / max_deviation),
        0
    )
    return priority_scores



# Function 9 - Score: -0.0397415458022838
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    ideal = 0.3
    deviation = np.abs(remaining_capacity - ideal)
    max_deviation = 1.0  # Maximum allowed deviation
    priority_scores = np.where(
        valid_mask,
        np.maximum(0, 1 - deviation / max_deviation),
        0
    )
    return priority_scores



# Function 10 - Score: -0.0397415458022838
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining_capacity = bins - item
    valid_mask = remaining_capacity >= 0
    ideal = 0.3
    distance = np.abs(remaining_capacity - ideal)
    max_distance = 1.0  # Adjust as needed
    priority_scores = np.where(
        valid_mask,
        np.maximum(0, 1 - distance / max_distance),
        0
    )
    return priority_scores



