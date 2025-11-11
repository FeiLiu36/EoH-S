import pickle
import numpy as np


class GetData:
    def __init__(self, n_instance, min_cities=20, max_cities=200, min_capacity=20, max_capacity=100):
        self.n_instance = n_instance
        self.min_cities = min_cities
        self.max_cities = max_cities
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity

    def generate_instances(self):
        """each instance -> (coordinates, distances, demands, capacity)"""
        np.random.seed(2024)
        instance_data = []
        for _ in range(self.n_instance):
            # Randomly select number of cities and capacity within specified ranges
            n_cities = np.random.randint(self.min_cities, self.max_cities + 1)
            capacity = np.random.randint(self.min_capacity, self.max_capacity + 1)

            coordinates = np.random.rand(n_cities, 2)
            demands = np.random.randint(1, 10, size=n_cities)
            distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
            instance_data.append((coordinates, distances, demands, capacity))
        return instance_data


if __name__ == '__main__':
    # Generate 10 instances with varying cities (20-200) and capacities (20-60)
    gd = GetData(128)
    data = gd.generate_instances()
    with open('data_cvrp.pkl', 'wb') as f:
        pickle.dump(data, f)

    prompt_code_temp = "import numpy as np\n\
    def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int: \n\
    \n\
        '''Design a novel algorithm to select the next node in each step.\n\
    \n\
        Args:\n\
        current_node: ID of the current node.\n\
        depot: ID of the depot.\n\
        unvisited_nodes: Array of IDs of unvisited nodes.\n\
        rest_capacity: rest capacity of vehicle \n\
        demands: demands of nodes \n\
        distance_matrix: Distance matrix of nodes.\n\
    \n\
        Return:\n\
        ID of the next node to visit.\n\
        '''\n\
        next_node = unvisited_nodes[0]\n\
    \n\
        return next_node\n"

    print(prompt_code_temp)
