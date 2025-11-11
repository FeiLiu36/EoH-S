import pickle
import numpy as np


class GetData:
    def __init__(self, n_instance, min_cities=20, max_cities=200, min_capacity=10, max_capacity=150):
        self.n_instance = n_instance
        self.min_cities = min_cities
        self.max_cities = max_cities
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity

    def generate_instances(self):
        """each instance -> (coordinates, distances, demands, capacity)"""
        #np.random.seed(2024)
        instance_data = []
        instance_data_lkh = []
        for _ in range(self.n_instance):
            # Randomly select number of cities and capacity within specified ranges
            n_cities = np.random.randint(self.min_cities, self.max_cities + 1)
            capacity = np.random.randint(self.min_capacity, self.max_capacity + 1)

            coordinates = np.random.rand(n_cities, 2)
            # Mild skew toward smaller values
            weights = np.linspace(5, 1.0, 9)  # Linearly decreasing weights from 1.5 to 1.0
            probabilities = weights / weights.sum()  # Normalize to get probabilities
            demands = np.random.choice(np.arange(1, 10), size=n_cities, p=probabilities)

            distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
            instance_data.append((coordinates, distances, demands, capacity))
            instance_data_lkh.append((coordinates[0].tolist(), coordinates[1:].tolist(), demands[1:].tolist(), capacity))

        return instance_data, instance_data_lkh


if __name__ == '__main__':
    # Generate 10 instances with varying cities (20-200) and capacities (20-60)
    gd = GetData(256)
    data, data_lkh= gd.generate_instances()

    with open('../data_cvrp_10_150_256.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open('../data_cvrp_10_150_256_lkh.pkl', 'wb') as f:
        pickle.dump(data_lkh, f)

