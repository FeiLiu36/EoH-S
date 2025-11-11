import pickle
import numpy as np


class GetData:
    def __init__(self, n_instance=16, n_cities=20, capacity=100):
        self.n_instance = n_instance
        self.cities = n_cities
        self.capacity = capacity

    def generate_instances(self):
        """each instance -> (coordinates, distances, demands, capacity)"""
        #np.random.seed(2024)
        instance_data = []
        instance_data_lkh = []
        for n_c in self.cities:
            for _ in range(self.n_instance):
                capacity = np.random.randint(self.capacity[0],self.capacity[1])
                print(capacity)
                coordinates = np.random.rand(n_c, 2)
                demands = np.random.randint(1, 10, size=n_c)
                distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
                instance_data.append((coordinates, distances, demands, capacity))
                instance_data_lkh.append((coordinates[0].tolist(), coordinates[1:].tolist(), demands[1:].tolist(), capacity))
        return instance_data, instance_data_lkh


if __name__ == '__main__':
    # Generate 10 instances with varying cities (20-200) and capacities (20-60)
    size = [50,100,200,500]
    capacity = [40,150]
    gd = GetData(32,size,capacity)

    data, data_lkh= gd.generate_instances()

    with open('../data_cvrp_50_500_32.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open('../data_cvrp_50_500_32_lkh.pkl', 'wb') as f:
        pickle.dump(data_lkh, f)
