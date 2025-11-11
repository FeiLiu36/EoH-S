import numpy as np
import pickle as pkl
import elkai

class GetData():
    def __init__(self, n_instance, n_cities):
        self.n_instance = n_instance
        self.n_cities = n_cities



    def lkh(self,distance_matrix):

        result_matrix = (distance_matrix * 100).tolist()
        cities = elkai.DistanceMatrix(result_matrix)
        route = cities.solve_tsp(runs=10)  # Will return something like [0, 2, 1, 0]

        # Calculate route length
        route_length = 0
        for i in range(len(route) - 1):
            from_city = route[i]
            to_city = route[i + 1]
            route_length += distance_matrix[from_city][to_city]

        print("Route:", route)
        print("Route length:", route_length)
        return route_length

    def generate_instances(self):
        np.random.seed(2024)
        instance_data = []

        for _ in range(self.n_instance):
            n_c = np.random.randint(self.n_cities[0], self.n_cities[1])
            coordinates = np.random.rand(n_c, 2)
            distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
            baseline_length = self.lkh(distances)
            instance_data.append((coordinates, distances, baseline_length))
        return instance_data

if __name__ == '__main__':
    n_cities = [10,1000]
    getdata = GetData(n_instance=32, n_cities=n_cities)
    dataset = getdata.generate_instances()
    pkl.dump(dataset, open('dataset_tsp.pkl', 'wb'))