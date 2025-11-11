
import pickle as pkl
import multiprocessing
import numpy as np
import torch
import elkai
import matplotlib.pyplot as plt
from typing import List, Tuple


class GetData():
    def __init__(self, n_instance, n_cities, distribution='uniform', distribution_args=None):
        self.n_instance = n_instance
        self.n_cities = n_cities  # Tuple (min, max)
        self.distribution = distribution
        self.distribution_args = distribution_args or {}

        # Initialize the appropriate distribution generator
        if self.distribution == 'uniform':
            self.sampler = self.sample_uniform
        elif self.distribution == 'cluster':
            from modules import Cluster  # Replace with your actual import
            self.sampler = Cluster(**self.distribution_args).sample
        elif self.distribution == 'mixed':
            from modules import Mixed
            self.sampler = Mixed(**self.distribution_args).sample
        elif self.distribution == 'gaussian_mixture':
            from modules import Gaussian_Mixture
            self.sampler = Gaussian_Mixture(**self.distribution_args).sample
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def sample_uniform(self, size):
        return torch.rand(size)

    def lkh(self, distance_matrix):
        result_matrix = (distance_matrix * 1000).tolist()
        cities = elkai.DistanceMatrix(result_matrix)
        route = cities.solve_tsp(runs=10)

        # Calculate route length
        route_length = 0
        for i in range(len(route) - 1):
            from_city = route[i]
            to_city = route[i + 1]
            route_length += distance_matrix[from_city][to_city]

        print("Route:", route)
        print("Route length:", route_length)
        return route_length

    def _process_instance(self, args) -> Tuple[np.ndarray, np.ndarray, float]:
        """Helper function for parallel processing"""
        n_c = args
        coords = self.sampler(size=(1, n_c, 2)).squeeze(0).numpy()
        distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
        baseline_length = self.lkh(distances)
        # baseline_length = 1.0
        return (coords, distances, baseline_length)

    def generate_instances(self, n_processes: int = None) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        torch.manual_seed(2024)

        # Generate random city counts for all instances
        n_cities_list = [np.random.randint(self.n_cities[0], self.n_cities[1])
                         for _ in range(self.n_instance)]

        # Set default number of processes to CPU count if not specified
        if n_processes is None:
            n_processes = multiprocessing.cpu_count()

        # Use multiprocessing to parallelize instance generation
        with multiprocessing.Pool(processes=n_processes) as pool:
            instance_data = pool.map(self._process_instance, n_cities_list)

        self.instance_data = instance_data  # Store for plotting
        return instance_data

    def plot_instances(self, n_plot=4):
        if not hasattr(self, 'instance_data'):
            print("Please call generate_instances() first.")
            return

        plt.figure(figsize=(15, 4))
        for i in range(min(n_plot, len(self.instance_data))):
            coords, _, _ = self.instance_data[i]
            plt.subplot(1, n_plot, i + 1)
            plt.scatter(coords[:, 0], coords[:, 1], c='blue', alpha=0.7)
            plt.title(f'Instance {i + 1}')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().set_aspect('equal', adjustable='box')
        plt.suptitle(f"TSP Instances with '{self.distribution}' Distribution", fontsize=14)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
 
    n_cities = [10,200]
    n_instance = 32
    distributions = ['cluster_n3_std03','cluster_n5_std03','cluster_n10_std03','cluster_n3_std07','cluster_n5_std07','cluster_n10_std07']
    for dis in distributions:
        n_cluster = int(dis.split('_')[1][1:])  # e.g., 'n3' -> 3
        std = float(dis.split('_')[2][3:]) / 100  # e.g., 'std07' -> 0.07

        print(f"Distribution: {dis}, n_cluster: {n_cluster}, std: {std}")

        datas = pkl.load(open('./dataset_tsp_'+str(n_cluster)+'_'+str(std)+'_32.pkl', 'rb'))
        print(len(datas))


