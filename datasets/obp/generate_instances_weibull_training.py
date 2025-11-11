import numpy as np
import pickle as pkl
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

def generate_weibull_dataset_fm(num_instances, num_items, capacity_limit, shape_candidates, scale_candidates):
    """
    Generate dataset with items following Weibull distributions with parameters
    sampled from candidate sets.

    Args:
        num_instances: Number of instances to generate
        num_items: [min_items, max_items] range for number of items
        capacity_limit: [min_capacity, max_capacity] range for knapsack capacity
        shape_candidates: List of candidate shape parameters (k) for Weibull distribution
        scale_candidates: List of candidate scale parameters (lambda) for Weibull distribution
    """
    dataset = {}

    for i in range(num_instances):
        np.random.seed(2025 + i)
        n_items = np.random.randint(low=num_items[0], high=num_items[1], size=1)[0]
        #capacity = np.random.randint(low=capacity_limit[0], high=capacity_limit[1], size=1)[0]
        capacity = capacity_limit[0]

        # Randomly select shape and scale parameters from candidate sets
        shape = np.random.choice(shape_candidates)
        scale = np.random.choice(scale_candidates)

        instance = {
            'capacity': capacity,
            'num_items': n_items,
            'items': [],
            'weibull_shape': shape,  # Store the parameters used for this instance
            'weibull_scale': scale
        }

        items = []

        # Generate random samples from Weibull with sampled parameters
        samples = np.random.weibull(shape, n_items) * scale

        # Clip the samples at the specified limit
        samples = np.clip(samples, 1, capacity)

        # Round the item sizes to the nearest integer
        sizes = np.round(samples).astype(int)

        # Add the items to the instance
        for size in sizes:
            items.append(size)

        instance['items'] = np.array(items)
        dataset[f'instance_{i}'] = instance

    return dataset


def plot_weibull_distributions(shape_candidates, scale_candidates, num_samples=10000, capacity_limit=100):
    """
    Plot Weibull distributions for different parameter combinations.

    Args:
        shape_candidates: List of shape parameters (k) to plot
        scale_candidates: List of scale parameters (lambda) to plot
        num_samples: Number of samples to generate for each distribution
        capacity_limit: Maximum value to clip the distributions
    """
    plt.figure(figsize=(12, 8))

    # Generate all combinations of shape and scale parameters
    for shape in shape_candidates:
        for scale in scale_candidates:
            # Generate samples
            samples = np.random.weibull(shape, num_samples) * scale
            samples = np.clip(samples, 1, capacity_limit)

            # Plot the distribution
            sns.kdeplot(samples,
                        label=f'Shape={shape}, Scale={scale}',
                        linewidth=2)

    plt.title('Weibull Distributions with Different Parameters', fontsize=14)
    plt.xlabel('Item Size', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, capacity_limit)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Define candidate sets for Weibull parameters
    shape_candidates = [1,3,5]  # k parameter (shape)
    scale_candidates = [5, 10, 20, 40, 80]  # lambda parameter (scale)

    # First plot the distributions to visualize them
    plot_weibull_distributions(shape_candidates, scale_candidates)

    dataset = generate_weibull_dataset_fm(
        128,
        [200, 2000],
        [100,100],
        shape_candidates,
        scale_candidates
    )

    pkl.dump(dataset, open('dataset_100_2k_128_5_80.pkl', 'wb'))
