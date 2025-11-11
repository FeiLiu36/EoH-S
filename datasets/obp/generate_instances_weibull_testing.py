import numpy as np
import pickle as pkl
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

def generate_normal_dataset_fm(num_instances, num_items, capacity_limit):
    """
    Generate dataset with items following normal distributions with parameters
    sampled from candidate sets.

    Args:
        num_instances: Number of instances to generate
        num_items: [min_items, max_items] range for number of items
        capacity_limit: [min_capacity, max_capacity] range for knapsack capacity
        mean_candidates: List of candidate mean parameters for normal distribution
        std_candidates: List of candidate standard deviation parameters for normal distribution
    """
    dataset = {}
    for n_items in num_items:
        for capacity in capacity_limit:
            for i in range(num_instances):

                # Randomly select shape and scale parameters from candidate sets
                mean = capacity/2
                std = capacity/5

                instance = {
                    'capacity': capacity,
                    'num_items': n_items,
                    'items': [],
                    'normal_mean': mean,  # Store the parameters used for this instance
                    'normal_std': std
                }

                items = []

                # Generate random samples from normal distribution with sampled parameters
                samples = np.random.normal(mean, std, n_items)

                # Clip the samples at the specified limits (minimum 1, maximum capacity)
                samples = np.clip(samples, 1, capacity)

                # Round the item sizes to the nearest integer
                sizes = np.round(samples).astype(int)

                # Add the items to the instance
                for size in sizes:
                    items.append(size)

                instance['items'] = np.array(items)
                dataset[f'instance_normal_{n_items}_{capacity}_{i}'] = instance

    return dataset

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
    for n_items in num_items:
        for capacity in capacity_limit:
            for i in range(num_instances):

                # Randomly select shape and scale parameters from candidate sets
                shape = shape_candidates[0]
                scale = scale_candidates[0]

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
                dataset[f'instance_weibull_{n_items}_{capacity}_{i}'] = instance

    return dataset

def generate_exp_dataset_fm(num_instances, num_items, capacity_limit):
    """
    Generate dataset with items following normal distributions with parameters
    sampled from candidate sets.

    Args:
        num_instances: Number of instances to generate
        num_items: [min_items, max_items] range for number of items
        capacity_limit: [min_capacity, max_capacity] range for knapsack capacity
        mean_candidates: List of candidate mean parameters for normal distribution
        std_candidates: List of candidate standard deviation parameters for normal distribution
    """
    dataset = {}
    for n_items in num_items:
        for capacity in capacity_limit:
            for i in range(num_instances):

                # Randomly select shape and scale parameters from candidate sets
                scale = capacity/4


                instance = {
                    'capacity': capacity,
                    'num_items': n_items,
                    'items': [],
                    'scale': scale
                }

                items = []

                # Generate random samples from normal distribution with sampled parameters
                samples = np.random.exponential(scale, n_items)

                # Clip the samples at the specified limits (minimum 1, maximum capacity)
                samples = np.clip(samples, 1, capacity)

                # Round the item sizes to the nearest integer
                sizes = np.round(samples).astype(int)

                # Add the items to the instance
                for size in sizes:
                    items.append(size)

                instance['items'] = np.array(items)
                dataset[f'instance_exp_{n_items}_{capacity}_{i}'] = instance

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
    shape_candidates = [3]  # k parameter (shape)
    scale_candidates = [45]  # lambda parameter (scale)

    # First plot the distributions to visualize them
    plot_weibull_distributions(shape_candidates, scale_candidates)

    dataset = generate_weibull_dataset_fm(
        5,
        [1000, 5000, 10000],
        [200,500],
        shape_candidates,
        scale_candidates
    )

    # Define candidate sets for normal distribution parameters



    dataset_normal = generate_normal_dataset_fm(
        5,
        [1000, 10000],
        [100]
    )


    dataset_exp = generate_exp_dataset_fm(
        5,
        [1000, 10000],
        [100]
    )

    combined_dataset = {}

    # Merge the Weibull dataset
    for key, value in dataset.items():
        combined_dataset[key] = value

    # Merge the Normal dataset
    for key, value in dataset_normal.items():
        # If key already exists, you might want to rename it or merge the values
        if key in combined_dataset:
            # Option 1: Rename the key by adding a suffix
            combined_dataset[f"{key}_normal"] = value
        else:
            combined_dataset[key] = value

    # Merge the Exponential dataset
    for key, value in dataset_exp.items():
        # If key already exists, you might want to rename it or merge the values
        if key in combined_dataset:
            # Option 1: Rename the key by adding a suffix
            combined_dataset[f"{key}_exp"] = value
        else:
            combined_dataset[key] = value

    pkl.dump(combined_dataset, open('dataset_testing.pkl', 'wb'))
