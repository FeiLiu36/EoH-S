import numpy as np
def priority( item : float , bins : np.ndarray ) -> np.ndarray:
    max_bin_cap = max( bins )
    bin_density = np .sum( bins ) / ( item * len ( bins ) )
    log_adj = np.log ( bins + 1) / np.log ( max_bin_cap + 1)
    score = ( bins - max_bin_cap ) **2 / item + bins **2 / ( item **2) + bins **2 / ( item **3) + bin_density * bins
    score [ bins > item ] = - score [ bins > item ]
    score [1:] -= score [: -1]
    score *= log_adj
    score += log_adj * bins
    score *= log_adj
    new_component = bins / ( item - bins + 1)
    score += new_component
    new_component = bins * np.log ( bins + 1) / ( item * np.log ( max_bin_cap + 1) ) * (1 - bins / item )
    score += new_component
    new_adjustment = ( bins / item ) * log_adj
    score += new_adjustment
    bins_adjusted = bins / item
    score += np.log ( bins_adjusted + 1) / np.log ( max_bin_cap + 1)
    new_component = ( bins - 1) / ( item - bins + 1) * log_adj / np.log ( max_bin_cap + 1)
    score += new_component
    new_component = log_adj * bins / ( item - bins )
    score += new_component
    new_component = bins * np.log ( bins + 1) / ( item **2) * (1 - bins / item )
    score += new_component
    return score
