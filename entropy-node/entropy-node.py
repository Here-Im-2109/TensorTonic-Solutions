import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y)
    rv, rc = np.unique(y, return_counts=True)
    p = rc / (np.sum(rc))
    # Filter out zero probabilities before computing logarithms
    p = p[p > 0]

    # Calculate Entropy
    return (-np.sum(p * np.log2(p)))