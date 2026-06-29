import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A) # convert numpy array
    return A.T # return the transpose value of the matrix
