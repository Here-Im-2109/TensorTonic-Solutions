import numpy as np

def _sigmoid(z):
    """ Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    
    n_samples, n_features = X.shape

    # Initialize parameters
    w = np.zeros(n_features) # weights
    b = 0.0 # bias

    # Gradient descent
    for _ in range(steps):
        # Linear predictions
        z = X @ w + b # mat_mul

        # Predicted probabilities
        y_pred = _sigmoid(z)

        # Calculate Error
        error = y_pred - y

        # Compute gradients
        dw = (X.T @ error) / n_samples
        db = np.mean(error)

        # Update parameters
        w = w - (lr * dw)
        b = b - (lr * db)

    return w, b