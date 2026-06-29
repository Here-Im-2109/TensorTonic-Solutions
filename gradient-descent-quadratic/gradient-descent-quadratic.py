def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    for _ in range(steps):
        grad = 2 * a * x + b # f(x) = ax² + bx + c -> f'(x) = 2 * a * x + b, we are using 2 cause - 1D Quadratic
        x = x - lr * grad
    return x
        