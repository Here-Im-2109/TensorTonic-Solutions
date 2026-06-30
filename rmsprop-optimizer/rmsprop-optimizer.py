import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    st = np.array(s)
    wt = np.array(w)
    g = np.array(g)

    st = beta * st + (1 - beta) * (g**2)
    wt = wt - ((lr / np.sqrt(st + eps) * g))

    return (wt, st)