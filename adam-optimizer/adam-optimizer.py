import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    Qt = np.array(param)
    gt = np.array(grad)
    mt = np.array(m)
    vt = np.array(v)
    b1 = beta1
    b2 = beta2
    
    mt = b1 * mt + (1 - b1) * gt

    vt = b2 * vt + (1 - b2) * (gt ** 2)

    m_hat = mt / (1 - b1 ** t)

    v_hat = vt / (1 - b2 ** t)

    Qt = Qt - lr * m_hat / (np.sqrt(v_hat) + eps)

    return Qt, mt, vt