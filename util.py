import numpy as np

def moon(n, noise):
    # m = n * noise
    m = int(n*noise)
    t = np.pi * np.random.rand(2 * m, 1)
    x = 6 * np.cos(t)
    y = 6 * np.sin(t)
    z = np.hstack([x, y])
    a = np.random.randn(m, 2) + z[:m]
    b = np.random.randn(m, 2) + np.array([6, 0]) + z[m:] * np.array([1, -1])
    x = np.concatenate([a, b]) * 0.15
    n = len(x)
    return x, n