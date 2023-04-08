import numpy as np
from scipy.stats import norm


chat_id = 337677338

def solution(p: float, x: np.array) -> tuple:
    s = 0.026
    x -= s

    s = 0.026
    x -= s

    n = x.shape[0]
    x_max = np.amax(x)
    alpha = (1 - p) ** (1 / n)
    return (s + x_max, s + x_max / alpha)
