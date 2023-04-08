import numpy as np
from scipy.stats import norm


chat_id = 337677338

def solution(p: float, x: np.array) -> tuple:
    s = 0.026
    x -= s

    n = x.shape[0]
    x_max = np.amax(x)
    return s + (x_max - s) / (((1 + p) / 2) ** (1 / n)), s + (x_max - s) / (((1 - p) / 2) ** (1 / n))
