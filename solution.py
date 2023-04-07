import numpy as np
from scipy.stats import norm


chat_id = 337677338

def solution(p: float, x: np.array) -> tuple:
    n = x.shape[0]
    alpha = 1 - p
    s = 0.026
    x -= s
    b0 = (n + 1) / n * np.amax(x)
    return s + max( b0, 2 * (np.mean(x) - (np.var(x) / n) ** 0.5 * norm.ppf(1 - alpha / 2)) ), \
           s + max( b0, 2 * (np.mean(x) - (np.var(x) / n) ** 0.5 * norm.ppf(alpha / 2)) )
