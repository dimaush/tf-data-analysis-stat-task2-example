import numpy as np
from scipy.stats import norm


chat_id = 337677338

def solution(p: float, x: np.array) -> tuple:
    s = 0.026
    x -= s
    
    n = x.shape[0]
    m = np.mean(x)
    d = (np.var(x) / n) ** 0.5
    
    alpha, beta = (1 - p) / 2, (1 - p) / 2
    z1 = norm.ppf(alpha)
    z2 = norm.ppf(1 - beta)
    
    return s + max(np.amax(x), 2 * (m + z1 * d)), s + max(np.amax(x), 2 * (m + z2 * d))
