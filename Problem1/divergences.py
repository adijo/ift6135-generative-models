import numpy as np
import scipy.stats
import scipy as sp


def jensen_shannon_divergence(p, q):
    p, q = np.asarray(p), np.asarray(q)
    p, q = p/p.sum(), q/q.sum()
    m = 0.5*(p + q)
    return sp.stats.entropy(p, m, np.e)/2.0 + sp.stats.entropy(q, m, np.e)/2.0


if __name__ == '__main__':
    # Test
    print(jensen_shannon_divergence([0.6, 0.4], [0.5, 0.5]))
