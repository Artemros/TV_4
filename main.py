from math import sqrt, exp, pi
import numpy as np
import matplotlib.pyplot as plt

import distributions as dst


def kernel_function(x: float):
    return exp(-x * x / 2) / sqrt(2 * pi)


def kernel_approximation(xs: np.ndarray, sel: np.ndarray, k):
    n = len(sel)
    s = sqrt((sel * sel).sum() / n - (sel.sum() / n) ** 2)
    h = 1.06 * s * (n ** (-0.2)) * k
    res = np.zeros_like(xs)
    for i in range(len(xs)):
        for v in sel:
            res[i] += kernel_function((xs[i] - v) / h)
        res[i] /= n * h
    return res


ds = dst.get_distributions()
ns = [20, 60, 100]
ks = [0.5, 1, 2]

for d in ds:
    fig, ax = plt.subplots(len(ks), len(ns))
    rng = (-4, 4) if not d.discrete() else (6, 14)
    for i in range(len(ns)):
        sel = np.array(dst.selection(d, ns[i]))
        x = np.linspace(*rng, 100)
        y1 = list(map(d.f, x))
        for j in range(len(ks)):
            y2 = kernel_approximation(x, sel, ks[j])
            ax1 = ax[i, j]
            ax1.plot(x, y1)
            ax1.plot(x, y2, label='diagnosis')
            ax1.legend()
            ax1.set_xlabel('x' if i == len(ns) - 1 else '')
            ax1.set_ylabel(f'f(x), n = {ns[i]}')
            if i == 0:
                ax1.set_title((d.name if j == 1 else '') + f"\n\nh' = h * {ks[j]}")
    plt.show()
