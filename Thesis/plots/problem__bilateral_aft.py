"""Technical background: illustration of AFT with a Duffing"""

import numpy as np

from pyHarm.DynamicOperator import compute_DFT


def main() -> None:
    nh = 6
    nti = 2048

    # Discrete Fourier transform operator
    DFTO = compute_DFT(nti, nh)

    # Demine periodic motion as simple sinus
    q_k = np.zeros(2 * nh + 1)
    q_k[2] = 1
    q_t = q_k @ DFTO["ft"]

    plot(nh, nti, q_t, DFTO)


def compute_force(q_t):
    """Bilateral gap implemented in pyHarm."""
    gap = 0.5
    k = 100
    f_t = np.zeros_like(q_t)
    contact_mask = np.abs(q_t) > gap
    f_t[contact_mask] = k * (np.abs(q_t[contact_mask]) - gap)
    return f_t


def plot(nh, nti, x_t, DFTO) -> None:
    import matplotlib.pyplot as plt
    from ..mplrc import load_rcparams, REPORT_TW

    load_rcparams()
    fig, axs = plt.subplots(2, 2, figsize=(0.9 * REPORT_TW, 0.8 * REPORT_TW))

    def label(i):
        if i == 0:
            return r"$c_0$"
        else:
            if i % 2 == 0:
                return f"$s_{{{(i + 1) // 2}}}$"
            else:
                return f"$c_{{{(i + 1) // 2}}}$"

    for ax in axs.flat[[-1]]:
        ax.set_xticks(np.arange(2 * nh + 1))
        ax.set_xticklabels([label(i) for i in (np.arange(2 * nh + 1))])
    for ax in axs.flat[:-1]:
        ax.set_xticks([0, nti])
        ax.set_xticklabels(["0", "T"])

    axs[0, 0].plot(x_t)

    f_t = compute_force(x_t)
    axs[0, 1].plot(f_t)

    f_k = f_t @ DFTO["tf"]
    axs[1, 1].bar(np.arange(2 * nh + 1), f_k)

    f_t_actual = f_k @ DFTO["ft"]
    axs[1, 0].plot(f_t_actual)

    axs[0, 0].set_ylabel(r"$q(t)$")
    axs[0, 1].set_ylabel(r"$f(t)$")
    axs[1, 0].set_ylabel("$f(t)$")
    axs[1, 1].set_ylabel(r"$\hat{f}(k)$")

    for ax in axs.flat:
        ax.grid(axis="x")
    fig.get_layout_engine().set(hspace=0.2, wspace=0.1)

    fig.show()
