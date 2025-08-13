"""Technical background: illustration of AFT with a Duffing"""

import numpy as np

from pyHarm.DynamicOperator import compute_DFT

def main() -> None:
    nh = 6
    nti = 2048

    # Discrete Fourier transform operator
    DFTO = compute_DFT(nti, nh)

    x_h = np.zeros(2 * nh + 1)
    x_h[2] = 0.6
    x_h[4] = -0.5
    x_t = x_h @ DFTO["ft"]

    plot(nh, nti, x_h, x_t, DFTO)


# We now apply a transformation on the time displacement
def deformation(x_t):
    return x_t**3  # cubic NL force of a duffing


def plot(nh, nti, x_h, x_t, DFTO) -> None:
    import matplotlib.pyplot as plt
    from ..mplrc import load_rcparams, REPORT_TW

    load_rcparams()
    fig, axs = plt.subplots(2, 2, figsize=(0.9*REPORT_TW, 0.8*REPORT_TW))

    def label(i):
        if i == 0:
            return r"$c_0$"
        else:
            if i % 2 == 0:
                return f"$s_{(i + 1) // 2}$"
            else:
                return f"$c_{(i + 1) // 2}$"

    for ax in axs[:, 0]:
        ax.set_xticks(np.arange(2 * nh + 1))
        ax.set_xticklabels([label(i) for i in (np.arange(2 * nh + 1))])
    for ax in axs[:, 1]:
        ax.set_xticks([0, nti])
        ax.set_xticklabels(["0", "T"])

    axs[0, 0].bar(np.arange(2 * nh + 1), x_h)
    axs[0, 1].plot(x_t)
    axs[1, 1].plot(deformation(x_t))
    axs[1, 0].bar(np.arange(2 * nh + 1), deformation(x_t) @ DFTO["tf"])

    axs[0, 0].set_ylabel("qhat(k)")
    axs[0, 1].set_ylabel("q(t)")
    axs[1, 0].set_ylabel("fhat(k)")
    axs[1, 1].set_ylabel("f(t)")

    # axs[0, 0].set_xlabel("Fourier coefficients")
    # axs[0, 1].set_xlabel("$t$")
    # axs[1, 0].set_xlabel("Fourier coefficients")
    # axs[1, 1].set_xlabel("$t$")

    for ax in axs.flat:
        ax.grid(axis="x")
    fig.get_layout_engine().set(hspace=0.2, wspace=0.1)

    fig.show()
