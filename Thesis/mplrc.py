"""Set some global matplotlib parameters."""

import matplotlib as mpl

# Measured in the latex document.
# Obviously depends on the paper type and margins width.
REPORT_TW = 5.90666  # [in]


def load_rcparams(style="running") -> None:
    mpl.rcParams["figure.constrained_layout.use"] = True
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.linewidth"] = 0.8
    mpl.rcParams["grid.alpha"] = 0.3
    # Here, figure.dpi is set to scale nicely on the screen.
    # If one desire to save the plot in raster format,
    # higher dpi values should be used (e.g., 250dpi).
    mpl.rcParams["figure.dpi"] = 109  # qhd 27in = 109 dpi

    # Running figures
    if style == "running":
        pass

    # Report figures
    if style == "report":
        mpl.rcParams["figure.figsize"] = (REPORT_TW, REPORT_TW / 1.618033989)

        # mpl.rcParams['mathtext.fontset'] = 'stix'
        # mpl.rcParams['font.family'] = 'serif'
        # # FIX: Plots can not be saved as pdf with some fonts
        # # E.g., STIX Two Text or Source Sans 3 causes the pdf backend to crash.
        # mpl.rcParams['font.serif'] = ['STIX Two Text'] + mpl.rcParams['font.serif']
        # mpl.rcParams['font.size'] = 11

        mpl.rcParams["font.sans-serif"] = ["Noto Sans"] + mpl.rcParams[
            "font.sans-serif"
        ]
        mpl.rcParams["font.size"] = 11

        # Those sizes are relative to font.size
        mpl.rcParams["axes.titlesize"] = "small"
        mpl.rcParams["axes.labelsize"] = "small"
        mpl.rcParams["xtick.labelsize"] = "x-small"
        mpl.rcParams["ytick.labelsize"] = "x-small"

    # Presentation figures
    if style == "slide":
        mpl.rcParams["mathtext.fontset"] = "stixsans"
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = ["Noto Sans"] + mpl.rcParams[
            "font.sans-serif"
        ]
        mpl.rcParams["font.size"] = 15

        # Those sizes are relative to font.size
        mpl.rcParams["axes.titlesize"] = "medium"
        mpl.rcParams["axes.labelsize"] = "medium"
        mpl.rcParams["xtick.labelsize"] = "small"
        mpl.rcParams["ytick.labelsize"] = "small"
