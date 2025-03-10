"""Set some global matplotlib parameters."""

import matplotlib as mpl

# Measured in the latex document.
# Obviously depends on the paper type and margins width.
REPORT_TW = 5.90666  # [in]

UCOLOR = {
    # Main teal color.
    "TealDark": "#00707F",
    "TealLight": "#5FA4B0",
    # Beige gray scale.
    "BeigeLight": "#E8E2DE",
    "BeigePale": "#E6E6E1",
    "BeigeDark": "#C6C0B4",
    # Faculty colors.
    "Yellow": "#FFD000",
    "OrangeLight": "#F8AA00",
    "OrangeDark": "#F07F3C",
    "Red": "#E62D31",
    "GreenPale": "#B9CD76",
    "GreenLight": "#7DB928",
    "Green": "#289B38",
    "GreenDark": "#00843B",
    "BlueLight": "#1FBADB",
    "BlueDark": "#005CA9",
    "LavenderDark": "#5B57A2",
    "LavenderLight": "#8DA6D6",
    "PurpleLight": "#A8589E",
    "PurpleDark": "#5B257D",
    "GrayDark": "#8C8B82",
    "GrayLight": "#B5B4A9",
}


def load_rcparams(style="running") -> None:
    mpl.rcParams["figure.constrained_layout.use"] = True
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.linewidth"] = 0.8
    mpl.rcParams["grid.alpha"] = 0.3
    # Here, figure.dpi is set to scale nicely on the screen.
    # If one desire to save the plot in raster format,
    # higher dpi values should be used (e.g., 250dpi).
    mpl.rcParams["figure.dpi"] = 109  # qhd 27in = 109 dpi

    custom_colorcycler = mpl.cycler(color=[
        UCOLOR["TealDark"], # C0
        UCOLOR["OrangeDark"], # C1
        UCOLOR["BlueLight"], # C2
        UCOLOR["PurpleLight"], # C3
        UCOLOR["GreenLight"], # C4
        UCOLOR["Yellow"], # C5
        UCOLOR["Red"], # C6
        UCOLOR["GrayDark"], # C7

    ])
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(custom_colorcycler)

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
