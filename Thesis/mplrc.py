"""Set some global matplotlib parameters."""

import matplotlib as mpl

# Text width of the report, as measured in the latex document.
REPORT_TW = 5.512  # [in]

# ULiège branding colors, form official chart.
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


def load_rcparams(style="report") -> None:
    # When saved as svg, consider fonts as fonts, not as paths.
    # Unless the font is not embeddable, it's better to treat fonts as fonts.
    # This gives better displays and allows to select text in PDFs.
    # It also allows the text to be modified with e.g. Inkscape.
    mpl.rcParams["svg.fonttype"] = "none"

    # Choose compiler for pgf backend, when figs are saved as pgf.
    mpl.rcParams["pgf.texsystem"] = "lualatex"

    # Plot style
    mpl.rcParams["figure.constrained_layout.use"] = True
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.linewidth"] = 0.6
    mpl.rcParams["grid.alpha"] = 0.3
    custom_colorcycler = mpl.cycler(
        color=[
            UCOLOR["TealDark"],  # C0
            UCOLOR["OrangeDark"],  # C1
            UCOLOR["BlueLight"],  # C2
            UCOLOR["PurpleLight"],  # C3
            UCOLOR["GreenLight"],  # C4
            UCOLOR["Yellow"],  # C5
            UCOLOR["Red"],  # C6
            UCOLOR["GrayDark"],  # C7
        ]
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(custom_colorcycler)

    # Here, figure.dpi is set to scale nicely on the screen.
    # If one desire to save the plot in raster format,
    # higher dpi values should be used (e.g., 300 dpi).
    mpl.rcParams["figure.dpi"] = 150

    # Report figures
    if style == "report":
        mpl.rcParams["figure.figsize"] = (REPORT_TW, REPORT_TW / 1.618033989)  # golden *_*

        mpl.rcParams["lines.linewidth"] = 1

        # These params will control how the pgf backend will treat the fonts.
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.size"] = 11

        # These options are not relevant when saved as pgf figure,
        # as the tex compilation will use the latex fonts anyways.
        mpl.rcParams["mathtext.fontset"] = "dejavusans"
        mpl.rcParams["font.serif"] = ["STIX Two Text"] + mpl.rcParams["font.serif"]
        mpl.rcParams["font.sans-serif"] = ["Noto Sans"] + mpl.rcParams["font.sans-serif"]

        # NOTE: possible to set them relative to font.size
        # E.g., use "medium", "small", "x-small", etc.
        # Here I set the fontsize according to the
        # \small and \footnotesize definitions in latex2e.
        mpl.rcParams["axes.titlesize"] = 10
        mpl.rcParams["axes.labelsize"] = 10
        mpl.rcParams["xtick.labelsize"] = 9
        mpl.rcParams["ytick.labelsize"] = 9
        mpl.rcParams["legend.fontsize"] = 9

    # Presentation figures
    if style == "slide":
        mpl.rcParams["lines.linewidth"] = 2

        # These params will control how the pgf backend will treat the fonts.
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.size"] = 15

        # These options are not relevant when saved as pgf figure,
        # as the tex compilation will use the latex fonts anyways.
        mpl.rcParams["mathtext.fontset"] = "dejavusans"
        mpl.rcParams["font.sans-serif"] = ["Noto Sans"] + mpl.rcParams["font.sans-serif"]

        # Those sizes are relative to font.size
        mpl.rcParams["axes.titlesize"] = "medium"
        mpl.rcParams["axes.labelsize"] = "medium"
        mpl.rcParams["xtick.labelsize"] = "small"
        mpl.rcParams["ytick.labelsize"] = "small"
