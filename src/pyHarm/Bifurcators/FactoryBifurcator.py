"""
This module contains the factory of the branching objects.

Attributes:
    Branching_dico (dict): Dictionary that contains ABCBranching objects as values and their factory_keyword attribute as their key.
"""

from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator
from pyHarm.Bifurcators.BifurcatorHill import BifurcatorHill

Bifurcator_dico = {
    BifurcatorHill.factory_keyword: BifurcatorHill,
}
"""dict: Dictionary that contains ABCBranching objects as values and their factory_keyword attribute as their key."""


def generateBifurcator(name_bifurcator, bifurcator_options) -> ABCBifurcator:
    """Factory function that creates an ABCBifurcator object.

    Args:
        name_bifurcator (str): Type of the ABCBifurcator object that is to be
          instantiated.
        bifurcator_options (dict): dictionary containing supplementary options
          for the branching to be instantiated.

    Returns:
        ABCBifurcator: Instance of the required ABCBifurcator class.
    """
    return Bifurcator_dico[name_bifurcator](**bifurcator_options)
