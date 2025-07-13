"""Factory for the bifurcation handling objects.

Attributes:
    Bifurcator_dico (dict):
      Dictionary that contains ABCBifurcator objects as values
      and their factory_keyword attribute as their key.
"""

from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator
from pyHarm.Bifurcators.BifurcatorABE import BifurcatorABE
from pyHarm.Bifurcators.BifurcatorHill import BifurcatorHill
from pyHarm.Bifurcators.BifurcatorPerturbation import BifurcatorPerturbation

Bifurcator_dico = {
    BifurcatorABE.factory_keyword: BifurcatorABE,
    BifurcatorPerturbation.factory_keyword: BifurcatorPerturbation,
    BifurcatorHill.factory_keyword: BifurcatorHill,
}
"""
dict:
  Dictionary that contains ABCBifurcator objects as values
  and their factory_keyword attribute as their key.
"""


def generateBifurcator(name_bifurcator: str | None, bifurcator_options) -> ABCBifurcator | None:
    """Factory function that creates an ABCBifurcator object.

    Args:
        name_bifurcator (str | None):
          Type of the ABCBifurcator object that is to be instantiated.
        bifurcator_options (dict):
          Dictionary containing supplementary options for the bifucator to be instantiated.

    Returns:
        ABCBifurcator: Instance of the required ABCBifurcator class.
    """
    if name_bifurcator is None:
        return None
    return Bifurcator_dico[name_bifurcator](bifurcator_options)
