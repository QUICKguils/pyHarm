"""Factory for the bifurcation handling objects.

Attributes:
    Bifurcator_dico (dict):
      Dictionary that contains ABCBifurcator objects as values
      and their factory_keyword attribute as their key.
"""

from pyHarm.Bifurcators.ABCBifurcator import ABCBifurcator
from pyHarm.Bifurcators.BifurcatorNone import BifurcatorNone
from pyHarm.Bifurcators.BifurcatorJump import BifurcatorJump
from pyHarm.Bifurcators.BifurcatorPerturbation import BifurcatorPerturbation
from pyHarm.Bifurcators.BifurcatorABE import BifurcatorABE

Bifurcator_dico = {
    BifurcatorNone.factory_keyword: BifurcatorNone,
    BifurcatorJump.factory_keyword: BifurcatorJump,
    BifurcatorPerturbation.factory_keyword: BifurcatorPerturbation,
    BifurcatorABE.factory_keyword: BifurcatorABE,
}
"""
dict:
  Dictionary that contains ABCBifurcator objects as values
  and their factory_keyword attribute as their key.
"""


def generateBifurcator(name_bifurcator: str, bifurcator_options) -> ABCBifurcator:
    """Factory function that creates an ABCBifurcator object.

    Args:
        name_bifurcator (str):
          Type of the ABCBifurcator object that is to be instantiated.
        bifurcator_options (dict):
          Dictionary containing supplementary options for the bifucator to be instantiated.

    Returns:
        ABCBifurcator: Instance of the required ABCBifurcator class.
    """
    return Bifurcator_dico[name_bifurcator](bifurcator_options)
