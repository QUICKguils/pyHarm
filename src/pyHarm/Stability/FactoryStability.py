"""Factory for the stability analyzer objects.

Attributes:
    Stability_dico (dict):
      Dictionary that contains ABCStability objects as values
      and their factory_keyword attribute as their key.
"""

from pyHarm.Stability import ABCStability
from pyHarm.Stability.StabilityHill import StabilityHill

Stability_dico = {
    StabilityHill.factory_keyword: StabilityHill,
}
"""
dict:
  Dictionary that contains ABCStability objects as values
  and their factory_keyword attribute as their key.
"""


def generateStability(name_stability: str | None, stability_options) -> ABCStability:
    """Factory function that creates an ABCStability object.

    Args:
        name_stability (str):
          Type of the ABCStability object that is to be instantiated.
        stability_options (dict):
          Dictionary containing supplementary options for the stability to be instantiated.

    Returns:
        ABCStability: Instance of the required ABCStability class.
    """
    if name_stability is None:
        return None
    return Stability_dico[name_stability](stability_options)
