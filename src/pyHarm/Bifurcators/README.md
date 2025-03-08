# Module Bifurcators

This module contains all the bifurcation detector types that are provided by
pyHarm. The module is organized around an abstract class **ABCBifurcator** and a
**FactoryBifurcator** that is in charge of creating the objects. All bifurcation
detector objects must comply with the **ABCBifurcator** abstract class. The
section below presents the different classes that are available in this module.

## ABCBifurcator

The **ABCBifurcator** class is an abstract class defining the essential
components of any bifurcation detection method.
The interface of this abstract class is composed of the following methods:

| Methods  | Use |
| :-       | :-  |
| `detect` | TBD |
| `switch` | TBD |

The parameters associated with the branch detectors are the following :

| Parameter | Use                                                                    | Default        |
| :-        | :-                                                                     | :-             |
| `verbose` | Displays messages when a bifurcation is encountered and gives its type | &check; : True |

### Examples of creating an `ABCBifurcator` and adding it into an input dictionary:

To be created, an `ABCBifurcator` subclass needs its abstract method to be defined :
```python
class FakeBif(ABCBifurcator):  # inherits from abstract class
    factory_keyword = "fakebifurcator"  # mandatory to define
    def predict(
            self, sollist: list[SystemSolution], ds :float, k_imposed=None
    ) -> tuple[np.ndarray,SystemSolution,float]:
        # code the way to get the predicted point,
        # the last SystemSolution point that is linked to the prediction
        # and the direction of the prediction that has been used depending if the angular frequency increases or decreases
        return xpred, lstpt, self.sign_ds

INP = {
    "analysis":{
        "FRF":{
            "study":"frf",
            ...,
            "predictor": "fakebifurcator",  # call the criterion using factory_keyword.
            "predictor_options":{"verbose": False},
            ...,
        },
        ...,
    },
    "plugin":[FakeBifurcator], # add to the factory using plugin
    ...,
}
```

## FactoryBifurcator

This file contains the dictionary of all the predictors that are available as
well as the function `generateBifurcator` that creates the predictor object.

## HillBifurcator `hill`

Uses the Hill's method combined with bordering techniques, to derived tests
functions that can detect three types of simple bifurcation:
- fold bifurcation,
- branching point bifurcation, and
- Neimark-Sacker bifurcation.

This bifurcation detector is mainly a python implementation of [[1]](#1).


### References

TODO: cite Krach, Detroux, maybe Lazarus
<a id="1">[1]</a> E. Allgower and K. Georg, *Numerical Continuation Methods -- An Introduction. Soc. Ind Appl Math. 2003.
