# Module Stability

This module contains all the stability analyses that are provided by
pyHarm. The module is organized around an abstract class **ABCStability** and a
**FactoryStability** that is in charge of creating the objects. All stability
analyses objects must comply with the **ABCStability** abstract class. The
section below presents the different classes that are available in this module.

## ABCStability

The **ABCStability** class is an abstract class defining the essential
components of any stability analysis.
The interface of this abstract class is composed of the following methods:

| Methods    | Use |
| :-         | :-  |
| `detect`   | TBD |
| `localize` | TBD |
| `track`    | TBD |

The parameters associated with the branch detectors are the following :

| Parameter | Use                                                                    | Default        |
| :-        | :-                                                                     | :-             |
| `verbose` | Displays messages when a bifurcation is encountered and gives its type | &check; : True |

### Examples of creating an `ABCStability` and adding it into an input dictionary:

To be created, an `ABCStability` subclass needs its abstract method to be defined :
```python
class FakeStab(ABCStability):  # inherits from abstract class
    factory_keyword = "fakestability"  # mandatory to define

    def predict():
        # your implementation

    def localize():
        # your implementation

    def track():
        # your implementation


INP = {
    "analysis":{
        "FRF":{
            "study":"frf",
            ...,
            "stability": "fakestability",  # call the criterion using factory_keyword.
            "stability_options":{"verbose": True},
            ...,
        },
        ...,
    },
    "plugin":[FakeStability], # add to the factory using plugin
    ...,
}
```

## FactoryStability

This file contains the dictionary of all the bifurcators that are available, as well as the function
`generateStability` that creates the bifurcator object.

## StabilityHill `hill`

Uses the Hill's method to assess the stability of the computed solutions.

### References

