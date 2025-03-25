# Analysis package presentation

This module contains all the analysis type that are provided by **pyHarm**. The module is organized
around an abstract class **ABCAnalysis** and a **Factory** that is in charge of creating the
objects. All analysis object must comply with the **ABCAnalysis** abstract class. The section below
presents the different classes that are available in this module.

## ABCAnalysis

The **ABCAnalysis** class is an abstract class defining the essential components of any analysis.
The initialisation of an instance requires an input dictionary containing the setup of the analysis,
a system of type **ABCSystem** and the number of degrees of freedom to be treated. Three abstract
methods are defined :

| Methods      | Use                                                                                                          |
| :-           | :-                                                                                                           |
| `initialise` | *Abstract method* : Initialises the analysis by treating the first point to solve                            |
| `makeStep`   | *Abstract method* : Method that describes the process of going from a solve point to the next point to solve |
| `Solve`      | *Abstract method* : Method that runs to whole solving process of the analysis                                |


### Examples of creating an `ABCAnalysis` and adding it into an input dictionary:

To be created, an `ABCAnalysis` needs its abstract methods to be defined :
```python
class FakeAnalysis(ABCAnalysis):
    factory_keyword="fakeana"
    def initialise(self, x0=None, **kwargs):
        # this method shall solve the given initial point of the analysis.
        pass
    def makeStep(self,**kwargs) :
        # this method shall make a step from previous solution and solve the new initial guess
        pass
    def Solve(self, x0=None, **kwargs):
        # This method shall run the initialise method and then loop over the makeStep method until a stopping criterion is reached.
        pass

INP = {
    ...,
    "analysis":{
        ...,
        "FakeAna":{
            "study":"fakeana",
            ...,
        },
        ...,
    },
    ...,
}
```

## FactoryNonLinearStudy

This file contains the dictionary of all the analysis that are available as well as the function
`generateNonLinearAnalysis` that creates the analysis objects based on the type of analysis and the
provided input.

## FRF_NonLinear `frf`

The **FRF_NonLinear** object inherits from **ABCAnalysis** and the analysis that performs a forced
response analysis. The process closely follows the advised process described in detail in [[1]](#1)
using a prediction/correction procedure. The method requires the following parameters into its input
dictionary :

| key | Use | Default value |
| :- | :- | :- |
|`solver`| Defines the **ABCNonLinearSolver** object to be created | &check; "scipyroot" |
|`predictor`| Defines the **ABCPredictor** object to be created | &check; "tangent"|
|`corrector`| Defines the **ABCCorrector** object to be created | &check; : "arc_length"|
|`reductors`| Defines the list of **ABCReductor** object to be created | &check; : [{"type":"noreductor"}]|
|`stepsizer`| Defines the **ABCStepSizeRule** object to be created | &check; : "acceptance" |
|`stopper`| Defines the **ABCStopCriterion** object to be created | &check; : "bounds" |
|`sign_ds`| Defines the direction of the path along the response curve : 1 forward, -1 backward| &check; : 1|
|`puls_inf`| Defines the lower bound of the angular frequency range [float:rad/s]| &check; : 0.1 |
|`puls_sup`| Defines the upper bound of the angular frequency range [float:rad/s]| &check; : 1.0 |
|`puls_start`| Defines starting angular frequency of interest [float:rad/s]| &check; : `puls_inf` or `puls_sup` depending on `sign_ds`|
|`ds_min`| Defines the minimal stepsize [float]| &check; : 1e-8|
|`ds_max`| Defines the maximal stepsize [float]| &check; : 1e0|
|`ds0`| Defines the starting stepsize [float]| &check; : 1e0|
|`purge_jacobians`| Defines if the jacobians are purged along the solving process [bool]| &check; : True|
|`verbose`| Defines if information along the solving is displayed [bool]| &check; : True|

## Linear_Analysis `linear_analysis`

The **Linear_Analysis** object inherits from **ABCAnalysis**. It first performs a modal analysis,
followed by a linear frequency response analysis by using mode superposition. The method requires
the following parameters into its input dictionary :

| key | Use | Default value |
| :- | :- | :- |
|`puls_inf`| Defines the lower bound of the angular frequency range [float:rad/s]| &check; : 0.1 |
|`puls_sup`| Defines the upper bound of the angular frequency range [float:rad/s]| &check; : 1.0 |
|`verbose`| Defines if information along the solving is displayed [bool]| &check; : True|
|`damping`| Defines the damping (either Rayleigh or modal) for the whole system [dict] | &check; : 0.1% modal damping |

### References

<a id="1">[1]</a> M. Krack and J. Gross, *Harmonic Balance for Non Linear Vibration Problems*. 2019.


