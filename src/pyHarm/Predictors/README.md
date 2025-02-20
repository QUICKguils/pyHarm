# Module Predictors

This module contains all the predictor types that are provided by pyHarm. The module is organized around an abstract class **ABCPredictor** and a **FactoryPredictor** that is in charge of creating the objects. All predictor object must comply with the **ABCPredictor** abstract class. The section below presents the different classes that are available in this module.

## ABCPredictor

The **ABCPredictor** class is an abstract class defining the essential components of any nonlinear solver. One abstract method is defined :

| Methods | Use |
| :- | :- |
|`bifurcation_detect`| Returns a flag if a bifurcation is encountered. Switches the direction of prediction if a turning point is encountered |
|`getPointerToSolution`| Returns the last accepted point from the **SystemSolution** list |
|`norm_dir`| Normalises the prediction direction depending on the required normalisation policy |
|`predict`| *Abstract method* : Returns the predicted point |

The parameters associated with the predictors are the following : 

| Parameter | Use | Default |
| :- | :- | :- |
|`norm`| Type of normalisation to apply; "norm1" to normalise the prediction vector to one, "om" to normalise the vector such that the angular frequency component is 1 [str] | &check; : "norm1" | 
|`bifurcation_detect`| Uses the detection of bifurcation function [bool] | &check; : True | 
|`verbose`| Displays messages when a bifurcation is encountered and gives its type | &check; : True | 


### Examples of creating an `ABCPredictor` and adding it into an input dictionary: 

To be created, an `ABCPredictor` subclass needs its abstract method to be defined : 
```python 
class FakePred(ABCPredictor): # inherits from abstract class
    factory_keyword="fakepred" # mandatory to define
    def predict(self, sollist:list[SystemSolution], ds:float, k_imposed=None) -> tuple[np.ndarray,SystemSolution,float]:
        # code the way to get the predicted point,
        # the last SystemSolution point that is linked to the prediction
        # and the direction of the prediction that has been used depending if the angular frequency increases or decreases
        return xpred,lstpt,self.sign_ds

INP = {
    "analysis":{
        "FRF":{
            "study":"frf",
            ...,
            "predictor":"fakepred", # call the criterion using factory_keyword.
            "predictor_options":{"norm":"om", "bifurcation_detection":False, "verbose":False},
            ...,
        },
        ...,
    },
    "plugin":[FakePred], # add to the factory using plugin
    ...,
}
```

## FactoryPredictor

This file contains the dictionary of all the predictors that are available as well as the function `generatePredictor` that creates the predictor object.

## PredictorPreviousSolution `previous`

Predicts the next point using the same displacement as the last converged point, and makes a step onto the angular frequency only.

 ## PredictorTangent `tangent`

 Uses the Jacobian at last converged point to compute the tangent of the Jacobian based on a QR decomposition (see [[1]](#1)). 

 ## PredictorSecant `secant`

Inherits from the tangent predictor to make the first point prediction. Otherwise, uses the two last converged points to compute the secant line and predict the next point based on the required step size. 

### References

<a id="1">[1]</a> E. Allgower and K. Georg, *Numerical Continuation Methods -- An Introduction. Soc. Ind Appl Math. 2003.



