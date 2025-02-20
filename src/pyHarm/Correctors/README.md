# Module Corrector

This module contains all the correctors type that are provided by **pyHarm**. The module is organized around an abstract class **ABCCorrector** and a **FactoryCorrector** that is in charge of creating the objects. All analysis object must comply with the **ABCCorrector** abstract class. The section below presents the different classes that are available in this module.

## ABCAnalysis

The **ABCCorrector** class is an abstract class defining the essential components of any corrector. Two abstract methods are defined :

| Methods | Use |
| :- | :- |
|`ClosureEquation`| *Abstract method* : Returns the residual value of the closure equation based on displacement vector and a **SystemSolution** object |
|`ClosureJacobian`| *Abstract method* : Returns the jacobian values with respect to displacement and the angular frequency of the closure equation based on displacement vector and a **SystemSolution** object |

### Examples of creating an `ABCCorrector` and adding it into an input dictionary: 

To be created, an `ABCCorrector` subclass needs its abstract method to be defined : 
```python 
class FakeCorrector(ABCCorrector): # inherits from abstract class
    factory_keyword="fakecorr" # mandatory to define
    def ClosureEquation(self, solx:np.ndarray,sol:SystemSolution,sollist:list[SystemSolution],**kwargs) -> np.ndarray:
        R_cont =  ... # compute the residual equation of the correction equation.
        return R_cont
    def ClosureJacobian(self, solx:np.ndarray,sol:SystemSolution,sollist:list[SystemSolution],**kwargs) -> tuple[np.ndarray,np.ndarray]:
        dRdx =  ... # compute the jacobian with respect to the displacement of the correction equation.
        dRdom =  ... # compute the jacobian with respect to the angular frequency of the correction equation.
        return dRdx, dRdom

INP = {
    "analysis":{
        "FRF":{
            "study":"frf",
            ...,
            "corrector":"fakecorr", # call the criterion using factory_keyword.
            ...,
        },
        ...,
    },
    "plugin":[FakeCorrector], # add to the factory using plugin
    ...,
}
```

## FactoryCorrector

This file contains the dictionary of all the correctors that are available as well as the function `generateCorrector` that creates the corrector object based on the desired type of corrector.

## Corrector_no_continuation `nocontinuation`

This corrector corresponds to a no-correction closure equation. The residual is expressed as follows : 

 $$R(x,\omega) = \omega - \omega_{start}$$

 which forces the angular frequency to remain constant.

 ## Corrector pseudo arc length `pseudo_arc_length`

 This corrector forces the seeked solution to belong to an hyperplan perpendicular to the prediction direction. This corrector allows for treating turning points. The residual is expressed as follows : 

 $$R(x,\omega) = <(x_{start} - x_{previous});(x-x_{start})>$$

 where $<;>$ defines the classical scalar product.

 ## Corrector arc length `arc_length`

 This corrector forces the seeked solution to belong to an hypersphere of radius $`ds`$, where $`ds`$ is the length of the prediction vector. This corrector also allows for treating turning points. The residual is expressed as follows : 

 $${\displaystyle R(x,\omega) = (x - x_{previous})^{2} + (\omega - \omega_{previous})^{2} + ds^{2}}$$


