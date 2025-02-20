# StopCriterion Package

This module contains all the stop criterion types that are provided by pyHarm. The module is organized around an abstract class **ABCStopCriterion** and a **FactoryStopCriterion** that is in charge of creating the objects. All stop criterion object must comply with the **ABCStopCriterion** abstract class. The section below presents the different classes that are available in this module.

## ABCStopCriterion

The **ABCStopCriterion** class is an abstract class defining the essential components of any stop criterion. One abstract method is defined :

| Methods | Use |
| :- | :- |
|`getStopCriterionStatus`| *Abstract method* : Returns a boolean that tells if the stop criterion is reached |

### Examples of creating an `ABCStopCriterion` and adding it into an input dictionary: 

To be created, an `ABCStopCriterion` subclass needs its abstract method to be defined : 
```python 
class FakeStopper(ABCStopCriterion): # inherits from abstract class
    factory_keyword="fakestopper" # mandatory to define
    def getStopCriterionStatus(self,sol:SystemSolution,sollist:list[SystemSolution],**kwargs) -> bool:
        if ... : 
            # Codes when to return True when condition is reached
            return True
        else : 
            return False

INP = {
    "analysis":{
        "FRF":{
            "study":"frf",
            ...,
            "stopper":"fakestopper", # calls the criterion using factory_keyword.
            ...,
        },
        ...,
    },
    "plugin":[FakeStop], # adds to the factory using plugin
    ...,
}
```

## FactoryStopCriterion

This file contains the dictionary of all the stop criteria that are available as well as the function `generateStopCriterion` that creates the criterion object based on the selected keyword.

## StopCriterionBounds `bounds`

The criterion gets activated if the bounds of angular frequency are reached (either lower or upper).

## StopCriterionBoundsOrSolNumber `solnumber`

The criterion gets activated either if the bounds of angular frequency are reached, or if the provided maximum number of solutions is reached.

| Parameter | Use | Default |
| :- | :- | :- |
|`max_solutions`| Number max of solutions before stopping the process | &check; : 100 |



