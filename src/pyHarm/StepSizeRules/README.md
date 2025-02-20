# Module StepSizeRules

This module contains all the step size rules type that are provided by **pyHarm**. The module is organized around an abstract class **ABCStepSizeRule** and a **FactoryStepSize** that is in charge of creating the objects. All step size rule object must comply with the **ABCStepSizeRule** abstract class. The section below presents the different classes that are available in this module.

## ABCStepSizeRule

The **ABCStepSizeRule** class is an abstract class defining the essential components of any step size rule. One abstract method is defined :

| Methods | Use |
| :- | :- |
|`ProjectInBounds`| Projects the step size over the bounds |
|`getStepSize`| *Abstract method* : Returns the step size based on the previous step size and the list of **SystemSolution** provided |


### Examples of creating an `ABCStepSizeRule` and adding it into an input dictionary: 

To be created, an `ABCStepSizeRule` subclass needs its abstract method to be defined : 
```python 
class FakeStep(ABCStepSizeRule): # inherits from abstract class
    factory_keyword="fakestep" # mandatory to define
    def getStepSize(self, ds:float, sollist:list[SystemSolution], **kwargs) -> float:
        # Codes when to return True when condition is reached
        return ds

INP = {
    "analysis":{
        "FRF":{
            "study":"frf",
            ...,
            "stepsizer":"fakestep", # calls the criterion using factory_keyword.
            ...,
        },
        ...,
    },
    "plugin":[FakeStep], # adds to the factory using plugin
    ...,
}
```

## FactoryStepSize

This file contains the dictionary of all the step size rules that are available as well as the function `generateStepSizeRule` that creates the **ABCStepSizeRule** object based on the type of corrector desired.

## StepSizeConstant `constant`

The step size remains constant by all means. Be careful to use a stop criterion that is compliant with this step size rule.

 ## StepSizeAcceptance `acceptance`

This rule returns a step size that is divided by two if the previous solution has not been accepted. Or multiply its value per 2 if a number of consecutive successes by the solver has been made (default is 5). Step size is projected over the defined bounds before being returned. 



