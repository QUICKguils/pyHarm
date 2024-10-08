## Design pattern: Factory Design Pattern

The pyHarm codebase employs the **Factory Design Pattern**, which can be observed throughout the subpackages. This design pattern is chosen to facilitate the distribution of responsibilities among classes, resulting in reduced dependency and tighter code coupling. Although it may not always be the most efficient pattern for every scenario, it is consistently implemented in the codebase to enhance comprehension of the underlying mechanisms in pyHarm. 

The Factory Design Pattern relies on abstract classes to define core interfaces for essential classes, denoted as `ABC***` within each subpackage. By utilizing abstraction, the coupling is minimized, ensuring that any class dependent on an abstract-based class can function properly as long as interactions are limited to the defined abstract methods. Moreover, the pattern involves a `Factory_***` responsible for creating instances of the `ABC***` objects. This factory consists of a collection of available `ABC***` subclasses, enabling the call of these classes. Through a factory function or class, based on the provided input, the desired object is instantiated and returned. 

For more comprehensive information, please refer to the [Refactoring Guru website](https://refactoring.guru/design-patterns/factory-method).

### Minimal example : 

```python
# Defines the abstract class : 
from abc import ABC, abstractmethod

class ABCThing(ABC): # the abstract class defining the interfaces of the class
    @property
    @abstractmethod
    def factory_keyword(self)->str:
        # some abstract property to be defined in instantiable objects
        ...
    @abstractmethod    
    def make_something(self):
        # some abstract method to be defined in instantiable objects
        ...

class Stuff(ABCThing):
    factory_keyword = 'stuff' # abstract property is defined
    def make_something(self):# abstract method is defined
        print('I do stuff')

class Things(ABCThing):
    factory_keyword = 'things'
    def make_something(self):
        print('I do things')

ABCThing_dico = { # some dictionnary to pick the right object to instantiate from
    'things':Things,
    'stuff':Stuff
}

def factory_ABCThing(type_abcthing:str) -> ABCThing:
    # some function to instantiate the objects
    Instance = ABCThing_dico[type_abcthing]()
    return Instance
```
## Plugin Design

Throughout the pyHarm tutorials, you will encounter an implemented **plugin system**. This system seamlessly integrates with the factories present in each pyHarm subpackage by enabling the addition of new classes. The plugin system offers a convenient approach, empowering developers to code a new class within pyHarm and effortlessly inject it into the appropriate factory. Consequently, pyHarm can immediately utilize this new class as if it were already part of the existing sources. 

Notably, the `Maestro` objects facilitate this process by incorporating any object declared within the input dictionary under the `plugin` key. This functionality significantly enhances the extensibility and adaptability of pyHarm, allowing for efficient integration of custom classes.

### Minimal example in pyHarm : 

```python
from pyHarm.BaseUtilFuncs import pyHarm_plugin
from pyHarm.Maestrro import Maestro
from pyHarm.StopCriterion.ABCStopCriterion import ABCStopCriterion

#################### Some new class :
class NotSmartStopper(ABCStopCriterion):
    factory_keyword = 'notsmart'
    def getStopCriterionStatus(self,sol:SystemSolution,sollist:list,**kwargs) -> bool:
        if len(sollist)>3 : 
            return True
        else : return False

#################### When you want to add the new class more manually : 
pyHarm_plugin(NotSmartStopper)

#################### When you work with the Maestro and want to use the new class : 
INP = {
    'analysis':{
        'FRF':{
            'study':'frf',
            ...,
            'stopper':'notsmart' # not initially known factory_keyword
        }
    }
    ...,
    'plugin':[NotSmartStopper] # add the new class in the right factory based on its ABC dependency
}
M = Maestro(INP)
M.operate()
```

## Advice for problem data input in `pyHarm`

### Advice for environment & problem building
Although `pyHarm` to this day does not have specific version dependencies, we advise that `pyHarm` is being used in a specific python environment in order to avoid any dependency problem with other libraries.

`pyHarm` can be used through classical python scripts, but we strongly advise to build the problem input under a **jupyter notebook** for greater interactivity and easier access to deaper useful objects. 

### Some feature to abuse

While trying to treat a problem, we strongly advise to start the process by buidling the system to be solved in the jupyter notebook and verify all the necessary degrees of freedom are taken into account. The dataFrame describing the degrees of freedom in the system is a great way of seeing if the kinematic conditions are applied on the right dofs and if all the necessary dofs are present. The dataFrame can be called using `ABCSystem.expl_dofs` which returns the dataFrame containing an explicit description of the dofs. 

```python
import pyHarm
input_dict = {
    "substructures":{
        "sub1":{
            ...
        },
        ...
    },
    "connectors":{
        ...
    },
    "kinematics":{
        ...
    }
}
M = pyHarm.Maestro(input_dict)
sys = M.system
sys.expl_dofs
```

The lists of `ABCElements` and `ABCKinematics` can also be verified in order to see if they match with the provided input file. You can have access to those lists through `ABCSystem.LE` (resp. `ABCSystem.LC`) for the list of elements (resp. kinematic conditions)