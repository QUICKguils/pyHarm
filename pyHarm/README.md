# Presentation of the main package

This is the main package of pyHarm. This package contains classes and functions that are used by pyHarm but that do not necessitate a factory design pattern. This package is composed of four main modules that are described in this page. 

## Maestro

The Maestro is a pyHarm class that is the main interface between the user and pyHarm core components. Its main job is to generate the system that is studied and the different analyses that are required. An instance can be created using an input dictionary that describes the different elements. The input dictionary must describe the problem completly and often contains the following keys : 

| Key | Contains |
| :- | :- |
|`analysis`| Contains keys and values as [dict] that describe all the analysis to run onto the system [dict] |
|`system`| Contains the information for the creation of the **ABCSystem** subclass object [dict] |
|`coordinates`|  Contains the definition of the local coordinates system that shall be created in the form of **CoordinateSystem** class object [dict] |
|`sustructures`| Contains keys and values as [dict] that describe all the **ABCElements** that **are** of type substructures [dict] |
|`connectors`| Contains keys and values as [dict] that describe all the **ABCElements** that **are not** substructures [dict] |
|`plugin`| List that contains pyHarm sub-module objects and inject them in the matching factory [list] |

### Example of input dictionary : 

The Maestro objects takes as an input a dictionary defining the different parts of the problem to be solved. Two keys are mandatory for the Maestro to be built correctly. The first one is the `system` keyword that defines the type of system studied, the number of harmonics, etc. The second mandatory keyword is the `analysis` keyword in which a dictionary must be defined with all the analysis that are required to the `Maestro` using the previously defined system. A minimal input dictionary would look like below : 
```python
INP = {
    "analysis":{
        "FRF":{
            "study":"frf",
            ...
        },
        "FRF2":{
            "study":"frf",
            ...
        }
    },
    "system":{
        "type":"Base",
        ...
    },
    ... # more keyword can be used
}
```


The main method the user will interact with is the following method :  

| Methods | Use |
| :- | :- |
|`operate`| Loops over the list of analyses and runs the `ABCAnalysis.Solve` method for each analysis |

### Example of operating a Maestro instance :

```python
import pyHarm
M = pyHarm.Maestro(INP)
M.operate(*args)
```

## BaseUtilFuncs

This file contains generic but useful functions that are used in many modules of pyHarm. 

| Function | Use |
| :- | :- |
|`getCustomOptionDictionary`| Completes an input dictionary with default values that are given as second argument of the function |
|`getIndexfromExpldofs`| Returns the indexes of a list dofs from the explicit dof DataFrame defined by a list of substructure, node number and direction  |
|`pyHarm_plugin`| Injects the provided class into the factory dictionary depending on the type of given object  |

### Example adding a plugin through the input dictionary :

New classes of already defined abstract classes can be added on the go using the pyHarm plugin system. The plug in of a new class can be defined in the launching script as follows : 
```python
class NewClass1(ABCObject_1): # inherits from ABCObject; being a reference to one of the Abstract classes defined in other pyHarm subpackages.
    self.factory_keyword="new_class_1"
    ...
class NewClass2(ABCObject_2):
    self.factory_keyword="new_class_2"
    ...

INP = {
    ...,
    "plugin":[NewClass1,NewClass2]
}
```
In the input, it is then possible to use those new objects by calling them using their defined `factory_keyword` attribute.

## DynamicOperator

This file contains functions that create the derivative operator as well as the Discrete Fourier Transform (DFT) operators. 

| Function | Use |
| :- | :- |
|`compute_DFT`| Returns a dictionary containing the DFT and DTF operators based on the input number of time stamps and number of harmonics |
|`nabla`| Returns the order 1 derivative operator based on the number of harmonics |

## CoordinateSystem

This file contains the **CoordinateSystem** class that defines local coordinate systems in pyHarm. The **CoordinateSystem** contains the following methods : 

| Methods | Use |
| :- | :- |
|`checkOrthonormal`| Checks if the basis is orthonormal and raises an error if axes are not orthogonal |
|`getTM`| Returns the transformation matrix from the global to the local system of coordinates |

### GlobalCoordinateSystem

Inherits from the **CoordinateSystem** class and corresponds to the unit vector on each direction. An instance is always constructed during the creation of an **ABCSystem**

### Example adding a local coordinate system in the input file :

```python
INP = {
    ...,
    "coordinates":{
        "local_dir":[[1,0,0],[0,0.5,0.5],[0,-0.5,0.5]],
        ...
    }
}
```

## SystemSolution

This file defines a class based on data/status. It is the main object that transits toward all other objects in pyHarm and corresponds to a Solution of the residual equations. It contains residual values, displacements at starting point, solution point and other useful information. **FirstSolution** is the daughter class that is used in the case of an initial solution. In this case the **FistSolution** cannot have a pointer to the last converged **SystemSolution** object.
