# Package Systems

This module contains all the system types that are provided by **pyHarm**. The module is organized
around an abstract class **ABCSystem** and a **FactorySystem** that is in charge of instantiating
the objects. All system objects must comply with the **ABCSystem** abstract class. The section below
presents the different modules that are available.

## ABCSystem

The **ABCSystem** class is an abstract class defining the essential components of any system. Two
abstract methods are mandatory :

| Methods    | Use                                                      |
| :-         | :-                                                       |
| `Residual` | *Abstract method* : Returns residual of all the elements |
| `Jacobian` | *Abstract method* : Returns jacobian of all the elements |


### Examples of creating an `ABCSystem` and adding it into an input dictionary:

To be created, an `ABCSystem` needs at least a `Substructure` but many substructures can be defined
in the input file. Connectors and kinematic conditions can be added to the problem to complete the
system to be solved. The integration in an input file would look like :

```python
class FakeSystem(ABCSystem):
    factory_keyword="fakesystem"
    def Residual(q:np.ndarray)->np.ndarray:
        # Codes how to assemble residuals
        return Residual_vector
    def Jacobian(q:np.ndarray)->tuple(np.ndarray,np.ndarray):
        # Codes how to assemble jacobians
        return dRdx, dRdom

INP = {
    ...,
    "system":{
        "type":FakeSystem.factory_keyword,
        ...,
    },
    "substructures":{
        "sub_1":{ # At least the definition of 1 substructure is mandatory
            ...,
        },
        ...,
    },
    "connectors":{
        "connector_1":{
            ...
        },
        ...
    },
    "kinematics":{
        "kine_1":{
            ...
        },
        ...,
    }
}
```

## FactorySystem

This file contains the dictionary of all the systems that are available as well as the function
`generateSystem` that creates the **ABCSystem** object. The `System_dico` attribute defined in this
module gathers all the **ABCSystem** subclasses that are available for creation.

## System `Base`

Basic system class that implements the computation of `Residual` and `Jacobian` methods by chaining
`_residual` and `_jacobian` methods onto the different lists of elements.

The parameters associated with the system creation are the following :

| Parameter | Use                                                                       | Default                                        |
| :-        | :-                                                                        | :-                                             |
| `nh`      | number of harmonics to be considered [int]                                | &check; : 1                                    |
| `nti`     | number of time instant used in the AFT procedure [int]                    | &check; : 128                                  |
| `adim`    | Parameters used for adimensionalising the system [dict[bool,float,float]] | &check; : {"status":False, "lc":1.0, "wc":1.0} |

### Examples of creating a `Base` system:

```python
INP = {
    ...,
    "system":{
        "type":"Base",
        "nh":3,
        "nti":128,
        "adim":{
            "status":True,
            "lc":3.14,
            "wc":1.618e+02
        }
    },
    "substructures":{
        ...,
    }
    ...,
}
```
