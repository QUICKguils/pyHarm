# Module KinematicConditions

This module contains all the kinematics conditions type that are provided by
**pyHarm**. The module is organized around an abstract class **ABCKinematic**
and a **FactoryKinematic** that is in charge of creating the objects. All
kinematic conditions object must comply with the **ABCKinematic** abstract
class. The section below presents the different classes that are available in
this module.

## ABCKinematic

The **ABCKinematic** class is an abstract class defining the essential
components of any kinematic conditions. Three abstract methods are defined :

| Methods      | Use                                                                                                                                                                                |
| :-           | :-                                                                                                                                                                                 |
| `complete_x` | *Abstract method* : From the input of displacement vector **x** given, provides a vector of same size that completes the displacement vector using the kinematic conditions        |
| `complete_R` | *Abstract method* : From the input of residual vector **R** given, provides a vector of same size that completes the residual vector using the kinematic conditions                |
| `complete_J` | *Abstract method* : From the input of the jacobians **dRdx** & **dRdom** given, provides jacobians of same size that completes the jacobians vector using the kinematic conditions |

### Examples of creating an `ABCKinematic` and adding it into an input dictionary:

To be created, an `ABCKinematic` subclass needs its abstract methods to be defined :
```python
class FakeKine(ABCKinematic): # inherits from abstract class
    factory_keyword="fakekine" # madatory to define
    def complete_x(self, x) :
        xadd = ... # code that completes x_add array
        return xadd
    def complete_R(self, R, x):
        R_add = ... # code that completes R_add
        return R_add
    def complete_J(self, Jx, Jom, x):
        Jx_add = ... # code that completes Jx_add
        Jom_add = ... # code that completes Jom_add
        return Jx_add, Jom_add


INP = {
    ...,
    "kinematics":{
        "fake_kine_cond":{
            "type":"fakekine",
            "connect":{
                "sub1":[...],
                "sub2":[...]
            },
            "dirs":[...],
            ...,
        }
    },
    ...,
}
```

## FactoryKinematic

This file contains the dictionary of all the kinematics conditions that are
available as well as the function `generateKinematic` that creates the kinematic
condition object based on the type of solver desired.

## General Order Displacement (GODisplacement) `GOdisplacement`

This kinematic condition imposes a displacement on a given dof such that :

$$\displaystyle x =  \frac{amp}{\omega^{dto}} \cdot l$$

where $l$ is the unit loading vector containing the `ho` and `phi` information.

| Parameter | Use                                                      | Default     |
| :-        | :-                                                       | :-          |
| `amp`     | Amplitude of the imposed displacement [float]            | &cross;     |
| `ho`      | Harmonic on which the imposed displacement is made [int] | &check; : 1 |
| `dto`     | Derivative order on the dof [int]                        | &check; : 0 |
| `phi`     | Phase value [dict]                                       | &check; : 0 |

### AccelImposed `AccelImposed`

Subclass of `GODisplacement` with `dto`=2 such that acceleration is imposed to a
given dof.

### SpeedImposed `SpeedImposed`

Subclass of `GODisplacement` with `dto`=1 such that speed is imposed to a given
dof.

### DispImposed `DispImposed`

Subclass of `GODisplacement` with `dto`=0 such that displacement is imposed to a
given dof.

## Base projection `BaseProjection`

This kinematic condition imposes a base projection between a master substructure
and a slave substructure through a transformation matrix. The residual of the
slave substructure is projected back onto the master substructure.

$$\begin{aligned} \displaystyle x_{slave} & =  \phi \cdot x_{master} \\ R_{master} & = \phi^{t} \cdot R_{slave}(x_{slave})\end{aligned}$$


| Parameter | Use                                                     | Default |
| :-        | :-                                                      | :-      |
| `phi`     | transformation matrix from master to slave [np.ndarray] | &cross; |

### References

<a id="1">[1]</a> E. Allgower and K. Georg, *Numerical Continuation Methods -- An Introduction. Soc. Ind Appl Math. 2003.



