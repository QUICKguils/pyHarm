# DLFTElements subpackage

This subpackage implements a formulation of element that differs from the classical elements using DLFT [[1]](#1).

## DLFTElement 

The DLFT method is a penalisation method for handling constraints. The **DLFTElement** is a modification of the **NodeToNodeElement** that complies with DLFT method peculiarities (thus remaining an abstract class). For a better understanding of DLFT method and its use see. All the **NodeToNodeElement** that uses DLFT must inherit from **DLFTElement** implementation as their residual value depends on the residual of the linear part of the system.

Redefined methods for complying with DLFT peculiarities are the following : 
- `_evalJaco_DF` : for taking into account that DLFT needs the linear residual value in order to be computed
- `__flag_update__` : to set `flag_DLFT` to True

## DLFTUniGap `DLFTUniGap`

This **DLFTElement** is the implementation of **PenaltyUnilateralGap** element using DLFT method. More details can be viewed in [[1]](#1) if needed.

| Parameter | Use | Default |
| :- | :- | :- |
|`eps`| Value of the penalisation coefficient [float] | &cross; |
|`g`| Size of the gap [float] | &cross; |
|`N0`| Normal force to the contact without forcing [float] | &cross; |

## DLFTBilateralGap ``

This **DLFTElement** is the implementation of **PenaltyBilateralGap** element using DLFT method. More details can be viewed in [[1]](#1) if needed.

| Parameter | Use | Default |
| :- | :- | :- |
|`eps`| Value of the penalisation coefficient [float] | &cross; |
|`g`| Size of the gap [float] | &cross; |


## DLFTFriction `DLFTFriction`

This **DLFTElement** is the implementation of **Jenkins** element using DLFT method. More details can be viewed in [[1]](#1) if needed.

| Parameter | Use | Default |
| :- | :- | :- |
|`eps`| Value of the penalisation coefficient [float] | &cross; |
|`mu`| friction coefficient [float] | &cross; |
|`N0`| Normal force to the contact at initial state [float] | &cross; |


## DLFT3D

This **DLFTElement** is the implementation of **Penalty3D** element using DLFT method. It is an assembly of **DLFTUniGap** and **DLFTFriction** residual and jacobian. More details can be viewed in [[1]](#1) if needed.

| Parameter | Use | Default |
| :- | :- | :- |
|`eps`| Value of the penalisation coefficient [float] | &cross; |
|`g`| Size of the gap [float] | &cross; |
|`mu`| friction coefficient [float] | &cross; |
|`N0`| Normal force to the contact at initial state [float] | &cross; |


### References

<a id="1">[1]</a> S. Nacivet, *Modélisation du frottement en pied d'aube par une approche fréquentielle*. PhD manuscript École Centrale de Lyon, 2002.



