# Module NonLinearSolver

This module contains all the nonlinear solvers type that are provided by
**pyHarm**. The module is organized around an abstract class **ABCNLSolver** and
a **FactoryNonLinearSolver** that is in charge of creating the objects. All
analysis object must comply with the **ABCNLSolver** abstract class. The section
below presents the different classes that are available in this module.

## ABNLSolver

The **ABCNLSolver** class is an abstract class defining the essential components
of any nonlinear solver. One abstract method is defined :

| Methods | Use                                                                                                                                                           |
| :-      | :-                                                                                                                                                            |
| `Solve` | *Abstract method* : From a provided **SystemSolution**, returns the solution obtained with the nonlinear solver by completing the provided **SystemSolution** |


### Examples of creating an `ABNLSolver` and adding it into an input dictionary:

To be created, an `ABNLSolver` subclass needs its abstract method to be defined :
```python
class FakeSolver(ABNLSolver): # inherits from abstract class
    factory_keyword="fakesolver" # madatory to define
    def Solve(self,sol:SystemSolution,SolList:list[SystemSolution]) -> SystemSolution:
        Solution = ... # routine to be completed such that the system is solved and a completed SystemSolution is returned
        return Solution

dict_solver_options = {...}

INP = {
    "analysis":{
        "FRF":{
            "study":"frf",
            ...,
            "solver":"fakesolver", # call the criterion using factory_keyword.
            "solver_options":dict_solver_options,
            ...,
        },
        ...,
    },
    "plugin":[FakeSolver], # add to the factory using plugin
    ...,
}
```

## FactoryNonLinearSolver

This file contains the dictionary of all the nonlinear solvers that are
available as well as the function `generateNonLinearSolver` that creates the
nonlinear solver object based on the type of solver desired.

## SolverScipyRoot `scipyroot`

This solver is a wrapping of the `scipy.optimize.root` function. The parameters
are the following :

| Parameter             | Use                                                                                           | Default                                             |
| :-                    | :-                                                                                            | :-                                                  |
| `end_status_accepted` | List of status that are considered to be accepted after exiting the root function [list[int]] | &check; : [1]                                       |
| `residual_tolerance`  | tolerance on the residual if status is not 1 and in the accepted list [float]                 | &check; : 1e-4                                      |
| `root`                | Dictionary of options given to root (see `scipy.optimize.root` options) [dict]                | &check; : {"method":"hybr","options":{"diag":None}} |

 ## SolverNewtonRaphson `NewtonRaphson`

 This solver is an implementation of a basic Newton Raphson procedure for
 solving the nonlinear system iteratively using linear solving techniques. The
 parameters are the following :

| Parameter      | Use                                                                                               | Default        |
| :-             | :-                                                                                                | :-             |
| `tol_residual` | Tolerance on the norm of the residual value for considering the point to be solved [float]        | &check; : 1e-8 |
| `tol_delta_x`  | Tolerance over the change in displacement solution for considering the point to be solved [float] | &check; : 1e-8 |
| `max_iter`     | Max number of iteration before exiting [int]                                                      | &check; : 50   |

 ## SolverMoorePenrose `MoorePenrose`

This solver is an implementation of the MoorePenrose iterative solver. It is an
iterative solver that seeks for a solution perpendicularly to the direction of
the previous iteration. It is equivalent to the Newton Raphson solver with
pseudo arc length corrector at the first iteration, but the direction of search
is updated at each iteration. See for more details [[1]](#1). The parameters are
the following :

| Parameter      | Use                                                                                               | Default        |
| :-             | :-                                                                                                | :-             |
| `tol_residual` | Tolerance on the norm of the residual value for considering the point to be solved [float]        | &check; : 1e-8 |
| `tol_delta_x`  | Tolerance over the change in displacement solution for considering the point to be solved [float] | &check; : 1e-8 |
| `max_iter`     | Max number of iteration before exiting [int]                                                      | &check; : 50   |

### References

<a id="1">[1]</a> E. Allgower and K. Georg, *Numerical Continuation Methods -- An Introduction. Soc. Ind Appl Math. 2003.
