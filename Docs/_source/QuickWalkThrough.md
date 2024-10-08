## Quick Walkthrough of Packages and Modules

In this section, the main idea behind every main package is given in order to give insights about how the packages interacts with each other. 

**Substructures package** : This package main responsability is to create new degrees of freedom to introduce into the problem. 

**Elements package** : This package responsability is to compute elementary residual contribution coming from linear or nonlinear forces attached to degrees of freedom. 

**KinematicConditions package** : This package responsability is to compute the displacement of degrees of freedom that are linked through a kinematic condition, and project the possible residual contribution of the slave dofs onto the master dofs of the kinematic conditions. 

**Systems package** : This package responsability is to make the assembly of all the residual contributions and kinematic conditions into a large full system matrix. It first expands the input displacement vector in order to take the kinematic conditions into account before computing all the elementary residuals and assembling them. A final step of projecting the kinematically constrained dofs residual and matrix reduction is made.

**Reductors package** : This package responsabiltiy is to introduce reduction mechanisms onto the output residual vector and jacobian matrix coming from the *System* before entering the solving process. 

**NonLinearSolver package** : This package responsability is the solving of the *System* calling  residual and jacobian matrix evaluations.

**Analysis package** : This package responsability is to automatize the solving of the nonlinear system for multiple *System* states until a stop criterion is met. For instance,in a forced response frequency analysis, the *System* is being solved for multiple values of input frequency in order to obtain the full forced response curve.

**Predictors/Correctors/StepSizeRules package** : Those package are added packages for specific `FRF_NonLinear` analysis.

**Maestro module** : This module reponsability is the reading of input dictionary, and running the analysis onto the described system. This is the main interface between `pyHarm` and the user.

