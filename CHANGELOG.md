# Changelog

Every modification is contained in this file. \
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)..

## [1.0.0] - 2024-10-09
### Added
- Initial release of the project.
- Implementation of core features:
  - Package `Analysis`: Contains analysis framework available in pyHarm. Main feature is `FRF` analysis that compute a forced response curve using continuation procedure
  - Package `Correctors`: Contains correction framework available in pyHarm for continuation procedure. Main correctors are the `arc_length`, `pseudo_arc_length` and `nocontinuation` correctors.
  - Package `Elements`: Contains all the contributors to the residual equation framework available in pyHarm. Main features are `NodeToNodeElemements` connecting two nodes together, as well as multiple precoded linear and nonlinear elements including physical behavior such as friction, unilateral and bilateral contact, linear matrix (mass, rigidity, etc ...)
  - Package `KinematicConditions`: Contains the kinematic condition framework available in pyHarm. Main feature are imposing a displacement or any derivative to a specific node, or projecting onto a chosen base. 
  - Package `NonLinearSolver`: Contains the solver framework available in pyHarm. Main feature is `scipyroot`, wrapping root function from scipy to the imposed framework. All classes contained in this package are responsible of solving the nonlinear algebric equation. 
  - Package `Predictors`: Contains the predictors framework available in pyHarm for the continuation process. Main feature is `tangent`, computing the tangent of the jacobian matrix at a solution point in order to provide a prediction direction.
  - Package `Reductors`: Contains the reductors framework available in pyHarm for the solving process. The reductors can be chained and their purpose is to facilitate the solving of the main algebric equations through reduction methods or preconditioning methods. Main feature are `NLdofs` reducing the system to solve using a nonlinear solver only on the nonlinear dofs, `globalHarmonic` reducing the number of harmonics to be solved, `AllgowerPreconditioner` that precondition the system using a scaling matrix based on the Jacobian evaluated at a given point.
  - Package `StopCriterion`: Contains the stop criterion framework available in pyHarm for analysis. Main feature is `bounds`, returning a stop criterion once angular frequency bounds are being reached.
  - Package `Substructures`: Contains the dofs creation framework available in pyHarm. Main feature is `substructure`, building a vector of dofs according to input matrices (mass, rigidity, etc ... ) and creating matrix elements according to the provided matrices.
  - Package `System`: Contains the system framework available in pyHarm. Main feature is `Base`, basic system that is responsible of assembling the residual and jacobian.
  - Module `Maestro`: Contains the main user interface, responsible for launching the analysis onto the provided system. This uses dictionnary as input and build the system to be solved accordingly.
- User documentation
  


### Changed
- No changes in this release.

### Deprecated
- No deprecations in this release.

### Removed
- No removals in this release.

### Fixed
- No fixes in this release.

### Security
- No security changes in this release.
