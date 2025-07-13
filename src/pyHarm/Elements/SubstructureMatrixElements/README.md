# SubstructureMatrixElement package
The `SubstructureMatrixElement` subpackage gives implementations of `ABCElement` that allows to
input a whole matrix based residual contribution. It affects a whole substructure.

## MatrixElement
The `MatrixElement` class is the abstract class ruling the definition of an element affecting a
whole substructure through a given input matrix. This abstract class inherits from `ABCElement` but
implements some of the abstract methods that are responsible for creating the indices of interest as
well as reading properly the input dictionary. Some abstract methods remain to be defined in
instantiable classes.
| Methods             | Use                                                                          |
| :-                  | :-                                                                           |
| `adim`              | **Abstract method** for adimensionalising the characteristics of the element |
| `evalResidual`      | **Abstract method** for evaluating the residual contribution of the element  |
| `evalJacobian`      | **Abstract method** for evaluating the jacobian contribution of the element  |
| `_generateMatrices` | **Abstract method** builds the necessary matrices from the input data        |

### Examples of creating an `MatrixElement` and adding it into an input dictionary:

```python
class GyroMatrix(MatrixElement): # inherits from abstract class
    """
    Gyroscopic matrix
    """
    factory_keyword = 'gyromatrix'

    def _generateMatrices(self, data):
        self.kronG = np.kron(np.linalg.matrix_power(self.nabla,1),data["matrix"]["G"])

    def adim(sel,lc,wc):
        self.kronG = self.kronG * lc * (wc**2)
        self.flag_adim = True

    def evalResidual(self,xg,om):
        x = xg[self.indices]
        self.R = (om**2 * self.kronG) @ x
        return self.R

    def evalJacobian(self,xg,om):
        x = xg[self.indices]
        self.J = om**2 * self.kronG
        self.dJdom = (2 * om * self.kronG) @ x
        return self.J, self.dJdom

GyroMatrixData = np.array([...])

INP = {
    ...,
    'substructure':{
        'sub':{...},
        ...,
    },
    'connectors':{
        'gyro':{
            'type':'gyromatrix',
            'connect':'sub',
            'matrix':{'G':GyroMatrixData}
        },
        ...,
    },
    ...,
    'plugin':[GyroMatrix]
}
```


## GeneralOrderMatrixElement `GOMatrix`
The `GeneralOrderMatrixElement` is a general linear matrix element application which takes as input
information about the time derivative through `dto` and the power of angular frequency to be applied
through `dom`.

| Parameter | Use                                                 | Default         |
| :-        | :-                                                  | :-              |
| `matrix`  | Contains the matrix to be applied                   | &cross;         |
| `dto`     | Order of the derivative in time                     | &cross;         |
| `dom`     | Order of the power applied to the angular frequency | &check; : `dto` |

## LinearHystMatrixElement `linear_hysteretic`
The `LinearHystMatrixElement` is derived from `GeneralOrderMatrixElement` imposing `dto` to 1 and
`dom` to 0. It allows taking into account linearized hysteretic damping effects.

| Parameter | Use                               | Default |
| :-        | :-                                | :-      |
| `matrix`  | contains the matrix to be applied | &cross; |

## Substructure `substructure`
The `Substructure` class is a `MatrixElement` that creates residual contribution coming from mass,
linear damping, gyroscopic and rigidity matrices.

| Parameter | Use                                                      | Default |
| :-        | :-                                                       | :-      |
| `matrix`  | contains the 4 matrices under the keys (`M`,`C`,`G`,`K`) | &cross; |

It allows implementing useful methods in the form of `get_Mass` and `get_Rigidity` in order to
extract initial mass and rigidity matrices.
