# Module Reductors

This module contains all the reductors types that are provided by pyHarm. The module is organized around an abstract class **ABCReductor** and a **FactoryReductor** that is in charge of creating the objects. All reductor objects must comply with the **ABCReductor** abstract class. The section below presents the different classes that are available in this module.

## ABCReductor

The **ABCReductor** class is an abstract class defining the essential components of any reductor used in pyHarm. Four abstract methods are defined :

| Methods | Use |
| :- | :- |
|`update_reductor`| *Abstract method* : From the predicted point, the Jacoban computed for this point, the DataFrame of explicit dofs list and the access to the **ABCSystem** object, updates the reductor by modifying expand, reduce_vector and reduce_matrix methods |
|`expand`| *Abstract method* :  From a reduced displacement dof, expands it to the full size |
|`reduce_vector`| *Abstract method* : From an original displacement or residual vector, reduces it to its reduced size |
|`reduce_matrix`| *Abstract method* : From an original jacobian, reduces it to its reduced size |

### Examples of creating an `ABCReductor` and adding it into an input dictionary: 

To be created, an `ABCReductor` subclass needs its abstract methods to be defined : 
```python 
class FakeReducer(ABCReductor): # inherits from abstract class
    factory_keyword="fakereducer" # mandatory to define
    def update_reductor(self, xpred, J_f, expl_dofs, *args) :
        # Codes how to update the reductor using the predicted point and its jacobian information
        return self.reduce_vector(xpred), self.reduce_matrix(J_f), self.output_expl_dofs
    def expand(self,q:np.ndarray) -> np.ndarray:
        x = ... # expands the reduced size vector to full size
        return x
    def reduce_vector(self,R:np.ndarray) -> np.ndarray:
        Rred = ... # reduces the residual vector size according to the reducing method
        return Rred
    def reduce_matrix(self,dJdxom:np.ndarray,*args) -> np.ndarray:
        dJdxom_red = ... # reduces the full jacobian according to the reducing method
        return dJdxom_red

INP = {
    "analysis":{
        "FRF":{
            "study":"frf",
            ...,
            # insert definition of the reductor in the chain of reducers to be used.
            "reductors":[...,{"type":"fakereducer",...},...], 
            ...,
        },
        ...,
    },
    "plugin":[FakeReducer], # add to the factory using plugin
    ...,
}
```


## FactoryReductor

This file contains the dictionary of all the reductors that are available as well as the function `generateReductor` that creates the reductor object.

## ChainReductor

The **ChainReductor** object is a reductor composed of a list of single reductors. It chains the operations of `update_reductor`, `expand`, `reduce_vector`, `reduce_matrix` within the list of reductors. This is the reductor that is generated in **FRF_NonLinear** analysis allowing for more than a single reductor to be used simulteanously. Some reductor might have an order imposed in the list of chained reductors. This precision is given in the presentation of each reductor.

## FactoryChainReductor

This file contains the function `generateChainReductor` that creates the **ChainReductor** object used by the analysis.

## NoReductor `noreductor`

This is the default reductor that does nothing.

## KrackPreconditioner `KrackPreconditioner`

This reductor does not reduce the size of the system to be solved but consists of a preconditioner. This preconditioner is built such that it creates a scaling matrix in order to set the displacement vector to 1 up to a certain cut-off. For more details see (see [[1]](#1)).

| Parameter | Use | Default |
| :- | :- | :- |
|`cut_off`| Displacement cut-off for which a scaling factor is imposed if the displacement is larger than the cut-off value [float] | &check; : 1e-7 | 

## AllgowerPreconditioner `AllgowerPreconditioner`

This reductor does not reduce the size of the system to be solved but consists of a preconditioner. It uses the QR decomposition of the Jacobian at the predicted point in order to compute a scaling matrix that preconditions the problem (see [[2]](#2)). 

## GlobalHarmonicReductor `globalHarmonic`

This reductor reduces the size of the system by cutting off harmonics based on a criterion described in [[3]](#3). The reductor only keeps the necessary number of harmonic based on a criterion that analyses the cross harmonic Jacobian block.

| Parameter | Use | Default |
| :- | :- | :- |
|`nh_start`| Number of harmonic taken for initial solution. By default takes all the harmonics of the system [str, np.ndarray] | &check; : "max_nh" | 
|`disp_cut_off`| Displacement cut-off for which the value of the $`\Delta q <`$ `disp_cut_off` [float] | &check; : 1e-9 | 
|`err_admissible`| Admissible error onto the criterion. If threshold is reached, more harmonics are selected [float] | &check; : 5e-2 | 
|`h_always_kept`| Array of harmonic number that shall be always kept within the solver [np.ndarray] | &check; : np.array([0,1])| 
|`tol_update`| Subcriterion that checks variations of the Jacobian. If the Jacobian has not changed from previous iteration, the proper criterion is not computed and the harmonic number is kept as previous selection [float] | &check; : 1e-9| 
|`verbose`| Boolean value for displaying information on the reductor results (harmonic number kept) [bool] | &check; : True| 

## LocalHarmonicReductor `localHarmonic`

This reductor is a subclass of **GlobalHarmonicReductor**. It reduces the size of the system by cutting off harmonics based on a criterion described in [[3]](#3). This reductor is local in the sense that it cuts the harmonic number dof per dof instead of the global selection of the **GlobalHarmonicReductor**. The parameters are the same as the **GlobalHarmonicReductor**.

## NLdofsReductor `NLdofs`

This reductor reduces the size of the system by cutting the linear part from the system to be solved, the linear part being solved at each iteration using a linear specific solver. For more details see [[4]](#4). The proposed methodology differs a bit from [[4]](#4) as in pyHarm the residual and the jacobian are evaluated for the whole system including the closure equation. 

$$\begin{pmatrix}R_{n}\\R_{l}\\R_{\omega}\end{pmatrix} = \begin{bmatrix} Z_{nn} & Z_{nl} & 0 \\ Z_{ln} & Z_{ll} & 0 \\C_{n} & C_{l} & C_{\omega} \\ \end{bmatrix} \cdot \begin{pmatrix} x_{n} \\ x_{l} \\ \omega \end{pmatrix} + \begin{pmatrix} F^{N}_{n} \\ 0 \\ 0 \end{pmatrix} + \begin{pmatrix} F^{E}_{n} \\ F^{E}_{l} \\ 0 \end{pmatrix}$$

This allows to express $x_{l}$ as the following : 

$$x_{l} = Z_{ll}^{-1} \cdot \left ( -Z_{ln} x_{n} - F^{E}_{l} - R_{l} \right )$$

where the equation is solved for $R_{l}=0$ at each expansion before evaluating the residual values $(R_{n}, R_{l}=0, R_{\omega})$ and their partial derivatives with respect to $(x_{n},x_{l},\omega)$ in the form of the Jacobian matrix.

The system can be then reduced by taking the new following residual vector : 
$$\begin{pmatrix} R_{n}^{red} \\ R_{\omega}^{red} \end{pmatrix} = \begin{pmatrix} R_{n} - Z_{nl}Z_{ll}^{-1} R_{l} \\ R_{\omega} - C_{l}Z_{ll}^{-1} R_{l}\end{pmatrix}$$

 where $R_{l}(x_{n},x_{l},\omega)=0$ by construction of $x_{l}$. This modification leads to the following modification of the Jacobian matrix : 

$$\begin{aligned} 
\frac{\partial R_{n}^{red}}{\partial x_{n}} & = \frac{\partial R_{n}}{\partial x_{n}} - \cancel{\frac{\partial Z_{nl}}{\partial x_{n}}Z_{ll}^{-1}R_{l}} - \cancel{Z_{nl}\frac{\partial Z_{ll}^{-1}}{\partial x_{n}}R_{l}} - Z_{nl}Z_{ll}^{-1}\frac{\partial R_{l}}{\partial x_{n}}  \\
\frac{\partial R_{n}^{red}}{\partial \omega} & =  \frac{\partial R_{n}}{\partial \omega} - \cancel{\frac{\partial Z_{nl}}{\partial \omega}Z_{ll}^{-1}R_{l}} - \cancel{Z_{nl}\frac{\partial Z_{ll}^{-1}}{\partial \omega}R_{l}} - Z_{nl}Z_{ll}^{-1}\frac{\partial R_{l}}{\partial \omega} 
\end{aligned}$$

The same developpements hold for the partial derivatives of $R_{\omega}$ using $C_{l} Z_{ll}^{-1}$ instead. The two terms are evaluated using a linear solver, taking advantage of the transposed form. Let $\Lambda$ be either $Z_{nl}$ or $C_{l}$. 

$$\begin{aligned}
T &= (\Lambda Z_{ll}^{-1})^{T} = (Z_{ll}^{-1})^{T} \cdot \Lambda^{T} = (Z_{ll}^{T})^{-1} \cdot \Lambda^{T}\\
\Leftrightarrow Z_{ll}^{T} \cdot T &= \Lambda^{T},
\end{aligned}$$

to be solved with a linear solver before taking the transpose of the given solution.


### References

<a id="1">[1]</a> M. Krack and J. Gross, *Harmonic Balance for Non Linear Vibration Problems*. 2019.

<a id="2">[2]</a> E. Allgower and K. Georg, *Numerical Continuation Methods -- An Introduction*. Soc. Ind Appl Math. 2003.

<a id="3">[3]</a> C. Gastaldi and T. M. Berruti, *A method to solve the efficiency-accuracy trade-off of multi-harmonic balance calculation of structures with friction contacts*. International Journal of Non-Linear Mechanics, 2017, 92, 25-40 

<a id="4">[4]</a> Y. Colaïtis, *Stratégie numérique pour l'analyse qualitative des interactions aube/carter*, PhD manuscript, Polytechnique Montréal, 2021.



