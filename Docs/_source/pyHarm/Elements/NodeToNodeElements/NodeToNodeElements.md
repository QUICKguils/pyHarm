# NodeToNodeElements package 
The `NodeToNodeElements` subpackage gives implementations of `ABCElement` that allows to connect two nodes.

## NodeToNodeElement
The `NodeToNodeElement` class is the abstract class ruling the definition of an element affecting two nodes of the system. This abstract class inherits from `ABCElement` but implements some of the abstract methods that are responsible for creating the indices of interest as well as reading properly the input dictionary. Some abstract methods remain to be defined in instantiable classes.
| Methods | Use |
| :- | :- |
|`adim`| **Abstract method** for adimensionalising the characteristics of the element |
|`evalResidual`| **Abstract method** for evaluating the residual contribution of the element |
|`evalJacobian`| **Abstract method** for evaluating the jacobian contribution of the element |

### Examples of creating an `NodeToNodeElement` and adding it into an input dictionary: 

```python 
class HarmSquarreSpring(NodeToNodeElement): # inherits from abstract class
    """ 
    Element connecting two nodes with a square spring directly on the harmonic vector.
    """
    factory_keyword = 'harmsquarrespring'
    default_values = {'k':1.0}
    def __post_init__(self):
        if 'k' not in data : 
            data['k'] = self.default_values['k']
        self.k = data['k']

    def adim(sel,lc,wc):
        self.k *= (lc**2)
        self.flag_adim = True
    
    def evalResidual(self,xg,om):
        x = xg[self.indices]
        self.R = self.k @ x**(2)
        return self.R

    def evalJacobian(self,xg,om):
        x = xg[self.indices]
        self.J = 2 * self.k @ x
        self.dJdom = np.zeros(x.shape)
        return self.J, self.dJdom


INP = {
    ...,
    'substructure':{
        'sub':{...},
        ...,
    },
    'connectors':{
        'HSL':{
            'type':'harmsquarrespring',
            'connect':{'sub':[0],'INTERNAL':[1]},
            'dirs':[0,4],
            'k':50.
        },
        ...,
    },
    ...,
    'plugin':[HarmSquarreSpring]
}
```
## GeneralOrderElement `GOElement`

The general order element is an element of which the residual in time domain can be expressed as : 

$$r(x(t)) = k \cdot \left (\frac{\partial^{to} x}{\partial t^{to}} \right )^{xo}.$$

The needed parameters are the following : 
| Parameter | Use | Default |
| :- | :- | :- |
|`k`| Value of the linear coefficient driving the residual | &cross; |
|`to`| Order of the derivative in time | &cross; |
|`xo`| Order of the power applied| &cross; |

Some specific **GeneralOrderElement** are proposed directly into the library of element because of their recurrent use.

### Linear Spring `LinearSpring`

Is an element inheriting from **GeneralOrderElement** class, that imposes `to`=0 and `xo`=1


### Linear Damper `LinearDamper`

Is an element inheriting from **GeneralOrderElement** class, that imposes `to`=1 and `xo`=1

### Cubic Spring `CubicSpring`

Is an element inheriting from **GeneralOrderElement** class, that imposes `to`=0 and `xo`=3


## GeneralOrderForcing `GOForcing`

The general order forcing is an element used to create forcing that does not depend on the displacement vector. An harmonic loading shall be given in the form of `ho` parameter as well as a phase lag `phi` in order to create a loading unitary vector on the right dofs. The order of derivative `dto` allows for loading with a polynomial dependance in $\omega$, while the parameter `amp` drives the amplitude. The residual can be expressed in the frequency domain by : 

$$r(x(\omega)) = amp \cdot \omega^{dto} \cdot \nabla^{dto} LoadingVector$$

where $\nabla$ is the derivative operator.

| Parameter | Use | Default |
| :- | :- | :- |
|`amp`| Amplitude of the forcing [float] | &cross; |
|`ho`| Order of the harmonic the loading is applied to [int] | &cross; |
|`phi`| Phase lag applied for cosine/sine sharing [float:rad]| &check; : 0 |
|`dto`| Order of the time derivative [int] | &cross; |

Some specific **GeneralOrderForcing** are proposed directly into the library of element because of their recurrent use.

### CosinusForcing `CosinusForcing`

Is an **GeneralOrderForcing** element that imposes `dto`=0, `ho`=1 and `phi`=0.

### SinusForcing `SinusForcing`

Is an **GeneralOrderForcing** element that imposes `dto`=0, `ho`=1 and `phi`=$\frac{\pi}{2}$

## PenaltyUnilateralGap `PenaltyUnilateralGap`

This **NodeToNodeElement** represents a gap that adds a rigidity to the connection when the gap is closed in the form of adding a linear spring. This gap closes on a single side (meaning the sign of the displacement is of importance). The residual of the gap is written in the following form in the time domain : 


$$\forall t \in [0,T], \begin{cases}r(x(t)) = 0, \text{ if } x(t)-g(t)>=0\\ r(x(t)) = k \cdot (x(t)-g(t)), \text{ if } x(t)-g(t)<0 \end{cases}$$

| Parameter | Use | Default |
| :- | :- | :- |
|`k`| Added linear spring when gap is closed [float] | &cross; |
|`g`| Size of the gap [float] | &cross; |

## PenaltyBilateralGap `PenaltyBilateralGap`

This **NodeToNodeElement** represents a gap that adds a rigidity to the connection when the gap is closed in the form of adding a linear spring. This gap closes when the norm of the dispacement is equal to the gap value (which means multiple direction can be regarded and the gap closes on both sides). The residual of the gap is written in the following form in the time domain : 


$$ \forall t \in [0,T], \begin{cases}r(x(t)) = 0, \text{ if } \|x(t)\|-g(t)>=0\\ r(x(t)) = k \cdot \|x(t)\|-g(t), \text{ if } \|x(t)-g(t)\|<0 \end{cases} $$

| Parameter | Use | Default |
| :- | :- | :- |
|`k`| Added linear spring when gap is closed [float] | &cross; |
|`g`| Size of the gap [float] | &cross; |

## PenaltyBilateralGapRegulPoly `PenaltyBilateralGapRegulPoly`

This **NodeToNodeElement** represents a gap that adds a rigidity to the connection when the gap is closed in the form of adding a linear spring. This gap closes when the norm of the dispacement is equal to the gap value (which means multiple direction can be regarded and the gap closes on both sides). A polynomial regularization is introduced to mitigate convergence issues at the contact transition. The recommended range for the regularization coefficient is [1e-5,1e-7]*k. The residual of the gap is written in the following form in the time domain : 


$$\forall t \in [0,T], r(x(t))=\frac{k(\left\| x(t)) \right\|-g)}{2}+\sqrt{\left( \frac{k(\left\| x(t) \right\|-g)}{2} \right)^{2}+\varepsilon^{2}}$$

| Parameter | Use | Default |
| :- | :- | :- |
|`k`| Added linear spring when gap is closed [float] | &cross; |
|`g`| Size of the gap [float] | &cross; |
|`eps`| Polynomial regularization coefficient [float] | &cross; |

## Jenkins `Jenkins`

This **NodeToNodeElement** is an implementation of the classical Jenkins element for modeling dry friction using coulomb laws. As friction is an hysteretic phenomenon, a correction loop is present in the residual computation to take into account the hysteretic loops. More details can be found in [[1]](#1). 

| Parameter | Use | Default |
| :- | :- | :- |
|`k`| Linear spring when element is not sliding [float] | &cross; |
|`mu`| Friction coefficient [float] | &cross; |
|`N0`| Normal force to the contact [float] | &cross; |

## Penalty3D `Penalty3D`

This **NodeToNodeElement** is an implementation of the classical friction contact element that allows contact separation. It is implemented using an assembly of **Jenkins** and **PenaltyUnilateralGap** residual and jacobian functions. 

| Parameter | Use | Default |
| :- | :- | :- |
|`k_n`| Normal spring associated with the **PenaltyUnilateralGap** element [float] | &cross; |
|`g`| Gap value [float] | &cross; |
|`k_t`| Tangent spring value associated with the **Jenkins** element [float] | &cross; |
|`mu`| Friction coefficient [float] | &cross; |
|`N0`| Normal force to the contact at initial state [float] | &cross; |

### References

<a id="1">[1]</a> E. Petrov and D. J. Ewins, *Analytical Formulation of Friction Interface Elements for Analysis of Nonlinear Multi-Harmonic Vibrations of Bladed Discs*. ASME Turbo Expo 2002: Power for Land, Sea, and Air.
