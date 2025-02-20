# Elements package 
An Element in pyHarm is defined as any object that has a contribution to the residual equation and its jacobian. Any element inherits from the `ABCElement` class. 

## ABCElement
The `ABCElement` class is the abstract class ruling the definition of an element in pyHarm. It is implemented with generic methods that are useful for all the other elements. The Element contains a set of flags to allow for easier discrimination as well as the following methods : 
| Methods | Use |
| :- | :- |
|`__str__`| Dunder method for print representation |
|`__init_data__`| Treats the input data dictionary and extracts main characteristics of the element |
|`generateIndices`| From a DataFrame containing the localisation of each dof of the system, returns the dofs of interest for this specific element, it shall also provide transformation matrices that are used for selecting master/slave dofs as well as the local directions |
|`adim`| **Abstract method** to adimensionalise the characteristics of the element |
|`evalResidual`| **Abstract method** to evaluate the residual contribution of the element |
|`evalJacobian`| **Abstract method** to evaluate the jacobian contribution of the element |

### Examples of creating an `ABCElement` and adding it into an input dictionary: 

The `ABCElement` class is not used directly for creating new elements. Instead, intermediate classes are built, implementing some of the abstract methods while remaining abstract class with no possibility of instantiation. Those intermediate classes are here to discriminate type of elements and are provided in subpackages of the Element package depending on their nature.
```python 
class SubToNodeElement(ABCElement): # inherits from abstract class
    """ 
    Abstract class that connects a whole substructure to a single node
    """

    def __init_data__(self, name, data, CS):
        """ 
        This method is used to interpret data dictionary
        """
        self.sub = list(data["connect"].keys())[0]
        self.sub_of_node = list(data["connect"].keys())[1]
        self.node_of_node = list(data["connect"].values())[1]
        self.indices = []
        self.name = name
        self.CS = CS
        self.data = data

    def __str__(self):
        return "Element of type {self.__class__.__name__} that links :\n - {self.sub} substructure {} \n to\n - {self.node_of_node} node of {self.sub_of_node} substructure"

    def generateIndices(self, expl_dofs: pd.DataFrame):
        """From the explicit dof DataFrame, generates the index of dofs concerned by the kinematic condition.
        
        Args:
            expl_dofs (pd.DataFrame): explicit dof DataFrame from the studied system.

        Attributes:
            indices (np.ndarray): index of the dofs that the kinematic conditions need.
            Pdir (np.ndarray): a slice of first dimension is a transformation matrix to a direction in local coordinate system.
            Pslave (np.ndarray): selection array that selects the slave dofs of the kinematic condition.
            Pmaster (np.ndarray): selection array that selects the master dofs of the kinematic condition.
        """
        # discriminate subs from the explicit dof list :
        sub_expl_list = expl_dofs[(
            (expl_dofs['sub']==) or 
            ((expl_dofs['sub']==self.sub_of_node) and (expl_dofs['node_num']==self.node_of_node) )
            )].reset_index(drop=True)
        self.indices = np.sort(sub_expl_list.index)
        self._generateMatrices(sub_expl_list)

    def _generateMatrices(self,sub_expl_dofs): 
        """
        Generates the Pslave, Pmaster matrices for selecting the slave or master dofs as well as Pdir matrix for selecting the intended directions
        """
        self.Pslave,self.Pmaster = ConstructorPslavemaster(self.sub, self.sub_of_node, self.node_of_node, sub_expl_dofs)
        ...

    

def ConstructorPslavemaster(nsub, subs, nodes, sub_expl_dofs):
    ...

```


## FactoryElements
The `FactoryElements` is the factory for creating Elements in pyHarm. It is composed of a dictionary containing references to all available and instantiable elements in pyHarm with key being the `factory_keyword` property defined in each instantiable element, and the reference to the class as value. The generation of the `ABCElement` subclass is made through the `generateElement` function.






















