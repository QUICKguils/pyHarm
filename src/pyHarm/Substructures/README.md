# Package Substructures

This package contains all the substructures types that are provided by **pyHarm**. The module is organized around an abstract class **ABCSubstructure** and a **FactorySubstructure** that is in charge of instantiating the objects. All system objects must comply with the **ABCSubstructure** abstract class. The main purpose of **ABCSubstructure** based classes is to generate a DataFrame and create the degrees of freedom. Additionaly the **ABCSubstructure** based classes can generate a dictionary that describes connectors or kinematic conditions to add to the **ABCSystem**.

The section below presents the different modules that are available.

## ABCSubstructure

The **ABCSubstructure** class is an abstract class defining the essential components of any substructure. Two abstract methods are mandatory : 

| Methods | Use |
| :- | :- |
|`factory_keyword`| *Abstract property* : keyword used in the factory to call instantiation of a class |
|`_add_connectors`| *Abstract method* : Returns a dictionary containing connectors to add to the problem |
|`_add_kinematics`| *Abstract method* : Returns a dictionary containing kinematic conditions to add to the problem |


### Examples of creating an `ABCSubstructure` and adding it into an input dictionary: 

To be created, an `ABCSubstructure` subclass needs its abstract methods to be defined : 
```python 
class FakeSubstructure(ABCStopCriterion): # inherits from abstract class
    factory_keyword="fakesub" # mandatory to define
    def _add_connectors(self,data:dict):
        con = {
            'connector_001':{
                'type':'LinearSpring',
                'connect':{self.name:[0],'INTERNAL':[1]},
                'dirs':[0,2],
                'k':1e9
            },
            'connector_002':{
                ...
            }
        }
        return con
    def _add_kinematics(self,data:dict):
        kin = {
            'kin_001':{
                'type':'AccelImposed',
                'connect':{self.name:[10]},
                'dirs':[1],
                'amp':.1
            }
        }
        return kin

INP = {
    "substructure":{
        "substructure_001":{
            "type":"fakesub", # calls the reader using factory_keyword.
            ...,
        },
        ...,
    },
    "plugin":[FakeSubstructure], # adds to the factory using plugin
    ...,
}
```

## OnlyDofs `onlydofs`

The `OnlyDofs` class is here to create dofs into the system, but does not create any new connectors or kinematic conditions. The mandatory parameters are `ndofs`, `nnodes`, `nmodes`. If `nmodes` is not provided, it is set to default 0.


| Parameter | Use | Default |
| :- | :- | :- |
|`nnodes`| number of nodes [int] | &cross; |
|`ndofs`| number of dofs per node [int] | &cross; |
|`nmodes`| Number of modes [int] | &check; : 0 |

## Substructure `substructure`

The `Substructure` class is the basic class of pyHarm to create dofs and add a `Substructure` Element to the System. The `Substructure` needs the matrices `M`,`G`,`C`,`K`, to be defined in order to be built as well as the number of dof per node `ndofs`. The other characteristics can be guessed. When using super-elements, the number of modes `nmodes` must be provided. It is assumed that the modal nodes are placed last in the matrices.


| Parameter | Use | Default |
| :- | :- | :- |
|`ndofs`| number of dofs per node [int] | &cross; |
|`matrix`| contains the necessary matrices [dict[str,np.ndarray]] | &check; : if `matrix` given|
|`filename`| name of the file that contains the necessary matrices [str] | &check; : if `filename` given |
|`reader`| reader use to interpret and modify input data to the right format [ABCReader] | &check; : 'generic' |
|`nmodes`| Number of modes [int] | &check; : 0 |
|`nnodes`| number of nodes [int] | &check; : auto-computed from matrix size|
