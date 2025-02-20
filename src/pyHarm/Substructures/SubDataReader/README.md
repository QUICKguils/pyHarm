# Package SubDataReader

This package contains all the substructure data reader types that are provided by **pyHarm**. The module is organized around an abstract class **ABCReader** and a **FactoryReader** that is in charge of instantiating the objects. All system objects must comply with the **ABCReader** abstract class. The section below presents the different modules that are available.

## ABCReader

The **ABCReader** class is an abstract class defining the essential components of any substructure. One abstract method is defined: 

| Methods | Use |
| :- | :- |
|`data_complete`| *Abstract method* : Complete the entry input dictionary with necessary informations such that output is conform with `ABCSubstructure` input|

### Examples of creating an `ABCReader` and adding it into an input dictionary: 

To be created, an `ABCReader` subclass needs its abstract method to be defined : 
```python 
class FakeReader(ABCStopCriterion): # inherits from abstract class
    factory_keyword="fakereader" # mandatory to define
    def data_complete(self,data:dict) -> dict:
        data['matrix'] = {
            'M':read_mass_matrix(data['filename'])
        }
        ...

INP = {
    "substructure":{
        "substructure_001":{
            ...,
            "reader":"fakereader", # calls the reader using factory_keyword.
            ...,
        },
        ...,
    },
    "plugin":[FakeReader], # adds to the factory using plugin
    ...,
}
```



## FactoryReader

This file contains the dictionary `SubstructureReaderDictionary` of all the stop criterions that are available as well as the function `generate_subreader` that creates the reader object based on the selected keyword.

## Generic Reader 

This is the basic reader provided by pyHarm. It can read input data in which the matrices are provided under the `matrix` key of the input dictionary, but it can also read `.mat` files using `scipy.io.loadmat` of certain formatting, as well as `.h5` files. All the tutorials in pyHarm are made using this generic reader. When the key `reader` is not provided in the description of the substructure, this reader is the default reader.

```python 

INP = {
    "substructure":{
        "substructure_001":{
            ...,
            "reader":"generic", # Not a mandatory key when using the default generic reader.
            ...,
        },
        ...,
    },
}
```