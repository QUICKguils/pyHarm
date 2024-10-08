## Basic installation : 

**pyHarm** is proposed as a full python package. In order to install the package, the user is invited to download/clone the source code from the gitlab and install the package using `pip` while in the project directory : 

```
pip install .
```

### Required libraries 

The required packages to run **pyHarm** are the following : 
- numpy
- scipy
- numba
- jax
- jax-lib
- pandas
- h5py
- matplotlib

We strongly advice to build `pyHarm` problems using **jupyter notebooks** and to add :
- notebook | jupyterlab

For developers, some other libraries can be added : 
- pytest
- sphinx


### Notes for developers : 

In order to include new developements into **pyHarm** it is advised to work in an environment where **pyHarm** is not installed and use it as a standalone package :

```
import sys
sys.path.append($PATH_TO_pyHarm_folder$)
import pyHarm
```

This way, any modification to the source file can be updated by reimporting the pyHarm module after restarting python kernel without reinstalling the package in the environment.