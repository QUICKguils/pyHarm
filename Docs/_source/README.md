# pyHarm presentation

pyHarm is an Harmonic Balance Method (HBM) based solver for mechanical nonlinear system simulations distributed under Apache 2.0 license (see `LICENSE` file for more detail about the license). The code is built as a python package and aims at performing a wide range of studies in the field of nonlinear dynamic simulations. Its main feature is Forced Response Frequency analysis using an harmonic balance solver enhanced with continuation methods.

The philosophy behind the code is to treat the mechanical system as an assembly of elementary elements/connectors such that their contribution to the residual and jacobian can be evaluated independantly. 

The code is extensively using the factory design pattern in the subpackages to introduce abstract and flexibility when developing new components.

**Documentation** is available on readthedocs : 
- [https://pyharm-saf.readthedocs.io/en/latest/](https://pyharm-saf.readthedocs.io/en/latest/)

## Basic Installation

**pyHarm** is provided as a complete Python package. To install the package, use the `pip` Python package installer with the following command:
```
pip install pyharm@git+https://gitlab.com/drti/pyharm
```

We strongly recommend using the package within a virtual environment dedicated to the library. A `pyharm_env.yml` file is available in the directory, enabling you to easily build a conda environment with the following command:
```
conda env create --name YOUR_ENV_NAME -f pyharm_env.yml
```
where `YOUR_ENV_NAME` is your chosen name for the environment. Otherwise, the default name `pyHarm_env` will be used and can be accessed via:
```
mamba activate YOUR_ENV_NAME
```

*For more details about the installation process, please refer to the dedicated section of the documentation.*

# Project content description

The repository comprises three folders. The core files of the pyHarm code are contained in the `pyHarm` folder. The `Tutorials` folder contains a set of Tutorials to learn how to use pyHarm in the form of *Jupyter Notebooks*. Finally, the `tests` folder contains a set of `pytest` tests divided into two sections : 
- unitests : contains small tests that check specific parts of the source code
- nonregression : contains complete analysis of use cases


To run the tests, use the following command with your **pyHarm** environment activated, replacing `NAME_TEST_CAT` with one of the aforementioned categories: 
```
pytest -m NAME_TEST_CAT
```

| `$TEST_SET$` | Description |
| :- | :- |
| all | run all the tests contained in test folder |
| unit | run only the `unit` tests that check specific parts of the source code |
| nonregression | run only the `nonregression` tests that contain complete analysis of use cases |

NB: *When installing the `pyHarm` package, the tests as well as the Tutorials do not get installed alongside.*