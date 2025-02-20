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

### Required Libraries

The required packages to run **pyHarm** are described in `pyproject.toml`:
- numpy
- scipy
- numba
- jax
- jaxlib
- pandas
- h5py
- matplotlib
- notebook

We strongly recommend building `pyHarm` projects using **Jupyter notebooks**. 

### Notes for Developers

To include new developments into **pyHarm**, it is advised to work in an environment where **pyHarm** is installed editable after cloning the repo in the folder of your choice:

```
git clone https://gitlab.com/drti/pyharm.git
```
```
cd pyharm
```
```
pip install -e .[dev]
```
The `[dev]` tag allows for the installation of the optional dependencies specific for pyHarm's development : 
- pytest in order to run the unitests 
- sphinx in order to build the documentation locally

This way, any modification to the source files can be updated by reimporting the pyHarm module after restarting the Python kernel, without reinstalling the package in the environment.

### Running the Unitary Tests and the Non-Regression Test Suite

The unitary tests are provided using the `pytest` library. They must be downloaded from the source repository at https://gitlab.com/drti/pyharm. The tests are classified into two categories: 
- **unit**: Unit tests for basic features.
- **nonregression**: Tests to ensure code non-regression, including proper physics problems.

To run the tests, use the following command with your **pyHarm** environment activated, replacing `NAME_TEST_CAT` with one of the aforementioned categories: 
```
pytest -m NAME_TEST_CAT
```