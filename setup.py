# Copyright 2024 SAFRAN SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from setuptools import setup, find_packages


def read_version():
    version_file = os.path.join(os.path.dirname(__file__), 'pyHarm', '__version__.py')
    with open(version_file) as f:
        exec(f.read())
        return locals()['__version__']


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyHarm',
    version=read_version(),
    description='A Harmonic Balance code for mechanical vibration systems with nonlinearities.',
    author='J.Armand, Q.Mercier',
    author_email="pyharm.fr.saf@safrangroup.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/drti/pyharm",
    packages=find_packages(exclude=['tests','tests.*']),
    install_requires=[
        'scipy>=1.7.1',
        'numpy>=1.20.3',
        'matplotlib>=3.4.3',
        'numba>=0.56.2',
        'h5py>=3.7.0',
        'jax>=0.4.12',
        'pandas>=1.5.1',
        'jaxlib>=0.4.12',
        'pytest >= 8.3.3',
        'notebook >= 7.2.2'
    ],
    include_package_data=True,
    package_data={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.6',


)