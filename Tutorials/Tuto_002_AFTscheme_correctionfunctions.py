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

import numpy as np


def cubic_problem_corr(g, R):
    return -g / (R * 6)


def taylor_factor_sinus_corr(n):
    return (-1) ** n / (np.prod(np.arange(1, (2 * n + 1) + 1)))


def x_order_corr(n):
    return 2 * n + 1


def sinR_corr(x, om, Pslave, DFT, DTF, k):
    x_dofs = (Pslave) @ x  # Pslave is used to obtain the right degrees of freedom
    # below code the obtention of the time displacement by using the inverse discrete Fourier operator DFT
    x_t = x_dofs @ DFT  # correction
    # Once the displacement is known in the time domain the force can be calculated
    sin_x_t = k * np.sin(x_t)  # correction
    # Once the effort is obtained in time domain, we need to put it back into the frequency domain using DTF
    sin_x_h = sin_x_t @ DTF  # correction
    return sin_x_h
