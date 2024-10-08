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


from pyHarm.Solver import SystemSolution,FirstSolution
import numpy as np

def failnumber_reached_corr(sollist:list[SystemSolution], failnumber_threshold:int, stop:bool) -> bool:
    """sollist contains all the solution points that have been seen during the analysis 
    including the false convergence. A solution inside the sollist has been rejected if 
    the flag solution.flag_accepted is set to False.
    Provides a function that returns True if the number of fail is higher that the threshold"""
    if len([sol for sol in sollist if not sol.flag_accepted])>=failnumber_threshold:
        stop = True  
    return stop


def compute_direction_corr(x_size:int) -> np.ndarray:
    """Computes the direction of the guess for the next point.
    The angular frequency is the last element in the vector ([-1])."""
    direction = np.zeros(x_size)
    direction[-1] = 1
    return direction # to be modified
def xpred_null_guess_corr(lstp_x:np.ndarray,direction:np.ndarray,ds:float, sign_ds:int) -> np.ndarray:
    """Computes the x predicted using the direction of prediction, 
    the step size and its sign, and the vector of the last accepted point.
    Remember to use the angular frequency of the last solution point lstp_x[-1]"""
    zeros_x = np.zeros(lstp_x.size)
    zeros_x[-1] = lstp_x[-1]
    xpred = zeros_x + sign_ds*ds*direction
    return xpred