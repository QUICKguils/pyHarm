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

from pyHarm.StopCriterion.ABCStopCriterion import ABCStopCriterion
from pyHarm.Solver import SystemSolution,FirstSolution

class StopCriterionBounds(ABCStopCriterion):
  """
  Subclass of ABCStopCriterion : Stopper based on the angular frequency bounds given, the computation stops once a solution is found outside of the bounds.
  


  """
  name = "Stop criterion when out of angular frequency bounds"
  factory_keyword : str = "bounds"

  def getStopCriterionStatus(self,sol:SystemSolution,sollist:list[SystemSolution],**kwargs) -> bool:
    """Returns True if the bounds are reached by the solution.
    
    Args: 
      sol (SystemSolution): Actual SystemSolution out of the solver process.
      sollist (list[SystemSolution]): List containing all previous SystemSolution.

    Returns:
      bool: True if the solution is out of the bounds or if the minimum step size is reached for at least two solutions.
    
    """
    if len(sollist)>=2 : 
      if (sollist[-1].ds==self.ds_min and sollist[-2].ds==self.ds_min) : 
        return True 
    if not isinstance(sol,FirstSolution) : 
      if (sol.x[-1]<=self.bound_inf or sol.x[-1]>=self.bound_sup) :
        return True 
    else : return False