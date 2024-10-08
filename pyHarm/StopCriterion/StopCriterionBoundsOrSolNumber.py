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

from pyHarm.StopCriterion.StopCriterionBounds import StopCriterionBounds
from pyHarm.Solver import SystemSolution
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary

class StopCriterionBoundsOrSolNumber(StopCriterionBounds):
  """
  Subclass of ABCStopCriterion : Stopper based on the number of solutions found.
  


  """
  name = "Stop criterion when out of angular frequency bounds"
  factory_keyword : str = "solnumber"
  default = {"max_solutions" : 100}
  def __init__(self,bounds:list[float,float],ds_min,**kwargs):
    super().__init__(bounds,ds_min,**kwargs)
    self.stopper_options = getCustomOptionDictionary(kwargs,self.default)
    self.max_solsaved = self.stopper_options["max_solutions"]

  def getStopCriterionStatus(self,sol:SystemSolution,sollist:list[SystemSolution],**kwargs) -> bool:
    """Returns True if the number of solutions in the list is reached.
    
    Args: 
      sol (SystemSolution): Actual SystemSolution out of the solver process.
      sollist (list[SystemSolution]): List containing all previous SystemSolution.

    Returns:
      bool: True if the number of solutions is reached.
    
    """
    if super().getStopCriterionStatus(sol,sollist,**kwargs) : return True
    if len(sollist)==self.max_solsaved: 
      return True
    else : return False