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

from pyHarm.Solver import FirstSolution, SystemSolution
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary
import abc
import scipy.linalg as spl
import numpy as np


class ABCPredictor(abc.ABC): 
    """Abstract class for the predictor: Any added predictor shall be constructed from this class.
    
    Args:
        sign_ds (float): if -1 predict in the direction of decreasing angular frequency, if 1 the opposite direction

    Attributes:
        flag_print (bool): information are printed during the analysis if True.
        predictor_options (dict): dictionary containing the kwargs and competed using the default options if the keywords are missing.
    """
    @property
    @abc.abstractmethod
    def factory_keyword(self)->str:
        """
        Returns:
            str: keyword that is used to call the creation of this class in the system factory.
        """
        ...
        
    default_options = {"norm":"norm1", "bifurcation_detect":True, "verbose":True}
    """dict: set of default parameters for the system class if not given in the input argument.
    
    It contains a normalisation parameter using the keyword 'norm' that can be set to either 'norm1' (default) if the direction is normed to 1 or 'om' if the direction is normed to 1 only for the angular frequency.
    It contains a bifurcation detection using the 'bifurcation_detect' keyword that can be set to True (default) if detection is needed.
    It contains a 'verbose' keyword that can be set to True (default) if information about detection of bifurcations is to be displayed during solving.
    """

    def __init__(self, sign_ds, **kwargs):
        self.sign_ds_init = sign_ds
        self.sign_ds = sign_ds
        self.predictor_options = getCustomOptionDictionary(kwargs, self.default_options)
        self.flag_print = self.predictor_options["verbose"]

    @abc.abstractmethod
    def predict(self,sollist:list[SystemSolution],ds:float) -> tuple[np.ndarray,SystemSolution,float]:
        """Predicts the next strating point.

        Args:
            sollist (list[SystemSolution]): list of SystemSolution already solved during the analysis.
            ds (float): step size for the prediction.
        """
        pass
################################################### /!\

    def bifurcation_detect(self,lstpt:SystemSolution):
        """Makes a bifurcation detection analysis computing determinant of jacobian matrix and analysing change of sign.

        Args:
            lstpt (SystemSolution): previously accepted point in direct link with the actual solved point.

        Attributes: 
            sign_ds (float): Attribute is modified if a turning point is detected.
        """
        if self.predictor_options["bifurcation_detect"] == False : 
            pass
        else:
            Jaco_qr = lstpt.getJacobian("qr")
            Jaco = lstpt.getJacobian("full")
            det_J_f = spl.det(Jaco)
            det_J_x = spl.det(Jaco[:-1,:-1])
            lstpt.det_J_f = det_J_f
            lstpt.det_J_x = det_J_x
            if isinstance(lstpt,FirstSolution) : 
                    det_J_x_prec = det_J_x
                    det_J_f_prec = det_J_f
            else : 
                    det_J_x_prec=lstpt.precedent_solution.det_J_x
                    det_J_f_prec=lstpt.precedent_solution.det_J_f

            if np.sign(det_J_x) != np.sign(det_J_x_prec) :
                    if np.sign(det_J_f) == np.sign(det_J_f_prec) or np.sign(det_J_f)!=np.sign(det_J_x):
                        self.sign_ds *= -1
                        lstpt.flag_bifurcation = True
                        if self.flag_print : print('Warning: a limit point (fold bifurcation) was detected --> path direction is reversed')
                    else : 
                        lstpt.flag_bifurcation = True
                        if self.flag_print : print("Warning: a branching point was detected")
                    pass
################################################### /!\

    def getPointerToSolution(self,sollist:list[SystemSolution],k_imposed=None) -> SystemSolution: 
        """Gets the last accepted solution in direct link with the studied point.

        Args:
            sollist (list[SystemSolution]): list of SystemSolution already solved during the analysis.
            k_imposed (None|int): if not None then the provided index is used as the last accepted point.

        Returns: 
            SystemSolution: last accepted point.
        """
        if k_imposed == None :
            lstpt = sollist[-1]
            k=0
            while not lstpt.flag_accepted : 
                lstpt = sollist[-1-k]
                k+=1
        else : lstpt = sollist[k_imposed]
        return lstpt

    def norm_dir(self,dir:float) -> float: 
        """Normalises the direction according to the choice of norm given in the class attributes.

        Args:
            dir (np.ndarray): prediction direction.

        Returns: 
            np.ndarray: normalized prediction direction.
        """
        if self.predictor_options["norm"] == "norm1" : 
            dir = dir/np.linalg.norm(dir)
        elif self.predictor_options["norm"] == "om" :
            dir = dir / dir[-1]
        else : 
            print("Wrong normalisation option, please choose between \"norm1\" and \"om\"")
        return dir