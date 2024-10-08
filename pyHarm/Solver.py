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
import scipy.linalg as spl
import copy 

class SystemSolution():
    """
    Class that represents a solution of the system to be solved.

    This class is the main object that transits in the analysis process while solving a problem. 
    The object contains information about its starting point, the previous SystemSolution it is linked to and the actual point the solver is studying.
    Once the solver has converged, the values of the residual and the solution point are stored in some of the attributes.

    Args:
        xs (np.ndarray): An array representing the starting point.
        last_solution_pointer: A pointer to the last converged solution object.
        **kwargs: Additional keyword arguments.
    
        
    Attributes:
        flag_restart (bool): Flag indicating if the solution is a restart (last_solution_pointer != index - 1).
        flag_accepted (bool): Flag indicating if the solution is considered valid.
        flag_bifurcation (bool): Flag indicating if a bifurcation has been detected at this point.
        flag_solved (bool): Flag indicating if the solution is complete and valid.
        flag_intosolver (bool): Flag indicating if the solution has gone through the solver.
        flag_R (bool): Flag indicating the presence of a residual result.
        flag_J (bool): Flag indicating the presence of a Jacobian result.
        flag_J_qr (bool): Flag indicating the availability of the Jacobian with QR decomposition.
        flag_J_lu (bool): Flag indicating the availability of the Jacobian with LU decomposition.
        flag_J_f (bool): Flag indicating the availability of the full-size Jacobian.
        index_insolve (int): The index of the solution within the solver.
        ds (float): The step size for the continuation.
        sign_ds (int): Sign of the step size.
        x_start (np.ndarray): An array representing the starting point.
        x (np.ndarray): An array representing the current solution point.
        x_pred (np.ndarray): An array representing the prediction point during continuation.
        R: Residual values.
        J_f: Full-size Jacobian.
        J_lu: Jacobian with LU decomposition.
        J_qr: Jacobian with QR decomposition.
        precedent_solution: A pointer to the last converged solution.

    """
    def __init__(self,xs:np.ndarray,last_solution_pointer=None,**kwargs):
        ######################### The flags of the SystemSolution class
        self._init_flags()
        ######################### Values and Vectors
        # The scalar/integer values
        self.index_insolve = 0
        self.ds = 0.
        self.sign_ds = 1.
        # The variable points of the SystemSolution
        self.x_start = xs
        self.x = copy.deepcopy(xs)
        self.x_pred = np.zeros_like(xs)
        ######################### Values and Vectors
        self.R = None
        self.J_f = None
        self.J_lu = None
        self.J_qr = None
        # If the continuation is going, give the pointer to the last conveged solution
        self.precedent_solution = last_solution_pointer

    def _init_flags(self,):
        """
        Initialise the different flag attributes of the class.
        """
        ######################### The flags of the SystemSolution class
        self.flag_restart = False # This flags is set to True whenever its last_solution_point!=index-1
        self.flag_accepted = False # This flag is set by the solver, if the solution can be considered valid
        self.flag_bifurcation = False # This flags True when bifurcation has been detected at this point
        self.flag_solved = False # this flag is set with method CheckComplete
        self.flag_intosolver = False # this flag is set to True when the Solution went through the solver
        ### Residual flag
        self.flag_R = False # Presence of a Residual result 
        ### Jacobian flags
        self.flag_J = False # presence of a Jacobian result
        self.flag_J_qr = False # Jacobian available with qr formalism of scipy.linalg.qr=[Q,R]
        self.flag_J_lu = False # Jacobian available with lu formalism of scipy.linalg.lu=[P,L,U]
        self.flag_J_f = False # Jacobian available full size


    def CheckComplete(self):
        """
        Checks if all elements required to proceed are present in the SystemSolution object.
        
        Returns:
            bool: True if the solution is considered valid.
        """
        # presence of Jacobian at the solution point :
        if self.flag_R and self.flag_J and self.flag_intosolver: self.flag_solved=True
        return self.flag_solved

    def SaveSolution(self,List:list):
        """
        Saves the SystemSolution object in the provided list if it is complete.
        
        Args:
            List (list): A list to save the SystemSolution object.
        
        Raises:
            ValueError: If the SystemSolution is not complete.
        """
        if self.CheckComplete() :
            List.append(self)
        else:
            raise ValueError("The SystemSolution is not complete and thus cannot be saved in the provided list")

    def getJacobian(self,format:str="full",dump:bool=False) -> np.ndarray: 
        """
        Returns the Jacobian in the specified format.
        
        Args:
            format (str): The format of the Jacobian. Options: "full", "qr", "lu" (default: "full").
            dump (bool): If True, erases the Jacobian result with the format_in (default: False).
            
        Returns:
            np.ndarray: The Jacobian in the requested format, or None if the Jacobian is not available.
            
        Raises:
            ValueError: If the format is not compatible.
        """
        format_poss = {"full":self.flag_J_f,"qr":self.flag_J_qr,"lu":self.flag_J_lu}
        if not self.flag_J : return None
        elif format not in format_poss : raise ValueError("Format not compatible")
        elif (format == "full" and self.flag_J_f) : return self.J_f
        elif (format == "qr" and self.flag_J_qr) : return self.J_qr
        elif (format == "lu" and self.flag_J_lu) : return self.J_lu
        else : 
            if (format == "qr" and self.flag_J_f) : 
                self.convertJacobian("full","qr")
                return self.J_qr
            if (format == "full" and self.flag_J_qr) : 
                self.convertJacobian("qr","full")
                return self.J_f

    def convertJacobian(self,format_in,format_out,dump=False):
        """
        Converts the Jacobian format from format_in to format_out.
        
        Args:
            format_in (str): The current format of the Jacobian.
            format_out (str): The desired format of the Jacobian.
            dump (bool): If True, erases the Jacobian result with the format_in (default: False).
            
        Returns:
            tuple: The updated flags and Jacobian data.
            
        Raises:
            ValueError: If the format is not compatible.
        """
        format_poss = {"full":[self.flag_J_f,self.J_f],"qr":[self.flag_J_qr,self.J_qr],"lu":[self.flag_J_lu,self.J_lu]}
        ### Converts the format_in for the Jacobian to the format out, if dump=True, 
        ### then erases the Jacobian result with the format_in
        def from_qr_to_full():
            self.J_f = np.dot(self.J_qr[0],self.J_qr[1])
            self.flag_J_f = True
            pass 

        def from_full_to_qr():
            self.J_qr = spl.qr(self.J_f,mode="full")
            self.flag_J_qr = True
            pass

        def from_full_to_lu():
            self.J_lu = spl.lu(self.J_f)
            self.flag_J_lu = True
            pass

        def from_lu_to_full():
            self.J_f = np.dot(self.J_lu[0],self.J_lu[1],self.J_lu[2])
            self.flag_J_f = True
            pass

        def from_qr_to_lu():
            from_qr_to_full()
            from_full_to_lu()
            pass

        def from_lu_to_qr():
            from_lu_to_full()
            from_full_to_qr()
            pass
        convert_func = {"full":{"qr":from_full_to_qr,"lu":from_full_to_lu},
                        "qr":{"full":from_qr_to_full,"lu":from_qr_to_lu},
                        "lu":{"full":from_lu_to_full,"qr":from_lu_to_qr}
                        }
        convert_func[format_in][format_out]()
        if dump : 
            format_poss[format_in][0] = False
            format_poss[format_in][1] = None
            if ((format_in,format_out) == ("qr","lu") or (format_in,format_out) == ("lu","qr")) :
                format_poss["full"][0] = False
                format_poss["full"][1] = None
        return format_poss[format_out]

class FirstSolution(SystemSolution):
    """Inherits from the SystemSolution class with one major difference: there is no previous SystemSolution point
    for this class.
    
    Args:
        xs (np.ndarray): An array representing the starting point.
    """
    def __init__(self,xs):
        super().__init__(xs,None)
        self.x = copy.deepcopy(self.x_start)
        self.x_pred = copy.deepcopy(self.x_start)

            
