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

import abc
import pandas as pd
import numpy as np

from pyHarm.CoordinateSystem import GlobalCoordinateSystem,generateCoordinateSystem
from pyHarm.Substructures.ABCSubstructure import ABCSubstructure
from pyHarm.Substructures.FactorySubstructure import generate_substructure
from pyHarm.Elements.ABCElement import ABCElement
from pyHarm.KinematicConditions.ABCKinematic import ABCKinematic
from pyHarm.Elements.FactoryElements import generateElement
from pyHarm.KinematicConditions.FactoryKinematic import generateKinematic
from pyHarm.BaseUtilFuncs import getCustomOptionDictionary

class ABCSystem(abc.ABC):
    """This is the abstract class ruling the system class. The system is responsible for assembling the Residual and Jacobian of a list of elements.
    
    Args:
        idata (dict): A dictionary containing the system parameters, the local coordinate systems, the substructures the kinematic conditions and the connectors.

    Attributes:
        system_options (dict): dictionary containing other options for creation of the system class.
        nh (int): number of harmonics.
        nti (int): number of time steps.
        adim (bool): adimension the equations using the characteristic length and angular frequency.
        adim_options (dict): contains the characteristic length and angular frequency.
        lc (float): characteristic length.
        wc (float): characteristic angular frequency.
        ndofs (int): total number of degrees of freedom.
        LE (list[ABCElement]): list of elements.
        LC (lit[ABCKinematic]): list of kinematic conditions.
        LE_extforcing (list[ABCElement]): list of elements of type external forcing.
        LE_linear (list[ABCElement]): list of elements that are linear towards the displacement.
        LE_nonlinear_dlft (list[ABCElement]): list of elements that are nonlinear while using DLFT formulation.
        LE_nonlinear_nodlft (list[ABCElement]): list of elements that are nonlinear while not using DLFT formulation.
        expl_dofs (pd.DataFrame): Dataframe that explicit the nature of the degrees of freedom vector.
        ndofs_solve (int): number of degree of freedom that are to be solved.
    """
    default = {"nh":1,"nti":128,"adim":{"status": False,
                                        "lc": 1.0,
                                        "wc": 1.0}}
    """dict: set of default parameters for the system class if not given in the input argument."""

    @property
    @abc.abstractmethod
    def factory_keyword(self):
        ...
        
    def __init__(self, idata:dict) -> None:
        self._maincarateristics(idata)
        self._buildElements(idata)
        self.ndofs = len(self.expl_dofs)
        self.LE_extforcing = [e for e in self.LE if ((not e.flag_nonlinear) and (e.flag_extforcing))]
        self.LE_linear = [e for e in self.LE if ((not e.flag_nonlinear) and (not e.flag_extforcing))]
        self.LE_nonlinear_dlft = [e for e in self.LE if ((e.flag_nonlinear) and (e.flag_DLFT))]
        self.LE_nonlinear_nodlft = [e for e in self.LE if ((e.flag_nonlinear) and not (e.flag_DLFT))]
        self._complete_expl_dofs()  
        self.__post_init__()

    def _maincarateristics(self,idata:dict) -> None:    
        """Set attributes of the system class based on the input dictionary.
        
        Args:
            idata (dict): A dictionary containing the system parameters.

        Attributes:
            system_options (dict): dictionary containing other options for creation of the system class.
            nh (int): number of harmonics.
            nti (int): number of time steps.
            adim (bool): adimension the equations using the characteristic length and angular frequency.
            adim_options (dict): contains the characteristic length and angular frequency.
            lc (float): characteristic length.
            wc (float): characteristic angular frequency.
        """
        self.system_options = getCustomOptionDictionary(idata.get("system",dict()),self.default)
        self.nh = self.system_options["nh"]
        self.nti = self.system_options["nti"]
        self.adim_options = getCustomOptionDictionary(self.system_options.get("adim",dict()),self.default["adim"])
        self.adim = self.adim_options["status"]
        if not self.adim :
            self.adim_options["lc"] = 1.
            self.adim_options["wc"] = 1.
        self.lc = self.adim_options["lc"]
        self.wc = self.adim_options["wc"]
    
    def _complete_expl_dofs(self,) -> None:    
        """Modify the explicit dof vector according to the presence of kinematic conditions or non-linear connexions on certain dofs.
        
        Attributes:
            expl_dofs (pd.DataFrame): Dataframe that explicit the nature of the degrees of freedom vector.
            ndofs_solve (int): number of degree of freedom that are to be solved.
        """
        # add a columns of bool into the explicit dof vector telling if a dof is coonected to a nonlinear element
        L_nl  =  self.LE_nonlinear_dlft + self.LE_nonlinear_nodlft
        nl_dofs = np.zeros(len(self.expl_dofs)).astype(int)
        if len(L_nl) != 0:
            nl_dofs[np.unique(np.concatenate([e.indices for e in L_nl]))] = 1
        self.expl_dofs["NL"] = nl_dofs
        kc_dofs = np.zeros(len(self.expl_dofs)).astype(int)
        if len(self.LC) != 0:
            kc_dofs[np.unique(np.concatenate([(e.Pslave@e.indices).astype(int) for e in self.LC]))] = 1
        self.expl_dofs["KC"] = kc_dofs
        self.index_keep = self.expl_dofs[self.expl_dofs["KC"]!=1].index
        self.kick_kc_dofs = np.zeros((len(self.index_keep),self.ndofs))
        self.kick_kc_dofs[np.arange(0,len(self.index_keep)),self.index_keep] = 1
        self.ndofs_solve = len(self.index_keep)

    def _buildElements(self,idata:dict) -> None:    
        """Build the elements of the system and the kinematic conditions using the factory functions.
        
        Args:
            idata (dict): A dictionary containing the system parameters, the local coordinate systems, the substructures the kinematic conditions and the connectors.

        Attributes:
            LE (list[ABCElement]): list of elements.
            LC (lit[ABCKinematic]): list of kinematic conditions.
        """
        ### --- Generate the Coordinate systems --- #
        max_ndof = np.max(np.array([s["ndofs"] for s in idata["substructures"].values()]))
        self.dict_CS = {"global":GlobalCoordinateSystem(max_ndof)}
        if "coordinates" in idata : 
            for k,v in idata["coordinates"].items() :
                self.dict_CS[k] = generateCoordinateSystem(v)
        ### --- Initialize list of all elements --- ###
        self.LS:list[ABCSubstructure] = []
        self.LE:list[ABCElement] = []
        self.LC:list[ABCKinematic] = []
        ### --- Generate the Substructures --- ###
        for name,data in idata["substructures"].items() : 
            self.LS.append(
                generate_substructure(self.nh,name,data)
            )
        ### --- Generate the explicit dof list --- ###
        self.expl_dofs = self._generate_system_dof_DataFrame()
        ### --- Generate the connectors --- ###
        for sub in self.LS :
            for name,data in sub.connectors.items() :
                self.LE.append(
                    generateElement(self.nh,self.nti,name,data,self.dict_CS)
                )
        if "connectors" in idata : 
            for name,data in idata["connectors"].items() : 
                self.LE.append(
                    generateElement(self.nh,self.nti,name,data,self.dict_CS)
                )
        ### --- Generate the connectors --- ###
        for sub in self.LS :
            for name,data in sub.kinematics.items() :
                self.LC.append(
                    generateKinematic(self.nh,self.nti,name,data,self.dict_CS)
                )
        if "kinematics" in idata : 
            for name,data in idata["kinematics"].items() : 
                self.LC.append(
                    generateKinematic(self.nh,self.nti,name,data,self.dict_CS)
                )
        ### --- Initialize the indices for each element --- ### 
        for E in self.LE : 
            E.generateIndices(self.expl_dofs)
            if self.adim : 
                E.adim(self.lc,self.wc)
        ### --- Initialize the indices for each element --- ### 
        for E in self.LC : 
            E.generateIndices(self.expl_dofs)
            if self.adim : 
                E.adim(self.lc,self.wc)
    
    def _generate_system_dof_DataFrame(self,):
        df = pd.concat([sub.edf for sub in self.LS],ignore_index=True)
        df = df.sort_values(['harm','cs']).reset_index(drop=True)
        return df





    def _get_expl_dofs_into_solver(self,) -> pd.DataFrame:
        """Returns the explicit dof DataFrame that has to be solved after applying the kinematic condtions.

        Returns:
            pd.DataFrame: Cut explicit dof DataFrame to be solved.
        """
        expl_dofs_into_solver = self.expl_dofs[self.expl_dofs["KC"]!=1].reset_index(drop=True)
        return expl_dofs_into_solver

    def _expand_q(self,q:np.ndarray) -> np.ndarray:
        """From the reduced size displacement vector, extend it to its full size.

        Args:
            q (np.ndarray): Reduced size displacement vector (without the kinematic conditions).

        Returns:
            np.ndarray: full size displacement vector without kinematic conditions applied.
        """
        x = np.zeros(self.ndofs+1)
        x[-1] = q[-1]
        x[self.index_keep] = q[:-1]
        return x
    
    def _complete_x(self,list_of_kine:list[ABCKinematic],x:np.ndarray,**kwargs) -> np.ndarray:
        """Apply the list of kinematic conditions and returns a vector of displacement to add to the full size displacement vector.

        Args:
            list_of_kine (list[ABCKinematic]): List of kinematic conditions to be applied.
            x (np.ndarray): Full size displacement vector without kinematic conditions applied.

        Returns:
            np.ndarray: vector of displacement to add in order to obtain the full size displacement vector with kinematic conditions.
        """
        xadd = np.zeros(self.ndofs+1)
        for k in list_of_kine : 
            xadd += k.complete_x(x)
        return xadd

    def get_full_disp(self,q:np.ndarray) -> np.ndarray : 
        """Apply the full list of kinematic conditions and returns the displacement vector.

        Args:
            q (list[ABCKinematic]): Reduced size displacement vector without kinematic conditions applied.

        Returns:
            np.ndarray: Full size displacement vector with kinematic conditions applied.
        """
        x = self._expand_q(q)
        x += self._complete_x(self.LC, x)
        return x

    def _complete_R(self,list_of_kine:list[ABCKinematic],R, x,**kwargs) -> np.ndarray:
        """Apply the list of kinematic conditions and returns a vector of residual to add to the full size residual vector.

        Args:
            list_of_kine (list[ABCKinematic]): List of kinematic conditions to be applied.
            R (np.ndarray): Full size residual vector without kinematic conditions applied.
            x (np.ndarray): Full size displacement vector without kinematic conditions applied.

        Returns:
            np.ndarray: vector of residuals to add in order to obtain the full size residual vector with kinematic conditions.
        """
        Radd = np.zeros(self.ndofs)
        for k in list_of_kine : 
            Radd += k.complete_R(R, x)
        return Radd

    def _complete_J(self,list_of_kine:list[ABCKinematic], Jx, Jom, x,**kwargs) -> tuple[np.ndarray]:
        """Apply the list of kinematic conditions and returns a jacobian matrices to add to the full size jacobian matrices.

        Args:
            list_of_kine (list[ABCKinematic]): List of kinematic conditions to be applied.
            Jx (np.ndarray): Full size jacobian matrix with respect to displacement without kinematic conditions applied.
            Jom (np.ndarray): Full size jacobian matrix with respect to angular frequency without kinematic conditions applied.
            x (np.ndarray): Full size displacement vector without kinematic conditions applied.

        Returns:
            tuple(np.ndarray): jacobian matrices to add in order to obtain the full size jacobian matrix with kinematic condtions.
        """
        Jx_add = np.zeros((self.ndofs,self.ndofs))
        Jom_add = np.zeros((self.ndofs,1))
        for k in list_of_kine : 
            jx_add, jom_add = k.complete_J(Jx, Jom, x)
            Jx_add += jx_add
            Jom_add += jom_add
        return Jx_add, Jom_add

    def __post_init__(self,) -> None:
        """Method that can be complete in subclasses in order to facilitate a add during the instanciation of a class.
        """
        pass

    def _residual(self,list_of_elems:list[ABCElement],x,**kwargs) -> np.ndarray:
        """Compute the residual of a list of elements and add them in a full size residual vector.

        Args:
            list_of_elems (list[ABCElement]): List of elements with a residual contribution.
            x (np.ndarray): Full size displacement vector.

        Returns:
            np.ndarray: residual vector computed at point x for the list of elements.
        """
        Rg = np.zeros(self.ndofs)
        for e in list_of_elems :
            Rg[e.indices]+=e.evalResidual(x[:-1],x[-1],**kwargs)
        return Rg
    
    def _jacobian(self,list_of_elems:list[ABCElement],x,**kwargs) -> tuple[np.ndarray]:
        """Compute the jacobian matrices of a list of elements and add them in a full size residual vector.

        Args:
            list_of_elems (list[ABCElement]): List of elements with a residual contribution.
            x (np.ndarray): Full size displacement vector.

        Returns:
            tuple(np.ndarray): jacobian matrices computed at point x for the list of elements.
        """
        dJdx = np.zeros((self.ndofs, self.ndofs))
        dJdom = np.zeros((self.ndofs, 1))
        for e in list_of_elems : 
            mshg = tuple(np.meshgrid(e.indices, e.indices, sparse=True, indexing='ij'))
            djdx,djdom=e.evalJacobian(x[:-1],x[-1],**kwargs)
            dJdx[mshg]+=djdx
            dJdom[e.indices]+=djdom.reshape(-1,1)
        return dJdx,dJdom
    
    def _get_assembled_mass_matrix(self,x,**kwargs):
        """Compute the mass matrix of the assembled system

        Args:
            x (np.ndarray): Full size displacement vector.

        Returns:
            M_assembled (np.ndarray): full mass matrix of the system
        """
        M_assembled = np.zeros((self.ndofs, self.ndofs))
        for e in self.LE: 
            mshg = tuple(np.meshgrid(e.indices, e.indices, sparse=True, indexing='ij'))
            if e.factory_keyword == 'substructure':
                M_assembled[mshg] -= e.kronM
            else:
                if e.flag_elemtype == 2: 
                    djdx, _ = e.evalJacobian(x[:-1],1.0,**kwargs) # legit "magic value" to obtain nabla² @ kronM
                    M_assembled[mshg] += djdx
        return M_assembled
    
    def _get_assembled_stiffness_matrix(self,x,**kwargs):
        """Compute the stiffness matrix of the assembled system

        Args:
            x (np.ndarray): Full size displacement vector.

        Returns:
            K_assembled (np.ndarray): full stiffness matrix of the system
        """
        K_assembled = np.zeros((self.ndofs, self.ndofs))
        for e in self.LE: 
            mshg = tuple(np.meshgrid(e.indices, e.indices, sparse=True, indexing='ij'))
            if e.factory_keyword == 'substructure':
                K_assembled[mshg] += e.kronK
            else:
                if e.flag_elemtype == 0: 
                    djdx, _ = e.evalJacobian(x[:-1],x[-1],**kwargs)
                    K_assembled[mshg] += djdx
        return K_assembled


    @abc.abstractmethod
    def Residual(self) -> np.ndarray:
        """Abstract method that is completed in each subclass and responsible to compute the residual vector of the whole system.
        """
        pass
    @abc.abstractmethod
    def Jacobian(self) -> tuple[np.ndarray]:
        """Abstract method that is completed in each subclass and responsible to compute the jacobian matrices of the whole system.
        """
        pass