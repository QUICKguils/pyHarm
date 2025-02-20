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

"""
This file contains a set of general useful functions that are in use in most of pyHarm modules.

Attributes:
    Dico_ABCClass_factory_keyword (dict): Dictionary linking the abstract classes defined in every available subpackages and their factory dictionary.
"""
import copy
import numpy as np
import pandas as pd

def getCustomOptionDictionary(custom_options:dict, default_options:dict):
    """Given a custom option dictionary and a default option dictionary,
    returns a new dictionary containing all the custom information along with the missing mandatory ones from the default dictionary.
    
    Args:
        custom_options (dict): A dictionary containing custom options.
        default_options (dict): A dictionary containing default options.
    
    Returns:
        dict: A new dictionary combining the custom options with the default options.
    """
    # Creation of the dictionary to return as a copy of the default one
    options = copy.copy(default_options)
    for key in custom_options.keys():
        options[key] = custom_options[key]
    return options


def getIndexfromExpldofs(expl_dofs:pd.DataFrame, list_of_caracteristics:list[tuple[str,int,list[int]]]):
    """Uses an explicit dof vector and returns an array of indices corresponding to the required dofs.
    Takes a list of tuples [(substructure_name[str], node_number[int], dir_num[list[int]]), ...].
    If None is given as input for the directions, then all the dofs from the node are returned.
    
    Args:
        expl_dofs (pd.DataFrame): A DataFrame representing the explicit representation of the degree of freedom vector.
        list_of_caracteristics (list[tuple[str,int,list[int]]]): A list of tuples specifying the substructure name, node number, and direction numbers.
    
    Returns:
        np.ndarray: An array of indices corresponding to the required dofs, sorted in ascending order.
    """
    matching=pd.Series([False]*len(expl_dofs))
    for sub,node,list_dirs in list_of_caracteristics: 
        submatch = (expl_dofs["sub"]==sub)
        nodematch = (expl_dofs["node_num"]==node)
        dof_match=pd.Series([False]*len(expl_dofs))
        for dir in list_dirs : 
            dof_match += (expl_dofs["dof_num"]==dir)
        matching += submatch*nodematch*dof_match
    return np.sort(expl_dofs[matching].index)

# Plugin system for the ABCClasses present in pyHarm
from pyHarm.Analysis.FactoryNonLinearStudy import ABCAnalysis, NonLinearStudy_kind
from pyHarm.Correctors.FactoryCorrector import ABCCorrector, Corrector_dico
from pyHarm.Elements.FactoryElements import ABCElement, ElementDictionary
from pyHarm.Predictors.FactoryPredictor import ABCPredictor, Predictor_dico
from pyHarm.Reductors.FactoryReductors import ABCReductor, Reductor_dico
from pyHarm.StepSizeRules.FactoryStepSize import ABCStepSizeRule, StepSizer_dico
from pyHarm.StopCriterion.FactoryStopCriterion import ABCStopCriterion, Stopper_dico
from pyHarm.Systems.FactorySystem import ABCSystem, System_dico
from pyHarm.KinematicConditions.FactoryKinematic import ABCKinematic, Kinematic_dico
from pyHarm.NonLinearSolver.FactoryNonLinearSolver import ABCNLSolver, Solver_dico
from pyHarm.Substructures.FactorySubstructure import SubstructureDico, ABCSubstructure
from pyHarm.Substructures.SubDataReader.FactoryReader import SubstructureReaderDictionary,ABCReader

Dico_ABCClass_factory_keyword = {
    ABCAnalysis :               NonLinearStudy_kind,
    ABCCorrector :              Corrector_dico,
    ABCElement :                ElementDictionary,
    ABCPredictor :              Predictor_dico,
    ABCReductor :               Reductor_dico,
    ABCStepSizeRule :           StepSizer_dico,
    ABCStopCriterion :          Stopper_dico,
    ABCSystem :                 System_dico,
    ABCKinematic :              Kinematic_dico,
    ABCNLSolver :               Solver_dico,
    ABCSubstructure:            SubstructureDico,
    ABCReader:                  SubstructureReaderDictionary
}
"""dict: Dictionary linking the abstract classes defined in every available subpackages and their factory dictionary."""

def pyHarm_plugin(cls):
    """Plugin function for the pyHarm module. Allows registering a class in the pyHarm factories.

    Args:
        cls: A class to be registered in one of the pyHarm factories.
    """
    def check_keyword(cls,Dico):
        if cls.factory_keyword in Dico.keys() : 
            print(f"Warning : {cls.factory_keyword} is being override by plugin - consider changing your factory_keyword")
    for ABCClass, Dico in Dico_ABCClass_factory_keyword.items() : 
        if issubclass(cls,ABCClass) :
            check_keyword(cls,Dico)
            Dico[cls.factory_keyword] = cls



######################################################################################################################################
#                                     ######         NUMBERING CONVENTION            ######                                          #
######################################################################################################################################
#                                                                                                                                    #
#   consider 3 masses, with nh = 1 :                                                                                                 #
#       substructure 1 :                                                                                                             #
#           mass 1 : 1 dof                                                                                                           #
#       substructure 2 :                                                                                                             #
#           mass 2 : 3 dof                                                                                                           #
#           mass 3 : 3 dof                                                                                                           #
#                                                                                                                                    #
#   substructure           1          ||                                            2                                                #
#   mass                   1          ||                      2                     ||                    3                          #
#   physical dof           0          ||                 0    1    2                ||                3   4   5                      #
#   harmonic        0   |  1c  |  1s  ||      0        |      1c     |      1s      ||     0       |      1c     |      1s           #       
#   dof type        x   |  x   |  x   ||  x   y   z    |  x   y   z  |  x   y   z   ||  x   y   z  |  x   y   z  |  x   y   z        #
#   global number   0   |  7   |  14  ||  1   2   3    |  8   9   10 |  15  16  17  ||  4   5   6  |  11  12  13 |  18  19  20       #
#   indices vector  0  7  14          ||  1  2  3  4  5  6  8  9  10  11  12  13  15  16  17                                         #
#                                                                                                                                    #
#   vocabulary :                                                                                                                     #
#       mass 1, 2, 3 : 3 harmonic blocks  (2nh+1) -> 0, 1c, 1s                                                                       #
#       mass 1 : 1 dof per harmonic block (accessible in element.ndof)                                                               #
#       mass 2 : 3 dof per harmonic block (accessible in element.ndof)                                                               #
#       mass 3 : 3 dof per harmonic block (accessible in element.ndof)                                                               #
#       stepHarmonicBlock :   in global index vector, this is the step to travel from one harmonic block to the next                 #
#                           in this case, stepHarmonicBlock = 7 = sum over each mass of nb of dof per harmonic block = 1 + 3 + 3     #
#                           in the general case, the number of dof per harmonic block of each tie needs to be substracted            #
#       element["sub"].indices : vector gathering all harmonic dofs of each subsctructure, sorted in ascending order                 #
#                                                                                                                                    #
#   nonlinear element between mass 2 and mass 3 in (y,z) is defined in json as :                                                    #
#       "nonlinearspring": {                                                                                                         #
#           "dofs": {                                                                                                                #
#                 "sub2": 1, 2                                                                                                       #
#                 "INTERNAL": 4, 5                                                                                                   #
#             },                                                                                                                     #
#            "type": "2D_gap",                                                                                                       #
#            "stiffness": 1.0                                                                                                        #
#                                                                                                                                    #
#   global number of nonlinear element  :    2 3 9 10 16 17 5 6 12 13 19 20                                                          #
#   indices vector of nonlinear element :    2 3 5 6 9 10 12 13 16 17 19 20                                                          #
#                                                                                                                                    #
######################################################################################################################################