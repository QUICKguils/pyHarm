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

from pyHarm.Substructures.SubDataReader.ABCReader import ABCReader

class GenericReader(ABCReader):
    """
    Generic read is the basic substructure reader in pyHarm. It can read '.mat', '.h5' files and complete the input dictionary as well as reading already completed input dictionary (containing all the matrices)

    Attributes:
        factory_keyword (str): keyword to be called when instantiating the object through the factory.
    """
    
    factory_keyword = 'generic'
    
    def data_complete(self,data:dict) -> dict:
        """
        Reads and completes the input dictionary from the file.

        Args:
            data (dict): input dictionary.

        Returns: 
            dict: Returns a completed version of the input data dictionary.
        """
        if 'matrix' not in data :
            data = self.read_and_complete_matrix(data)

        if 'nmodes' not in data : 
            data['nmodes'] = 0

        if 'nnodes' in data : 
            pass
        elif (('nnodes' not in data) and (len(data['matrix'])!=0)) : 
            matrix1 = list(data['matrix'].values())[0]
            if (len(matrix1) - data['nmodes'])%data['ndofs'] != 0 : 
                raise ValueError("Resulting number of nodes is not an int.\nVerify that you provided the right number of dofs per node")
            data['nnodes'] = (len(matrix1) - data['nmodes'])//data['ndofs']
        else :
            raise ValueError("Unable to generate a substructure with the given inputs.\nPlease give either a dictionary containing the matrices or a filename")


        if "matching" not in data :
             data["matching"] = [i for i in range(data["ndofs"])]

        return data
    
    def read_and_complete_matrix(self,data):
        """
        Reads and completes the 'matrix' input dictionary value from the input dictionary.

        Args:
            data (dict): input dictionary.

        Returns: 
            dict: Returns a completed version of the input data dictionary.
        """
        if 'filename' in data :
            filename = data['filename']
            if filename.endswith(".mat") : 
                matrix,data = self.read_mat_files(filename,data)
            ### --- How to read h5 type of input file --- ###
            elif filename.endswith(".h5") :
                matrix,data = self.read_h5_files(filename,data)
        else : matrix=dict()
        data['matrix'] = matrix    
        return data
    
    def read_mat_files(self,filename,data) :
        """
        Reads and completes the 'matrix' input dictionary value from the input dictionary when a .mat file is required to be read.

        Args:
            data (dict): input dictionary.

        Returns: 
            dict: Returns a completed version of the input data dictionary.
        """
        from scipy.io import loadmat
        Mat = loadmat(filename)
        matrix = {
            "M":Mat["M"],
            "C":Mat["C"],
            "G":Mat["G"],
            "K":Mat["K"]
        }
        return matrix,data
    
    def read_h5_files(self,filename,data):
        """
        Reads and completes the 'matrix' input dictionary value from the input dictionary when a .h5 file is required to be read.

        Args:
            data (dict): input dictionary.

        Returns: 
            dict: Returns a completed version of the input data dictionary.
        """
        from h5py import File
        Mat = File(filename, 'r') 
        matrix = {
            "K":Mat['Data_SE']['Onde_0']['RAIDEUR'][:,:],
            "M":Mat['Data_SE']['Onde_0']['MASSE'][:,:],
            "G":0*Mat['Data_SE']['Onde_0']['MASSE'][:,:]
        }
        if "damping" in data and "Rayleigh" in data["damping"]:
            alpha = data["damping"]["Rayleigh"]["coef_M"]
            beta  = data["damping"]["Rayleigh"]["coef_K"]
            matrix['C'] = float(alpha) * matrix['M'] + float(beta) * matrix['K']
        else :
            matrix['C'] = float(alpha) * matrix['M'] + float(beta) * matrix['K']
        data["nmodes"] = len(Mat['Data_SE']['Onde_0']['FREQ'][:])
        return matrix,data
        