# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" 
Implementation of ""A Low-Complexity Detection Scheme for Generalized Spatial Modulation Aided Single Carrier Systems" in context of SC SM-MIMO LSA system.

Therefore following assumptions have been made:
 - N_t is 1
 - Frequency selective fading is at hand

"""


import sys
import numpy as np
import scipy.linalg as lin


def detector():
    return

def bpsk(symbols_per_frame):
    bpsk_map = np.array([1, -1])
    #symbol_indices_train = np.random.randint(0, bpsk_map.size, symbols_per_frame)
    symbol_indices_data = np.random.randint(0, bpsk_map.size, symbols_per_frame) 
    #s_train = bpsk_map[symbol_indices_train]
    return np.array([bpsk_map[symbol_indices_data]])

def noise(elements_per_frame):
    return np.random.randn(elements_per_frame, 1) + 1j * np.random.randn(elements_per_frame, 1)


def main():
    # Number of transmit antennas.
    #N_t = 1

    # Number of reception antennas.
    N_r = 2
    
    # Frame length of our transmission - K symbols for each transmission.
    K = 8

    # Number of multipath links.
    P = 3
    
    # Length of the Zero-Prefix
    ZP_len = P-1
    
    # H_t: tN_r x tN_t
    # The submatrices of H. Elements are complex Gaussian distributed.
    #CN_mean = [0, 0]
    #CN_cov = [[1, 0], [0, 1]]

    #OM = np.array([[0 + 0j], [0 + 0j]])
    #print(OM)
    #x = np.random.multivariate_normal(CN_mean, CN_cov, 1)
    H_0 = np.random.randn(2,1) + 1j * np.random.randn(2,1)
    #print(H_0.size)
    H_1 = np.random.randn(2,1) + 1j * np.random.randn(2,1)
    #print(H_1)
    H_2 = np.random.randn(2,1) + 1j * np.random.randn(2,1)

    
    # H: (K+P-1)N_r x KN_t
    # General Channel Matrix containing the submatrices of the channel.
    #rowList = [H_0, H_1, H_2, OM, OM, OM, OM, OM, OM, OM]
    #columnList = [OM, OM, OM, OM]
    #channelMatrix = lin.toeplitz([1,2,3,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0])
    #channelMatrix = lin.toeplitz(rowList, columnList)
    #print(channelMatrix[0])
    #print(channelMatrix[1])
    #print(channelMatrix[2])

    # H: (K+P-1)N_r x KN_t
    # General Channel Matrix containing the submatrices of the channel.
    inputs = (H_0, H_1, H_2)
    # Number of columns is KN_t = input channel matrices + C.
    number_columns = len(inputs) + 5
    # Number of rows is (K+P-1)N_r.
    number_rows = 10
    # All submatrices should have the same dimensions.
    nb_sub_rows, nb_sub_columns = H_0.shape
    # Create 4-dimensional matrix using the submatrices.
    channelMatrix = np.zeros((number_rows, nb_sub_rows, number_columns, nb_sub_columns), dtype = H_0.dtype)
    for index, subMatrix in enumerate(inputs):
        for element in range(number_columns):
            channelMatrix[index + element, : , element, :] = subMatrix
    # Flatten the 4-dimensional matrix.
    channelMatrix.shape = (number_rows * nb_sub_rows, number_columns * nb_sub_columns)
    #print(channelMatrix.size)

    # x: KN_t x 1 // x_k: N_t x 1
    # Signal Vector. No spatial information for now.
    signalVector = np.transpose(bpsk(K))
    #print(signalVector)
    
    # n: (K+P-1)N_r x 1
    # Noise Vector. Elements are complex Gaussian distributed.
    noiseVector = noise((K+P-1)*N_r)
    #print(noiseVector)
    
    # Y: (K+P-1)N_r x 1
    rxVector = channelMatrix.dot(signalVector) + noiseVector
    print(rxVector)
    

if __name__ == '__main__':
    main()
