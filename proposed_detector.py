# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the GNU General Public License v3.0 is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" 
Implementation of ""A Low-Complexity Detection Scheme for Generalized Spatial Modulation Aided Single Carrier Systems" in context of SC SM-MIMO LSA system.

Therefore following assumptions have been made:
 - Small scale fading
 - Frequency selective fading is at hand

"""


import sys
import numpy as np
#import scipy.linalg as lin


def detector(rxVector, H, n_rows, n_columns):
    # Possible symbols: FOR NOW, LIMITED ANTENNA INDEX INFORMATION FOR 2x2.
    #symbolList = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    symbolList = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    
    # M algorithm: breadth-first search with M survivors.
    M = 2
    # E: List of lists - list of the metrics of the M survivor paths. Sublists grow with each iteration (N_t to K * N_t).
    #E = []
    # D: List of lists - list of the symbols of the M survivor paths. Sublists grow with each iteration (N_t to K * N_t).
    D = []
    # Vector with the accumulated M lowest estimated metrics for each iteration.
    e = []
    # Vector with the estimated symbols; grows with each iteration.
    #x = []

    # Build D and e.
    for m in range(0, M):
        D.append([])
        e.append(0)
    
    # Repeat detection process K times
    for step, y in enumerate(rxVector[: 8]):
        # List of selected metrics.
        possibleMetrics = []
        # Find the corresponding metrics.
        for m in range(0, M):
            # Create seperate list with the current symbols.
            possibleSymbolVector = list(D[m])
            #print(D[m])
            for possibleSymbol in symbolList:
                # Add the possible symbol to the seperate list.
                possibleSymbolVector.append(possibleSymbol)
                #print(str(possibleSymbolVector) + "## " + str(step))
                # Make list accessible to numpy operations.
                xm = np.asarray(possibleSymbolVector)
                xm = np.reshape(xm, (n_columns * (step + 1), 1))
                # All H_x block-submatrices are n_rows x n_columns.
                h = H[n_rows * step : n_rows * step + n_rows, : n_columns * step + n_columns]
                #print(h)
                #print(str(h.size) + "  " + str(h.shape))
                #h = h.reshape(1, h.size)
                # Compute the metrics for each candidate vector.
                # Each metric is a tuple of the value and m to keep the path information.
                metric = (e[m] + np.linalg.norm(rxVector[n_rows * step : n_rows * step + n_rows] - h.dot(xm)), m, possibleSymbolVector.pop())
                # Avoid adding the same metric multiple times.
                # As the metric tuple includes m, different paths can have the same metric value.
                if step == 0:
                    if metric[0] not in [usedMetric[0] for usedMetric in possibleMetrics]:
                        possibleMetrics.append(metric)
                elif metric not in possibleMetrics:
                    possibleMetrics.append(metric)
        # Obtain corresponding M candidate vectors.
        # Keep the M elements with the smallest metrics. By using sorted(), the inherent order is not changed.
        #print(possibleMetrics)
        bestMetrics = sorted(possibleMetrics)[0 : M]
        #print(str(bestMetrics) + "#####")
        for m, metric in enumerate(bestMetrics):
            # Find the m corresponding to the metric.
            position = metric[1]
            # Add the metric to the accumulated metrics.
            e[m] = e[position] + metric[0]
            # Find and append the symbol corresponding to the index of the value.
            # Check if the previous symbols have been the same, i.e. the m is similar to the m of the metric.
            if D[m][:] == D[position][:]:
            # Append the corresponding symbol to the list.
                D[m].append(metric[2])
            else:
                # Drop the current symbol list in favor of the list to which the metric belongs.
                # Avoid appending multiple times by only copying as far as the current step allows.
                D[m] = D[position][: step]
                # Append the corresponding symbol to the list.
                D[m].append(metric[2])
            #print(str(D[m]) + " ### " + str(e[m]))
                
    # Final detection: Find the best overall path.
    # The minimal path is the first Vector of our M possible transmission scenarios.
    min_path = D[0]
    # Return the Vector of the K * N_t symbols with the best overall metrics.
    return np.reshape(min_path, (8 * 2,1))

def bpsk(symbols_per_frame):
    bpsk_map = np.array([1, -1])
    #symbol_indices_train = np.random.randint(0, bpsk_map.size, symbols_per_frame)
    symbol_indices_data = np.random.randint(0, bpsk_map.size, symbols_per_frame) 
    #s_train = bpsk_map[symbol_indices_train]
    return bpsk_map[symbol_indices_data]

def addZeroPrefix(symbol_list, prefix_len):
    while prefix_len:
        symbol_list.insert(0, 0)
        prefix_len = prefix_len - 1
    return symbol_list

def noise(elements_per_frame, SNR):
    return np.random.normal(0, np.sqrt(10**(-SNR / 10) / 2), (elements_per_frame, 1)) + 1j * np.random.normal(0, np.sqrt(10**(-SNR / 10) / 2), (elements_per_frame, 1))


def main():
    # Number of transmit antennas.
    N_t = 2

    # Number of reception antennas.
    N_r = 2
    
    # Frame length of our transmission - K symbols for each transmission.
    K = 8

    # Number of multipath links.
    P = 3
    
    # Length of the Zero-Prefix.
    ZP_len = P-1

    # Signal to Noise Ratio.
    SNR = 1
        # H_t: tN_r x tN_t
    # The submatrices of H. Elements are complex Gaussian distributed.
    #CN_mean = [0, 0]
    #CN_cov = [[1, 0], [0, 1]]

    #OM = np.array([[0 + 0j], [0 + 0j]])
    #print(OM)
    #x = np.random.multivariate_normal(CN_mean, CN_cov, 1)
    #H_0 = np.sqrt(10**(-SNR/10) / 2) * (np.random.randn(N_r, N_t) + 1j * np.random.randn(N_r, N_t))
    #print(H_0.size)
    #H_1 = np.sqrt(10**(-SNR/10) / 2) * (np.random.randn(N_r, N_t) + 1j * np.random.randn(N_r, N_t))
    #print(H_1)
    #H_2 = np.sqrt(10**(-SNR/10) / 2) * (np.random.randn(N_r, N_t) + 1j * np.random.randn(N_r, N_t))

    
    # H: (K+P-1)N_r x KN_t
    # General Channel Matrix containing the submatrices of the channel.
    #rowList = [H_0, H_1, H_2, OM, OM, OM, OM, OM, OM, OM]
    #columnList = [OM, OM, OM, OM]
    #channelMatrix = lin.toeplitz([1,2,3,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0])
    #channelMatrix = lin.toeplitz(rowList, columnList)
    #print(channelMatrix[0])
    #print(channelMatrix[1])
    #print(channelMatrix[2])

    # If there is no multipath-propagation:
    if P < 2:
        # Do something.
        print("No multipath-propagation.")
        
    # H: (K+P-1)N_r x KN_t
    # General Channel Matrix containing the submatrices of the channel.
    subMatrices = dict()
    for index in range(0, P):
        #subMatrices[index] = np.sqrt(10**(-SNR/10) / 2) * (np.random.randn(N_r, N_t) + 1j * np.random.randn(N_r, N_t))
        subMatrices[index] = np.random.randn(N_r, N_t) + 1j * np.random.randn(N_r, N_t)
    #print(subMatrices[0])
    #inputs = (H_0, H_1, H_2)
    # Number of columns is KN_t = input channel matrices + C.
    number_columns = K
    #print(number_columns)
    # Number of rows is (K+P-1)N_r.
    number_rows = (K + P - 1)
    #print(number_rows)
    # All submatrices should have the same dimensions.
    nb_sub_rows, nb_sub_columns = subMatrices[0].shape
    # Create 4-dimensional matrix using the submatrices.
    channelMatrix = np.zeros((number_rows, nb_sub_rows, number_columns, nb_sub_columns), dtype = subMatrices[0].dtype)
    for index, subMatrix in subMatrices.items():
        for element in range(number_columns):
            channelMatrix[index + element, : , element, :] = subMatrix
    # Flatten the 4-dimensional matrix.
    channelMatrix.shape = (number_rows * nb_sub_rows, number_columns * nb_sub_columns)
    #print(channelMatrix.shape)

    # x: KN_t x 1 // x_k: N_t x 1
    # Signal Vector. No spatial information for now.
    #zeroPrefix = np.zeros(ZP_len, dtype=np.int)
    #signalList = np.concatenate((zeroPrefix, bpsk(K * N_t - ZP_len)))
    signalList = bpsk(K * N_t)
    # Add a Zero-Prefix with length P-1
    #prefixedSignalList = addZeroPrefix(signalList.tolist(), ZP_len)
    # Transpose 
    signalVector = np.transpose(np.array([signalList]))
    #prefixedSignalVector = np.transpose(np.array([prefixedSignalList]))
    #print(signalVector)
    
    # n: (K+P-1)N_r x 1
    # Noise Vector. Elements are complex Gaussian distributed.
    #noiseVector = np.sqrt(10**(-SNR / 10)) / 2 * noise((K + P - 1) * N_r, SNR)
    noiseVector = noise((K + P - 1) * N_r, SNR)
    #print(noiseVector)
    
    # Y: (K+P-1)N_r x 1
    rxVector = channelMatrix.dot(signalVector) + noiseVector
    #print(rxVector.size)

    # DETECT.
    estimatedVector = detector(rxVector, channelMatrix, N_r, N_t)
    #print(estimatedVector.flatten())
    #print(np.asarray(estimatedVector))

    # Show if any errors have occured.
    #estimatedVector = np.asarray(estimatedVector)
    #print(np.array_equal(signalVector, estimatedVector))
    signalVector = signalVector.flatten()
    estimatedVector = estimatedVector.flatten()
    count = 0
    for index in range(2, len(signalVector)):
        if signalVector[index] != estimatedVector[index]:
            count += 1
    
    print(count)
    

if __name__ == '__main__':
    main()
