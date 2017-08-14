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
Implementation of "A Low-Complexity Detection Scheme for Generalized Spatial Modulation Aided
Single Carrier Systems" in context of SC SM-MIMO LSA system.

Therefore following assumptions have been made:
 - Small scale fading
 - Frequency selective fading is at hand

"""


import sys
import random
from operator import itemgetter
import numpy as np


def create_possible_symbols(modulation_symbols, n_t):
    """ Return the spatially modulated symbols from the modulated symbols. """
    index_list = list(range(1, n_t + 1))
    possible_symbols = []
    for mod_symbol in modulation_symbols:
        for antenna_index in index_list:
            possible_symbols.append([0] * (antenna_index - 1)
                                    + mod_symbol
                                    + [0] * (n_t - antenna_index))
    return possible_symbols


def create_transmission_frame(possible_symbols, k):
    """ Return a transmission frame using k random symbols from a list of all possible symbols. """
    signal_vector = []
    for _ in range(0, k):
        signal_vector.append(random.choice(possible_symbols))
    return signal_vector


def detector(symbol_list, rx_vector, channel_matrix, n_rows, n_columns, k, M):
    """ Detect the received symbols using the M-based LSS detection scheme. """
    # D: List of lists - list of the symbols of the M survivor paths.
    # Sub-lists contain estimated symbols, grow with each iteration (N_t to K * N_t).
    D = []
    # Vector with the accumulated M lowest estimated metrics for each iteration.
    e = []

    # Build D and e.
    for _ in range(0, M):
        D.append([])
        e.append(0)

    # Repeat detection process K times
    for step in range(0, k):
        # List of selected metrics.
        possible_metrics = []
        # Find the corresponding metrics.
        for m in range(0, M):
            # Create separate list with the current symbols.
            possible_symbol_vector = list(D[m])
            #print(D[m])
            for possible_symbol in symbol_list:
                # Add the possible symbol to the seperate list.
                possible_symbol_vector.append(possible_symbol)
                #print(str(possible_symbol_vector) + "## " + str(step))
                # Make list accessible to Numpy operations.
                #xm = np.asarray(possible_symbol_vector)
                x_m = np.reshape(possible_symbol_vector, (n_columns * (step + 1), 1))
                # All H_x block-submatrices are n_rows x n_columns.
                h = channel_matrix[n_rows * step : n_rows * step + n_rows,
                                   : n_columns * step + n_columns]
                #print(h)
                #print(str(h.size) + "  " + str(h.shape))
                #h = h.reshape(1, h.size)
                # Compute the metrics for each candidate vector.
                # Each metric is a tuple of the value and m to keep the path information.
                metric = (e[m] + (np.linalg.norm(rx_vector[n_rows * step : n_rows * step + n_rows]
                                                 - h.dot(x_m)))**2, m, possible_symbol_vector.pop())
                #print(metric)
                # Avoid adding the same metric multiple times.
                # As the metric tuple includes m, different paths can have the same metric value.
                if step == 0:
                    if metric[0] not in [usedMetric[0] for usedMetric in possible_metrics]:
                        possible_metrics.append(metric)
                elif metric not in possible_metrics:
                    possible_metrics.append(metric)
        # Obtain corresponding M candidate vectors: Keep the M elements with the smallest metrics.
        # Using sorted(), the inherent order is not changed.
        #print(possible_metrics)
        #best_metrics = sorted(possible_metrics, key=itemgetter(0))[0 : M]
        possible_metrics.sort()
        #print("////" + str(best_metrics))
        # As e is updated, a copied list is needed to avoid the accumulation of metric values.
        #previous_e = list(e)
        # As D is updated, a copied list is needed to avoid the misinterpretation of paths.
        previous_D = list(D)
        for m, metric in enumerate(possible_metrics[0 : M]):
            # Find the m corresponding to the metric.
            position = metric[1]
            # Add the metric to the accumulated metrics.
            #e[m] = previous_e[position] + metric[0]
            e[m] = metric[0]
            # Find and append the symbol corresponding to the index of the value.
            # Check if the previous symbols have been the same,
            # i.e. the m is similar to the m of the metric.
            if D[m][:] == previous_D[position][:]:
            # Append the corresponding symbol to the list.
                D[m].append(metric[2])
            else:
                # Drop the current symbol list in favor of the list to which the metric belongs.
                # Avoid appending multiple times by only copying as far as the current step allows.
                D[m] = previous_D[position][: step]
                # Append the corresponding symbol to the list.
                D[m].append(metric[2])
            #print(str(D[m]) + " ### " + str(e[m]))

    # Final detection: Find the frame with the overall minimum metric of the M estimated frames.
    # This time the metric is calculated by a  complete maximum likelihood detection.
    path_list = []
    for index, estimated_symbols in enumerate(D):
        symbols = np.reshape(estimated_symbols, (n_columns * k, 1))
        path = ((np.linalg.norm(rx_vector - channel_matrix.dot(symbols))**2), index)
        path_list.append(path)
    path_list.sort()
    #print(path_list)
    best_m = path_list[0][1]
    #print("==== " + str(D[best_m]))
    # Return the Vector of the K * N_t symbols with the best overall metrics.
    return D[best_m]

def bpsk():
    """ Create BPSK Symbols. """
    return [[1], [-1]]

def add_zero_prefix(symbol_list, prefix_len):
    """ Add a Zero-Prefix to a frame, i.e. a list of symbols. """
    while prefix_len:
        symbol_list.insert(0, 0)
        prefix_len = prefix_len - 1
    return symbol_list

def noise(elements_per_frame, SNR):
    """ Create a random complex noise vector, CN(0, sigma**2) distributed. """
    return (np.random.normal(0, np.sqrt(10**(-SNR / 10) / 2), (elements_per_frame, 1))
            + 1j * np.random.normal(0, np.sqrt(10**(-SNR / 10) / 2), (elements_per_frame, 1)))


def main():
    """ Main function. """
    # Read input arguments.
    input1 = sys.argv[1]
    input2 = sys.argv[2]

    # Number of transmit antennas.
    N_t = 2

    # Number of reception antennas.
    N_r = 2

    # Frame length of our transmission - K symbols for each transmission.
    K = 4

    # Number of multipath links.
    P = 3

    # Length of the Zero-Prefix.
    ZP_len = P-1

    # Signal to Noise Ratio.
    SNR = int(input1)

    # M algorithm: breadth-first search with M survivors.
    M = int(input2)

    # If there is no multipath-propagation:
    if P < 2:
        print("No multipath-propagation.")

    # H: (K+P-1)N_r x KN_t
    # General Channel Matrix containing the sub-matrices of the channel.
    sub_matrices = dict()
    # Each sub-matrix has a dimension of N_r x N_t.
    for index in range(0, P):
        sub_matrices[index] = np.random.randn(N_r, N_t) / 2 + 1j * np.random.randn(N_r, N_t) / 2
    # Number of columns of sub-matrices is K.
    nb_columns = K
    # Number of rows of sub-matrices is (K+P-1).
    nb_rows = (K + P - 1)
    # Create 4-dimensional matrix using the sub-matrices.
    channel_matrix = np.zeros((nb_rows, N_r, nb_columns, N_t), dtype=sub_matrices[0].dtype)
    for index, sub_matrix in sub_matrices.items():
        for element in range(nb_columns):
            channel_matrix[index + element, :, element, :] = sub_matrix
    # Flatten the 4-dimensional matrix.
    channel_matrix.shape = (nb_rows * N_r, nb_columns * N_t)
    #print(channelMatrix.shape)

    # x: KN_t x 1 // x_k: N_t x 1
    # Signal Vector. Signals consists of spatial modulation symbols.
    possible_symbols = create_possible_symbols(bpsk(), N_t)
    signal_list = create_transmission_frame(possible_symbols, K)
    #print(signal_list)
    # Add a Zero-Prefix with length P-1
    #prefixed_signal_list = add_zero_prefix(signal_list.tolist(), ZP_len)
    # Create the corresponding signal vector with the expected shape and type.
    signal_vector = np.reshape(signal_list, (K * N_t, 1))
    #print(np.transpose(signal_vector))

    # n: (K+P-1)N_r x 1
    # Noise Vector. Elements are complex Gaussian distributed.
    noise_vector = noise((K + P - 1) * N_r, SNR)
    #print(noise_vector)

    # Y: (K+P-1)N_r x 1
    # Reception Vector. Signal attenuated by frequency selective fading channel.
    rx_vector = channel_matrix.dot(signal_vector) + noise_vector
    #print(rx_vector)

    # DETECT.
    estimated_vector = np.reshape(detector(possible_symbols,
                                           rx_vector,
                                           channel_matrix,
                                           N_r,
                                           N_t,
                                           K,
                                           M),
                                  (K * N_t, 1))

    # Show if any errors have occurred.
    signal_vector = signal_vector.flatten()
    estimated_vector = estimated_vector.flatten()
    count = 0
    for index in range(0, len(signal_vector)):
        if signal_vector[index] != estimated_vector[index]:
            count += 1
            #print("++ " + str(signal_vector[index]))
            #print("__ " +  str(estimated_vector[index]))
    print(count)


if __name__ == '__main__':
    main()
