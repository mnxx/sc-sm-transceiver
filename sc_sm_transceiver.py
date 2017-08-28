# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Implementation of the Single Carrier Spatial Modulation scheme. """


import random
import numpy as np


class Transceiver:
    """
    Implementation of the transmission of the scheme.
    """

    def __init__(self, antenna_setup, modulation_symbols):
        self.n_t = antenna_setup[0]
        self.n_r = antenna_setup[1]
        self.possible_symbols = self._create_possible_symbols(modulation_symbols)

    def _create_possible_symbols(self, modulation_symbols):
        """ Return the spatially modulated symbols from the modulated symbols. """
        index_list = list(range(1, self.n_t + 1))
        possible_symbols = []
        for mod_symbol in modulation_symbols:
            for antenna_index in index_list:
                possible_symbols.append([0] * (antenna_index - 1)
                                        + mod_symbol
                                        + [0] * (self.n_t - antenna_index))
        return possible_symbols

    def get_symbol_list(self):
        """ Return a list of the possible SM-symbols. """
        return self.possible_symbols

    def create_transmission_frame(self, k):
        """ Create a transmission frame using k random symbols from a list of possible symbols. """
        signal_vector = []
        for _ in range(0, k):
            signal_vector.append(random.choice(self.possible_symbols))
        # Flatten the list, as Spatial Modulation symbols consist of multiple symbols.
        signal_vector = [symbol for sm_symbol in signal_vector for symbol in sm_symbol]
        return signal_vector

    def add_zero_pad(self, frame, prefix_len):
        """ Add a Zero-Pad to a frame, i.e. a list of symbols. """
        for _ in range(0, prefix_len):
            frame.append(0)
        return frame

    def transmit_frame(self, k, prefix_len):
        """ Concatenation Function: Create a transmission frame, add a Zero-Pad. """
        frame = self.add_zero_pad(self.create_transmission_frame(k), prefix_len)
        # Reshape list to a numpy array in vector form (K * N_t, 1).
        return np.reshape(frame[: k * self.n_t], (k * self.n_t, 1))

    def create_frames(self):
        """ Create transmission frames. """
        return

    def remove_zero_pad(self, frame, prefix_len):
        """ Remove the padding of a frame, i.e. a list of symbols. """
        return frame[: len(frame) - prefix_len]


class LSSDetector:
    """
    Implementation of the M-algorithm based single-stream detector
    proposed by L. Xiao, D. Lilin, Y. Zhang, Y. Xiao P. Yang and S. Li:
    "A Low-Complexity Detection Scheme for Generalized Spatial Modulation Aided
    Single Carrier Systems" in the context of Single Carrier SM-MIMO LSA system.
    """

    def __init__(self, antenna_setup, M):
        self.n_t = antenna_setup[0]
        self.n_r = antenna_setup[1]
        self.M = M

    def set_M(self, new_M):
        """ Function to change the value of M. """
        self.M = new_M

    def detect(self, frame_len, symbol_list, channel, rx_vector):
        """ Detect the sent symbols by applying the M-algorithm based single-stream ML approach. """
        # D: List of lists - list of the symbols of the M survivor paths.
        # Sub-lists contain estimated symbols, grow with each iteration (N_t to K * N_t).
        D = []
        # e: List with the accumulated M lowest estimated metrics for each iteration.
        e = []
        # The D and e lists each contain M sub-elements.
        for _ in range(0, self.M):
            D.append([])
            e.append(0)
        # Decision process is repeated K (the frame length) steps.
        for step in range(0, frame_len):
            # Each step we create a list of all the metrics resulting from the possible symbols.
            possible_metrics = []
            # Find the corresponding metrics.
            for m in range(0, self.M):
                # Create separate list with the current symbols.
                possible_symbol_vector = list(D[m])
                for possible_symbol in symbol_list:
                    # Add the possible symbol to the separate list.
                    possible_symbol_vector.append(possible_symbol)
                    # Reshape list to a numpy array in vector form (K * N_t, 1).
                    x_m = np.reshape(possible_symbol_vector, (self.n_t * (step + 1), 1))
                    # All H_x block-sub-matrices are N_r x N_t.
                    h = channel[self.n_r * step : self.n_r * step + self.n_r,
                                : self.n_t * step + self.n_t]
                    # Compute the metrics for each candidate vector.
                    # Each metric is a tuple of the value and m to keep the path information.
                    metric = (e[m] + (np.linalg.norm(rx_vector[self.n_r * step : self.n_r * step + self.n_r]
                                                     - h.dot(x_m)))**2, m, possible_symbol_vector.pop())
                    # Avoid adding the same metric multiple times.
                    # As the metric tuple includes m, different paths can have the same metric value.
                    if step == 0:
                        if metric[0] not in [usedMetric[0] for usedMetric in possible_metrics]:
                            possible_metrics.append(metric)
                    elif metric not in possible_metrics:
                        possible_metrics.append(metric)
            # Sort the accumulated metrics by their metric value.
            possible_metrics.sort()
            # As D is updated, a copied list is needed to avoid the misinterpretation of paths.
            previous_D = list(D)
            # Obtain M possible symbol vectors: Keep the M paths with the smallest metrics.
            for m, metric in enumerate(possible_metrics[0 : self.M]):
                # Find the m corresponding to the metric.
                path = metric[1]
                # Update the value of the accumulated metrics.
                e[m] = metric[0]
                # Find and append the symbol corresponding to the index of the value.
                # Check if the previous symbols have been the same,
                # i.e. the m is similar to the m of the metric.
                if D[m][:] == previous_D[path][:]:
                    # Append the corresponding symbol to the list.
                    D[m].append(metric[2])
                else:
                    # Drop the current symbol list in favor of the list to which the metric belongs.
                    # Avoid appending multiple times by copying as far as the current step allows.
                    D[m] = previous_D[path][: step]
                    # Append the corresponding symbol to the list.
                    D[m].append(metric[2])
        # Final detection: Find the frame with the overall minimum metric of the M estimated frames.
        # This time the metric is calculated by a maximum likelihood detection.
        final_metric_list = []
        for index, estimated_symbols in enumerate(D):
            symbols = np.reshape(estimated_symbols, (self.n_t * frame_len, 1))
            final_metric = ((np.linalg.norm(rx_vector - channel.dot(symbols))**2), index)
            final_metric_list.append(final_metric)
        final_metric_list.sort()
        best_m = final_metric_list[0][1]
        # Return the Vector of the K * N_t symbols with the best overall metrics.
        return D[best_m]

    def _single_stream_detect(self, D, e):
        """ Function performing single-stream ML for each step in the detection. """


class ChannelEstimator:
    """
    Implementation of a channel estimation approach for the Single Carrier Spatial Modulation scheme.
    Channel estimation is done by observing the channel impulse response, using a spreading sequence,
    i.e. gold sequence, and subsequent correlation.
    """

    def __init__(self, antenna_setup):
        self.n_t = antenna_setup[0]
        self.n_r = antenna_setup[1]

    def generate_gold_sequence(self, sequence_length):
        """ Function to create a gold sequence of a given size. """
        # Generator function.
        sequence_1 = [2, 5, 9]
        sequence_2 = [3, 4, 6, 8, 9]
        seq = [sequence_1, sequence_2]
        for shift in range(0, sequence_length):
            seq.append(np.logical_xor(sequence_1, np.roll(sequence_2, -shift)))
        return seq
