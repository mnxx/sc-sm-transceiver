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


def flatten_list(given_list_of_lists):
    """ Function to flatten a list of lists. """
    given_list_of_lists = list(given_list_of_lists)
    return [symbol for sub_list in given_list_of_lists for symbol in sub_list]

def split_list(given_list, block_length):
    """ Function to split a list into blocks of a given block length. """
    split = []
    # Flatten the given list.
    list_to_split = flatten_list(given_list)
    nb_blocks = np.ceil(len(list_to_split) / block_length)
    # Split the flattened list into blocks.
    for step in range(0, nb_blocks):
        split.append(list_to_split[step * block_length : step * block_length + block_length])
    # If last frame does not have enough symbols: Add zeros.
    while len(split[-1]) < block_length:
        split[-1].append(0)
    return split


class Transceiver:
    """
    Implementation of the transmission of the scheme.
    """

    def __init__(self, antenna_setup, modulation_symbols):
        self.n_t = antenna_setup[0]
        self.n_r = antenna_setup[1]
        self.modulation_index = int(np.log2(len(modulation_symbols)))
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
        """ Return a list of the possible SM symbols. """
        return self.possible_symbols

    def training_data_to_blocks(self, training_sequences):
        """ Create a training vector by modulating the data bits to modulation symbols. """
        symbols = []
        # Use a different training sequence for each transmit antenna.
        for sequence_index in range(0, self.n_t):
            # For now: Use sequences from 2 onward.
            training_sequence = training_sequences[sequence_index + 2]
            # Split training sequence into blocks and concatenate with the previous sequences.
            symbols += split_list(training_sequence, self.modulation_index)
        return symbols

    def data_to_blocks(self, data):
        """ Create a transmit vector by modulating the data bits to SC-SM symbols. """
        return int(np.log2(self.n_t)) + data

    def create_sm_symbol(self, antenna_index, modulated_symbol):
        """ Create Spatial Modulation symbols from a given antenna index and modulated symbols. """
        # Convention: Antenna indices start with 1.
        antenna_index = antenna_index + 1
        # The training data blocks are transformed to SM symbols for each antenna.
        return [0] * (antenna_index - 1) + [modulated_symbol] + [0] * (self.n_t - antenna_index)

    def training_symbols_to_frame(self, training_symbols):
        """ Function to create frames from modulated training symbols. """
        training_frame = []
        # Use a training sequence for each transmit antenna.
        for antenna_index in range(0, self.n_t):
            # The structure of the training_symbol list
            # has to imply a new spread sequence for each new antenna.
            for modulated_symbol in training_symbols:
                training_frame.append(self.create_sm_symbol(antenna_index, modulated_symbol))
        return training_frame

    def data_symbols_to_frame(self, index_list, data_symbols):
        """ Function to create SC-SM data frames from the index and the modulated data symbols. """
        return

    def create_frame(self, symbol_vector):
        return symbol_vector

    def create_training_frame(self, k):
         """ Create a training frame using k symbols from a list of possible symbols. """

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

    def lfsr(self, polynomials, seed):
        """ Function to simulate the operations of an LFSR. """
        register = np.array(seed, dtype='bool')
        feedback_list = []
        for _ in range(0, 2**len(seed) - 1):
            # Output of the register.
            feedback = register[0]
            feedback_list.append(feedback)
            # Calculate next iteration.
            for tab in polynomials:
                feedback ^= register[tab]
            # Shift register.
            register = np.roll(register, -1)
            register[-1] = feedback
        return feedback_list

    def generate_gold_sequence(self, sequence_length):
        """ Function to create a gold sequence of a given size. """
        # Generator polynomials:
        # https://github.com/mubeta06/python/blob/master/signal_processing/sp/gold.py
        polynomials = {5:[[2],[1,2,3]], 6:[[5],[1,4,5]], 7:[[4],[4,5,6]],
                        8:[[1,2,3,6,7],[1,2,7]], 9:[[5],[3,5,6]],
                       10:[[2,5,9],[3,4,6,8,9]], 11:[[9],[3,6,9]]}
        seed = list(np.ones(sequence_length))
        sequence_1 = self.lfsr(polynomials[sequence_length][0], seed)
        sequence_2 = self.lfsr(polynomials[sequence_length][1], seed)
        seq = [sequence_1, sequence_2]
        for shift in range(0, len(sequence_1)):
            seq.append(np.logical_xor(sequence_1, np.roll(sequence_2, -shift)))
        return seq
