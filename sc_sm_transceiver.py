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

def split_list(list_to_split, block_length):
    """ Function to split a list into blocks of a given block length. """
    split = []
    nb_blocks = int(np.ceil(len(list_to_split) / block_length))
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

    def __init__(self, antenna_setup, symbols_per_frame, sample_rate, modulation_symbols):
        self.n_t = antenna_setup[0]
        self.n_r = antenna_setup[1]
        self.k = symbols_per_frame
        self.sample_rate = sample_rate
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

    def set_symbols_per_frame(self, symbols_per_frame):
        """ Set the number of symbols per frame. """
        self.k = symbols_per_frame

    def bits_to_index(self, bits):
        """ Return the decimal value of a series of bits. """
        return

    def training_data_to_blocks(self, training_sequences):
        """ Create a training vector by modulating the data bits to modulation symbols. """
        symbols = []
        # Use a different training sequence for each transmit antenna.
        for sequence_index in range(0, self.n_t):
            # For now: Use sequences from 2 onward.
            training_sequence = training_sequences[sequence_index + 2]
            # Split training sequence into blocks and concatenate with the previous sequences.
            # IF SHAPE TWO-DIMENSIONAL: Flatten the given list.
            #list_to_split = flatten_list(given_list)
            symbols += split_list(training_sequence, self.modulation_index)
        return symbols

    def data_to_blocks(self, data):
        """ Create a transmit vector by modulating the data bits to SC-SM symbols. """
        symbols = []
        nb_antenna_bits = int(np.log2(self.n_t))
        # Put the data bits into spatial modulation blocks.
        symbols += split_list(data, self.modulation_index + nb_antenna_bits)
        # Extract bits to convey antenna index information and bits to modulate.
        for index, symbol in enumerate(symbols):
            symbols[index] = [symbol[: nb_antenna_bits], symbol[nb_antenna_bits :]]
        return symbols

    def sm_modulation(self, antenna_index, modulated_symbol):
        """ Create Spatial Modulation symbols from a given antenna index and modulated symbols. """
        # Convention: Antenna indices start with 1.
        antenna_index = antenna_index + 1
        # The training data blocks are transformed to SM symbols for each antenna.
        return [0] * (antenna_index - 1) + [modulated_symbol] + [0] * (self.n_t - antenna_index)

    def sm_demodulation(self, sm_symbol):
        """ Demodulate Spatial Modulation symbols to an antenna index and a modulated symbol. """
        antenna_index = 0
        modulated_symbol = 0
        for symbol in sm_symbol:
            if symbol == 0:
                antenna_index += 1
            else:
                modulated_symbol = symbol
                break
        return [int(bit) for bit in list(format(antenna_index, 'b'))] + [modulated_symbol]

    def training_symbols_to_frames(self, training_symbols):
        """ Function to create frames from modulated training symbols. """
        training_frame = []
        # Use a training sequence for each transmit antenna.
        seq_len = len(training_symbols) / self.n_t
        # The structure of the training_symbol list
        # has to imply a new spread sequence for each new antenna.
        for index, modulated_symbol in enumerate(training_symbols):
            training_frame.append(self.sm_modulation(int(index / seq_len), modulated_symbol))
        # Assume symbols vector has size K * N_t (BPSK).
        frame_list = split_list(flatten_list(training_frame), self.k * self.n_t)
        for index, frame in enumerate(frame_list):
            frame_list[index] = np.reshape(frame, (self.k * self.n_t, 1))
        return frame_list

    def data_symbols_to_frames(self, data_symbols):
        """ Function to create SC-SM data frames from the index and the modulated data symbols. """
        # Assume symbols vector has size K * N_t (BPSK).
        frame_list = split_list(flatten_list(data_symbols), self.k * self.n_t)
        for index, frame in enumerate(frame_list):
            frame_list[index] = np.reshape(frame, (self.k * self.n_t, 1))
        return frame_list

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
        return np.reshape(frame[: k * self.n_t], (k * self.n_t))

    def upsampling(self, rate, frame):
        """ Upsample a given frame with a given upsampling rate. """
        # Frame should be a (x, 1) Numpy array.
        upsampled_frame = np.zeros((rate * frame.size), dtype=complex)
        for index, val in enumerate(frame):
            upsampled_frame[rate * index] = val
        return upsampled_frame

    def rrc_filter(self, beta, span, sps, frame):
        """ Root-Raised-Cosine-Filter: Interpolate and pulse shape a given frame. """
        T_sample = 1.0 / self.sample_rate
        T_s = sps * T_sample
        factor = 1 / np.sqrt(T_s)
        # Length of the filter is the number of symbols  multiplied by the number of samples per symbol.
        h_rrc = np.zeros(span * sps, dtype=float)
        for index in range(0, span * sps):
            t = (index - span * sps / 2) * T_sample
            if t == 0:
                h_rrc[index] = factor * (1 - beta + (4 * beta / np.pi))
            elif beta != 0 and t == T_s / 4 / beta:
                h_rrc[index] = factor * beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / 4 / beta) + (1 - 2 / np.pi) * np.cos(np.pi / 4 / beta))
            elif beta != 0 and t == -T_s / 4 / beta:
                h_rrc[index] = factor * beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / 4 / beta) + (1 - 2 / np.pi) * np.cos(np.pi / 4 / beta))
            else:
                h_rrc[index] = factor * ((np.sin(np.pi * t / T_s * (1 - beta)) + 4 * beta * t / T_s * np.cos(np.pi * t / T_s * (1 + beta))) /
                                         (np.pi * t / T_s * (1 - (4 * beta * t / T_s)**2)))
        h_rrc = h_rrc / np.sqrt(np.sum(np.abs(h_rrc)**2))
        return np.convolve(h_rrc, frame, 'same')

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
        nb_paths = 0
        # Decision process is repeated K (the frame length) steps.
        # First step: all M paths consist of one symbol: choose M different symbols.
        possible_first_metrics = []
        for index, possible_symbol in enumerate(symbol_list):
            # Reshape list to a numpy array in vector form (K * N_t, 1).
            #print(str(len(possible_symbol)) + " ~ " + str(self.n_t) + " ~~ " + str(possible_symbol))
            x_m = np.reshape(possible_symbol, (self.n_t))
            # All H_x block-sub-matrices are N_r x N_t.
            h = channel[: self.n_r, : self.n_t]
            # Compute the metrics for each candidate vector.
            # Each metric is a tuple of the value and the symbol.
            #print(str(rx_vector[: self.n_r].shape) + " ### " + str(h.shape) + " ## " + str(h.dot(x_m)) + " ### " + str(possible_symbol))
            metric = ((np.linalg.norm(rx_vector[: self.n_r]
                                      - h.dot(x_m)))**2, possible_symbol)
            possible_first_metrics.append(metric)
        possible_first_metrics.sort()
        #print(possible_first_metrics)
        for m, metric in enumerate(possible_first_metrics[0 : self.M]):
            # Update the value of the accumulated metrics.
            e[m] = metric[0]
            # Find and append the symbol corresponding to the index of the value.
            # Append the corresponding symbol to the list.
            D[m].append(metric[1])
            nb_paths += 1
        # Subsequent steps allow for adoption of the same symbol, or change to better path with different symbol.
        for step in range(1, frame_len):
            # Each step we create a list of all the metrics resulting from the possible symbols.
            possible_metrics = []
            # Find the corresponding metrics.
            for m in range(0, nb_paths * len(symbol_list)):
                # M is upper bound.
                if m == self.M:
                    break
                # Create separate list with the current symbols.
                possible_symbol_vector = list(D[m])
                #print("~ " + str(np.mod(count2, 4)) + " ~ " + str(possible_symbol_vector))
                for possible_symbol in symbol_list:
                    # Add the possible symbol to the separate list.
                    possible_symbol_vector.append(possible_symbol)
                    # Reshape list to a numpy array in vector form (K * N_t, 1).
                    #count += 1
                    #print(count)
                    #print(len(possible_symbol_vector))
                    x_m = np.reshape(possible_symbol_vector, (self.n_t * (step + 1)))
                    # All H_x block-sub-matrices are N_r x N_t.
                    h = channel[self.n_r * step : self.n_r * step + self.n_r,
                                : self.n_t * step + self.n_t]
                    # Compute the metrics for each candidate vector.
                    # Each metric is a tuple of the value and m to keep the path information.
                    metric = (e[m] + (np.linalg.norm(rx_vector[self.n_r * step : self.n_r * step + self.n_r]
                                                     - h.dot(x_m)))**2, m, possible_symbol_vector.pop())
                    # Avoid adding the same metric multiple times.
                    # As the metric tuple includes m, different paths can have the same metric value.
                    if metric not in possible_metrics:
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
                    # A new path is about to be constructed.
                    if D[m] == []:
                        nb_paths += 1
                    # Drop the current symbol list in favor of the list to which the metric belongs.
                    # Avoid appending multiple times by copying as far as the current step allows.
                    D[m] = previous_D[path][: step]
                    # Append the corresponding symbol to the list.
                    D[m].append(metric[2])
        # Final detection: Find the frame with the overall minimum metric of the M estimated frames.
        # This time the metric is calculated by a maximum likelihood detection.
        final_metric_list = []
        #print(D[0][: 10])
        #print(e)
        for index, estimated_symbols in enumerate(D):
            #print(str(len(estimated_symbols)) + " ~ " + str(self.n_t * frame_len) + " ~~ " + str(estimated_symbols))
            symbols = np.reshape(estimated_symbols, (self.n_t * frame_len))
            final_metric = ((np.linalg.norm(rx_vector - channel.dot(symbols))**2), index)
            final_metric_list.append(final_metric)
        final_metric_list.sort()
        #print(final_metric_list)
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

    def __init__(self, antenna_setup, frame_length, sample_rate, samples_per_symbol):
        self.n_t = antenna_setup[0]
        self.n_r = antenna_setup[1]
        self.frame_length = frame_length
        self.sample_rate = sample_rate
        self.sps = samples_per_symbol

    def ccorr(self, x_array, y_array):
        """ Calculate the circular correlation of 1-D input numpy arrays using DFT. """
        return np.fft.ifft(np.fft.fft(x_array) * np.fft.fft(y_array).conj())

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
        polynomials = {5:[[2], [1, 2, 3]], 6:[[5], [1, 4, 5]], 7:[[4], [4, 5, 6]],
                       8:[[1, 2, 3, 6, 7], [1, 2, 7]], 9:[[5], [3, 5, 6]],
                       10:[[2, 5, 9], [3, 4, 6, 8, 9]], 11:[[9], [3, 6, 9]]}
        seed = list(np.ones(sequence_length))
        sequence_1 = self.lfsr(polynomials[sequence_length][0], seed)
        sequence_2 = self.lfsr(polynomials[sequence_length][1], seed)
        seq = [sequence_1, sequence_2]
        for shift in range(0, len(sequence_1)):
            seq.append(np.logical_xor(sequence_1, np.roll(sequence_2, -shift)))
        # Turn boolean numpy array into list of lists of 0 and 1.
        seq_list = []
        for sub_list in seq:
            seq_list.append(list(np.where(sub_list, 1, 0)))
        return seq_list

    def generate_zadoffchu_sequence(self, root_index, sequence_length):
        """ Function to create a Zadoff-Chu sequence of a given root index and size. """
        return np.exp((-1j * np.pi * root_index * np.arange(sequence_length) * (np.arange(sequence_length) + 1)) / sequence_length)

    def create_flc_frame(self, sequence):
        """ Function to create a transmission frame used for the Fixed-Lag-Correlation. """
        return np.concatenate((sequence, sequence))

    def create_indexed_flc_frame(self, sequence, antenna_index):
        """ Function to create a transmission frame used for the Fixed-Lag-Correlation. """
        # Convention: Antenna indices start with 1.
        antenna_index = antenna_index + 1
        #print(sequence[: 10])
        frame = np.concatenate((sequence, sequence))
        # The training data blocks are transformed to SM symbols for each antenna.
        # NON-GENERIC!
        complete_frame = np.zeros(frame.size * self.n_t, dtype=complex)
        for index, symbol in enumerate(frame):
            sm_symbol = [0] * (antenna_index - 1) + [symbol] + [0] * (self.n_t - antenna_index)
            for _ in range(0, self.n_t):
                complete_frame[index * self.n_t + _] = sm_symbol[_]
        return complete_frame

    def create_flc_prime(self, sequence):
        """ Function to create a frame used by the Fixed-Lag-Correlation at the receiver. """
        # Sequence should be a one-dimensional Numpy array.
        return np.roll(sequence, int(np.ceil(sequence.size / 2)))

    def estimate_frequency_offset(self, signal):
        """ Function to estimate the frequency offset for the proposed estimation scheme. """
        mid = int(self.frame_length * self.sps / 2)
        # Use the first and third quarter for the phase comparison.
        points = signal[: int(mid / 2)] * np.conj(signal[: + mid : int(mid / 2) + mid])
        phase_difference = -np.angle(points)
        return (np.mean(phase_difference) * self.sample_rate / (2 * np.pi * mid))

    def estimate_phase_offset(self, point):
        """ Function to estimate the phase offset for the proposed estimation scheme. """
        # The phase of the correlation maximum is the phase-offset.
        return np.angle(point)

    def estimate_frame(self, signal):
        """ Function to synchronize a signal in the time domain. """
        sum_energy = []
        for frame in range(0, self.sps):
            # The signal is divided in N frames for each of the N samples per symbol.
            sum_energy.append((np.sum(np.absolute(signal[frame][:]**2)), frame))
        # Use the frame with the maximum energy.
        samples_to_use = max(sum_energy)[1]
        return samples_to_use

    def sync_frame_offset(self, signal, samples_to_use):
        """ Function to synchronize a signal with a given frame offset. """
        frame_to_use = np.zeros((int(signal.size / self.sps)), dtype=complex)
        for index in range(0, int(signal.size / self.sps)):
            frame_to_use[index] = signal[index * self.sps + samples_to_use]
        return frame_to_use

    def sync_frequency_offset(self, signal, frequency_offset):
        """ Function to synchronize a signal with a given frequency offset. """
        for index, element in enumerate(signal):
                signal[index] = element * np.exp(-2j * np.pi * frequency_offset * index / self.sample_rate)
        return signal

    def sync_phase_offset(self, signal, phase_offset):
        """ Function to synchronize a signal with a given phase offset. """
        return signal * np.exp(-1j * phase_offset)

    def synchronize(self, signal, c_prime, sps):
        """ Concatenation of the synchronization methods. """
        # Synchronize frequency-offset first.
        frequency_offset = self.estimate_frequency_offset(signal, sps)
        signal = self.synchronize_phase_offset(signal, frequency_offset)
        # Correlate
        signal = np.correlate(signal, c_prime, mode='full')
        return

    def extract_channel(self, correlated_channel_responses):
        """ Function to estimate a channel based on a correlation representing a channel response. """
        channel_matrix = np.array([])
        sub_matrices = dict()
        channel_list = []
        multipaths = []
        for index, antenna_response in enumerate(correlated_channel_responses):
            multipaths.append([])
            best_path = max(antenna_response)
            # Threshold depends on sequence length and noise.
            threshold = 0.35 * best_path
            for val in antenna_response:
                if val > threshold:
                    multipaths[index].append(val / best_path)
        # DIFFERENT NUMBER OF MULTI-PATHS FOR DIFFERENT N_t NOT TAKEN INTO ACCOUNT.
        nb_multipaths = int(np.ceil(len(multipaths[0]) / self.n_r))
        # Create list of lists: Each sub-list represents a multi-path.
        for path in range(0, nb_multipaths):
            channel_list.append([])
            # Create a sub-list for each row.
            for row in range(0, self.n_r):
                channel_list[path].append([])
        # Insert the values into the list of lists of lists.
        for multipath in multipaths:
            for index, column in enumerate(split_list(multipath, self.n_r)):
                for row in range(0, self.n_r):
                    channel_list[index][row].append(column[row])
        # Create a channel matrix based on the multi-paths.
        for _ in range(0, nb_multipaths):
            # Number of rows and columns of each sub-matrix is N_r and N_t.
            sub_matrices[_] = np.array(channel_list[_])
        # Create 4-dimensional matrix using the sub-matrices.
        nb_rows = self.frame_length + nb_multipaths - 1
        nb_columns = self.frame_length
        channel_matrix = np.zeros((nb_rows, self.n_r, nb_columns, self.n_t),
                                  dtype=sub_matrices[0].dtype)
        for index, sub_matrix in sub_matrices.items():
            for element in range(nb_columns):
                channel_matrix[index + element, :, element, :] = sub_matrix
        # Flatten the 4-dimensional matrix.
        channel_matrix.shape = (nb_rows * self.n_r, nb_columns * self.n_t)
        #print(nb_multipaths)
        return channel_matrix

    def extract_channel_response(self, channel_responses, response_to_use, strongest_path):
        """ Extract the channel impulse response vector corresponding to a transmit antenna. """
        #n_r = channel_responses.shape[0]
        # Extract actual multi-paths for each reception antenna.
        # Create list of lists of lists indicating the value and position of the multi-path.
        multipaths = []
        counter = []
        for ra in range(0, self.n_r):
            antenna_response = channel_responses[ra][response_to_use]
            multipaths.append([])
            counter.append(0)
            #best_path = max(antenna_response)
            threshold = 0.1 * strongest_path
            for index, val in enumerate(antenna_response):
                abs_val = np.absolute(val)
                if abs_val > threshold:
                    multipaths[ra].append([val / strongest_path, index])
                    counter[ra] += 1
            # ADD EXCEPTION IF NO MULTIPATH HAS BEEN FOUND!
            fastest_path = multipaths[ra][0][1]
            for path in multipaths[ra]:
                path[1] = path[1] - fastest_path
                #print(path[1])
        nb_multipaths = max(counter)
        #print(nb_multipaths)
        #print(multipaths[0][: 10])
        # Vectors have to be of size ((K + P - 1) * N_r,).
        channel_vector = np.zeros((self.n_r * (self.frame_length + nb_multipaths - 1)), dtype=complex)
        for ra in range(0, self.n_r):
            for index, val in enumerate(multipaths[ra]):
                channel_vector[index * self.n_r + ra] = val[0]
        return channel_vector

    def recreate_channel(self, correlated_channel_responses):
        """ Function to estimate a channel based on a correlation representing a channel response. """
        # Different reception antennas can have a different amount of multi-paths.
        max_multipaths = max([response.size for response in correlated_channel_responses])
        for index, response in enumerate(correlated_channel_responses):
            while correlated_channel_responses[index].size < max_multipaths:
                correlated_channel_responses[index] = np.concatenate((correlated_channel_responses[index], np.zeros(1)))
        channel_matrix = np.zeros((max_multipaths, self.frame_length * len(correlated_channel_responses)), dtype=complex)
        # Fill first column of Block-Toeplitz-Matrix.
        for index, column in enumerate(correlated_channel_responses):
            #print(column.size)
            channel_matrix[:, index] = column
        # Fill the following columns with a rotation of the first.
        for symbol in range(1, self.frame_length):
            for index, column in enumerate(correlated_channel_responses):
                channel_matrix[:, len(correlated_channel_responses) * symbol + index] = np.roll(column, self.n_r * symbol)
        return channel_matrix
