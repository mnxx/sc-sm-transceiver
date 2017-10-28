# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Implementation of different MIMO channels. """


import numpy as np
from itertools import takewhile


class MIMOChannel:
    """ Super class for MIMO fading channels. Uniform frequency selective fading is the default. """

    def __init__(self, frame_len, multipaths, antenna_setup, snr):
        self.frame_len = frame_len
        self.multipaths = multipaths
        self.n_t = antenna_setup[0]
        self.n_r = antenna_setup[1]
        self.snr = snr
        self.channel_matrix = np.array([])
        self.sub_matrices = dict()
        self.ta_channel_matrix = np.array([])

    def create_channel_matrix(self):
        """ Create the corresponding Block-Toeplitz channel matrix. """
        nb_rows = self.frame_len + self.multipaths - 1
        nb_columns = self.frame_len
        # Each sub-matrix has a dimension of N_r x N_t.
        for _ in range(0, self.multipaths):
            # Number of rows and columns of each sub-matrix is N_r and N_t.
            self.sub_matrices[_] = (np.random.randn(self.n_r, self.n_t)
                                    + 1j * np.random.randn(self.n_r, self.n_t)) / np.sqrt(2)
        # Create 4-dimensional matrix using the sub-matrices.
        self.channel_matrix = np.zeros((nb_rows, self.n_r, nb_columns, self.n_t),
                                       dtype=self.sub_matrices[0].dtype)
        for index, sub_matrix in self.sub_matrices.items():
            for element in range(nb_columns):
                self.channel_matrix[index + element, :, element, :] = sub_matrix
        # Flatten the 4-dimensional matrix.
        self.channel_matrix.shape = (nb_rows * self.n_r, nb_columns * self.n_t)

    def set_snr(self, new_snr):
        """ Set the SNR of the channel to a new value. """
        self.snr = new_snr

    def get_channel_matrix(self):
        """ Return the created channel matrix. Use for perfect Channel State Information. """
        return self.channel_matrix

    def create_awgn_vector(self):
        """ Create an additive white Gaussian noise vector. """
        return (np.random.normal(0, np.sqrt(10**(-self.snr / 10) / 2),
                                 ((self.frame_len + self.multipaths - 1) * self.n_r))
                + 1j * np.random.normal(0, np.sqrt(10**(-self.snr / 10) / 2),
                                        ((self.frame_len + self.multipaths - 1) * self.n_r)))

    def add_awgn(self, vector_len):
        """ Create an AWGN Vector of a given size. """
        return (np.random.normal(0, np.sqrt(10**(-self.snr / 10) / 2),
                                 vector_len)
                + 1j * np.random.normal(0, np.sqrt(10**(-self.snr / 10) / 2),
                                        vector_len))

    def apply_channel(self, signal_vector):
        """ Directly apply the effects of the frequency selective channel on a signal vector. """
        return (self.channel_matrix).dot(signal_vector) + self.create_awgn_vector()

    def apply_new_channel(self, signal_vector):
        """
        Directly apply the effects of the frequency selective channel on a signal vector.
        Used for single frame analysis, or coherence time is equal to frame duration.
        """
        self.create_channel_matrix()
        return (self.channel_matrix).dot(signal_vector) + self.create_awgn_vector()

    def apply_channel_without_awgn(self, signal_vector):
        """ Directly apply the frequency selective channel without AWGN on a signal vector. """
        #self.create_channel_matrix()
        return (self.channel_matrix).dot(signal_vector)

    def get_ce_error_matrix(self, ce_snr):
        """ Add Gaussian noise to channel coefficients to simulate channel-estimation errors. """
        ce_sub_matrices = dict()
        for _ in range(0, self.multipaths):
            #print(self.sub_matrices[_])
            ce_sub_matrices[_] = (self.sub_matrices[_]
                                  + np.random.normal(0, np.sqrt(10**(-ce_snr / 10) / 2),
                                                     (self.n_r, self.n_t))
                                  + 1j * np.random.normal(0, np.sqrt(10**(-ce_snr / 10) / 2),
                                                          (self.n_r, self.n_t)))
            #print("x")
            #print(ce_sub_matrices[_])
            #print("---")
        nb_rows = self.frame_len + self.multipaths - 1
        nb_columns = self.frame_len
        # Create 4-dimensional matrix using the sub-matrices.
        ce_channel_matrix = np.zeros((nb_rows, self.n_r, nb_columns, self.n_t),
                                     dtype=ce_sub_matrices[0].dtype)
        for index, sub_matrix in ce_sub_matrices.items():
            for element in range(nb_columns):
                ce_channel_matrix[index + element, :, element, :] = sub_matrix
        # Flatten the 4-dimensional matrix.
        ce_channel_matrix.shape = (nb_rows * self.n_r, nb_columns * self.n_t)
        return ce_channel_matrix

    def apply_phase_offset(self, signal, phase_offset):
        """ Apply a phase offset to a given signal. """
        return signal * np.exp(1j * phase_offset)

    def apply_frequency_offset(self, signal, sample_rate, frequency_offset):
        """ Apply a frequency offset to a given signal. """
        for index, element in enumerate(signal):
            signal[index] = element * np.exp(2j * np.pi * frequency_offset * index / sample_rate)
        return signal


class LTEChannel(MIMOChannel):
    """ Class implementing different LTE channel scenarios defined by 3GPP. """

    def create_EPA(self, sample_rate, sps):
        """ Create a matrix corresponding to the LTE Extended Pedestrian A Model. """
        # Channel model: tap delay in [10ns = 1e-8s], attenuation linear.
        #channel_lte_epa = np.zeros(42)
        channel_lte_epa = dict()
        channel_lte_epa[0] = 1
        channel_lte_epa[3] = 0.8
        channel_lte_epa[7] = 0.63
        channel_lte_epa[9] = 0.5
        channel_lte_epa[11] = 0.16
        channel_lte_epa[19] = 0.02
        channel_lte_epa[41] = 0.01
        # EPA: Last echo at 410ns.
        # Decide on the number of multipath links based on the symbol rate.
        last_sample = int(np.ceil(410 * 1e-9 * sample_rate))
        nb_multipaths = int(np.ceil(last_sample / sps))
        # Adapt the taps to the sample rate.
        for key in list(channel_lte_epa.keys()):
            tap = int((key * sample_rate / 1e8))
            if tap not in channel_lte_epa:
                channel_lte_epa[tap] = channel_lte_epa.pop(key)
            else:
                del channel_lte_epa[key]
        # Create the corresponding channel matrix.
        self.create_LTE_channel_matrix(channel_lte_epa, nb_multipaths, sps)

    def create_EVA(self, sample_rate, sps):
        """ Create a matrix corresponding to the LTE Extended Vehicular A Model. """
        # Channel model: tap delay in [10ns = 1e-8s], attenuation linear.
        channel_lte_eva = dict()
        channel_lte_eva[0] = 1
        channel_lte_eva[3] = 0.71
        channel_lte_eva[15] = 0.72
        channel_lte_eva[31] = 0.72
        channel_lte_eva[37] = 0.87
        channel_lte_eva[71] = 0.12
        channel_lte_eva[109] = 0.2
        channel_lte_eva[173] = 0.06
        channel_lte_eva[251] = 0.02
        # EVA: Last echo at 2520ns.
        # Decide on the number of multipath links based on the symbol rate.
        last_sample = int(np.ceil(2510 * 1e-9 * sample_rate))
        nb_multipaths = int(np.ceil(last_sample / sps))
        # Adapt the taps to the sample rate.
        for key in list(channel_lte_eva.keys()):
            tap = int((key * sample_rate / 1e8))
            if tap not in channel_lte_eva:
                channel_lte_eva[tap] = channel_lte_eva.pop(key)
            else:
                del channel_lte_eva[key]
        # Create the corresponding channel matrix.
        self.create_LTE_channel_matrix(channel_lte_eva, nb_multipaths, sps)

    def create_ETU(self, sample_rate, sps):
        """ Create a matrix corresponding to the LTE Extended Urban Model. """
        # Channel model: tap delay in [10ns = 1e-8s], attenuation linear.
        channel_lte_etu = dict()
        channel_lte_etu[0] = 0.8
        channel_lte_etu[5] = 0.8
        channel_lte_etu[12] = 0.8
        channel_lte_etu[20] = 1
        channel_lte_etu[23] = 1
        channel_lte_etu[50] = 1
        channel_lte_etu[160] = 0.5
        channel_lte_etu[230] = 0.32
        channel_lte_etu[500] = 0.2
        # ETU: Last echo at 5010ns.
        # Decide on the number of multipath links based on the symbol rate.
        last_sample = int(np.ceil(5000 * 1e-9 * sample_rate))
        nb_multipaths = int(np.ceil(last_sample / sps))
        # Adapt the taps to the sample rate.
        for key in list(channel_lte_etu.keys()):
            tap = int((key * sample_rate / 1e8))
            if tap not in channel_lte_etu:
                channel_lte_etu[tap] = channel_lte_etu.pop(key)
            else:
                del channel_lte_etu[key]
        # Create the corresponding channel matrix.
        self.create_LTE_channel_matrix(channel_lte_etu, nb_multipaths, sps)

    def create_LTE_channel_matrix(self, channel_dict, nb_multipaths, sps):
        """ Create a channel matrix based on an LTE channel model. """
        key_list = []
        # Assign the echos to the corresponding multipath sub-matrices.
        for _ in range(0, nb_multipaths):
            key_list.append([])
            for key in list(channel_dict.keys()):
                if key in range(_ * sps, _ * sps + sps):
                    key_list[_].append(key)
        #print(key_list)
        norm_factor = 1
        for _ in range(0, nb_multipaths):
            sub_norm_factor = 0
            # Number of rows and columns of each sub-matrix is sps * N_r and sps * N_t.
            self.sub_matrices[_] = np.zeros((sps * self.n_r, sps * self.n_t), dtype=complex)
            for key in key_list[_]:
                pos = np.mod(key, sps)
                for tx in range(0, self.n_t):
                    for rx in range(0, self.n_r):
                        self.sub_matrices[_][pos + rx * sps, tx * sps] = (np.random.normal(0, np.sqrt(channel_dict[key] / 2), 1)
                                                                          + 1j * np.random.normal(0, np.sqrt(channel_dict[key] / 2), 1))
                sub_norm_factor += channel_dict[key]
            # Find the strongest multipath link.
            if sub_norm_factor > norm_factor:
                norm_factor = sub_norm_factor
        # Normalize the channel powers according to the strongest multipath.
        for _ in range(0, nb_multipaths):
            self.sub_matrices[_]  = self.sub_matrices[_] / norm_factor
        # Create 4-dimensional matrix using the sub-matrices.
        nb_rows = self.frame_len + nb_multipaths - 1
        nb_columns = self.frame_len
        self.channel_matrix = np.zeros((nb_rows, sps * self.n_r, nb_columns, sps * self.n_t),
                                       dtype=self.sub_matrices[0].dtype)
        for index, sub_matrix in self.sub_matrices.items():
            for element in range(nb_columns):
                self.channel_matrix[index + element, :, element, :] = sub_matrix
        # Flatten the 4-dimensional matrix.
        self.channel_matrix.shape = (nb_rows * sps * self.n_r, nb_columns * sps * self.n_t)
        #print(self.channel_matrix[: 16, : 2])

class HardCodedChannel(MIMOChannel):
    """ Class defining hard coded channel scenarios to facilitate analyses. """

    def create_channel_matrix(self):
        """ Create the corresponding Block-Toeplitz channel matrix. """
        nb_rows = self.frame_len + self.multipaths - 1
        nb_columns = self.frame_len
        #for _ in range(0, self.multipaths):
            # Number of rows and columns of each sub-matrix is N_r and N_t.
            # Hard coded values: ONLY WORKS FOR 2x2 SCENARIO!
        self.sub_matrices[0] = np.array([[1.0, 0.7],
                                         [0.8, 0.6]])
        self.sub_matrices[1] = np.array([[0.4, 0.3],
                                         [0.3, 0.2]])
        self.sub_matrices[2] = np.array([[0.3, 0.3],
                                         [0.2, 0.2]])
        # Create 4-dimensional matrix using the sub-matrices.
        self.channel_matrix = np.zeros((nb_rows, self.n_r, nb_columns, self.n_t),
                                       dtype=self.sub_matrices[2].dtype)
        for index, sub_matrix in self.sub_matrices.items():
            for element in range(nb_columns):
                self.channel_matrix[index + element, :, element, :] = sub_matrix
        # Flatten the 4-dimensional matrix.
        self.channel_matrix.shape = (nb_rows * self.n_r, nb_columns * self.n_t)

    def create_ta_channel_matrix(self, sps, antenna):
        """ Create a shortened Block-Toeplitz channel matrix for the training case. """
        nb_rows = self.frame_len + self.multipaths - 1
        nb_columns = self.frame_len
        #for _ in range(0, self.multipaths):
            # Number of rows and columns of each sub-matrix is N_r and N_t.
            # Hard coded values: ONLY WORKS FOR 2x2 SCENARIO!
        s_sub_matrices = dict()
        if antenna == 0:
            s_sub_matrices[0] = np.array([[1.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.6, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0]])
            #self.sub_matrices[1] = np.array([[0.0],
            #                                 [0.0]])
            #self.sub_matrices[2] = np.array([[0.0],
            #                                 [0.0]])
            #self.sub_matrices[3] = np.array([[0.0],
            #                                 [0.0]])
            s_sub_matrices[1] = np.array([[0.3 + 0.2j, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.4 + 0.2j, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0]])
            #self.sub_matrices[5] = np.array([[0.0],
            #                                 [0.0]])
            #self.sub_matrices[6] = np.array([[0.0],
            #                                 [0.0]])
            #self.sub_matrices[7] = np.array([[0.0],
            #                                 [0.0]])
            s_sub_matrices[2] = np.array([[0.2 + 0.1j, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.3 + 0.2j, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0]])
            #self.sub_matrices[9] = np.array([[0.0],
            #                                 [0.0]])
            #self.sub_matrices[10] = np.array([[0.0],
            #                                 [0.0]])
            #self.sub_matrices[11] = np.array([[0.0],
            #                                 [0.0]])
        else:
            s_sub_matrices[0] = np.array([[0.7, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [1.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0]])
            #self.sub_matrices[1] = np.array([[0.0],
            #                                 [0.0]])
            #self.sub_matrices[2] = np.array([[0.0],
            #                                 [0.0]])
            #self.sub_matrices[3] = np.array([[0.0],
            #                                 [0.0]])
            s_sub_matrices[1] = np.array([[0.2j, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.5j, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0]])
            #self.sub_matrices[5] = np.array([[0.0],
            #                                 [0.0]])
            #self.sub_matrices[6] = np.array([[0.0],
            #                                 [0.0]])
            #self.sub_matrices[7] = np.array([[0.0],
            #                                 [0.0]])
            s_sub_matrices[2] = np.array([[0.15, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.25, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0],
                                          [0.0, 0, 0, 0]])
            #self.sub_matrices[9] = np.array([[0.0],
            #                                 [0.0]])
            #self.sub_matrices[10] = np.array([[0.0],
            #                                 [0.0]])
            #self.sub_matrices[11] = np.array([[0.0],
            #                                 [0.0]])
        # Create 4-dimensional matrix using the sub-matrices.
        self.ta_channel_matrix = np.zeros((nb_rows, self.n_r * sps, nb_columns, sps),
                                          dtype=complex)
        for index, sub_matrix in s_sub_matrices.items():
            for element in range(nb_columns):
                self.ta_channel_matrix[index + element, :, element, :] = sub_matrix
        # Flatten the 4-dimensional matrix.
        self.ta_channel_matrix.shape = (nb_rows * self.n_r * sps, nb_columns * sps)
        #print(self.ta_channel_matrix[: 8, : 2])
        # Save the channel vector of the transmit antenna in the channel matrix.
        #ta_channel_vector = np.vstack((self.sub_matrices.values()))
        #if self.channel_matrix.size == 0:
        #    self.channel_matrix = np.reshape(ta_channel_vector, (ta_channel_vector.size, 1))
        #else:
            #self.channel_matrix = np.concatenate((self.channel_matrix, self.ta_channel_vector), axis=1)
            #self.channel_matrix = np.reshape(self.channel_matrix, tuple(reversed(self.ta_channel_matrix.shape)), 'F')
        #    self.channel_matrix = np.hstack((self.channel_matrix, ta_channel_vector))


    def apply_composed_channel(self, sps, signal_vector):
        """ Create a shortened Block-Toeplitz channel matrix for the training case. """
        nb_rows = self.frame_len + self.multipaths - 1
        nb_columns = self.frame_len
        #for _ in range(0, self.multipaths):
            # Number of rows and columns of each sub-matrix is N_r and N_t.
            # Hard coded values: ONLY WORKS FOR 2x2 SCENARIO!
        s_sub_matrices = dict()
        s_sub_matrices[0] = np.array([[1.0, 0, 0, 0, 0.7, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0.6, 0, 0, 0, 1.0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]])
        #self.sub_matrices[1] = np.array([[0.0, 0.0],
        #                                 [0.0, 0.0]])
        #self.sub_matrices[2] = np.array([[0.0, 0.0],
        #                                 [0.0, 0.0]])
        #self.sub_matrices[3] = np.array([[0.0, 0.0],
        #                                 [0.0, 0.0]])
        s_sub_matrices[1] = np.array([[0.3 + 0.2j, 0, 0, 0, 0.2j, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0.4 + 0.2j, 0, 0, 0, 0.5j, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]])
        #self.sub_matrices[5] = np.array([[0.0, 0.0],
        #                                 [0.0, 0.0]])
        #self.sub_matrices[6] = np.array([[0.0, 0.0],
        #                                 [0.0, 0.0]])
        #self.sub_matrices[7] = np.array([[0.0, 0.0],
        #                                 [0.0, 0.0]])
        s_sub_matrices[2] = np.array([[0.2 + 0.1j, 0, 0, 0, 0.15, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0.3 + 0.2j, 0, 0, 0, 0.25, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]])
        #self.sub_matrices[9] = np.array([[0.0, 0.0],
        #                                 [0.0, 0.0]])
        #self.sub_matrices[10] = np.array([[0.0, 0.0],
        #                                  [0.0, 0.0]])
        #self.sub_matrices[11] = np.array([[0.0, 0.0],
        #                                  [0.0, 0.0]])
        # Create 4-dimensional matrix using the sub-matrices.
        c_channel_matrix = np.zeros((nb_rows, self.n_r * sps, nb_columns, self.n_t * sps),
                                          dtype=complex)
        for index, sub_matrix in s_sub_matrices.items():
            #print(" +++ " + str(index))
            for element in range(nb_columns):
                c_channel_matrix[index + element, :, element, :] = sub_matrix
        # Flatten the 4-dimensional matrix.
        c_channel_matrix.shape = (nb_rows * self.n_r * sps, nb_columns * self.n_t * sps)
        #print(c_channel_matrix[: 16, : 6])
        #print(c_channel_matrix.shape)
        return (c_channel_matrix).dot(signal_vector)

    def apply_ta_channel_without_awgn(self, signal_vector):
        """ Directly apply the frequency selective channel without AWGN on a signal vector. """
        return (self.ta_channel_matrix).dot(signal_vector)

    def create_channel_matrix_from_ta_vectors(self, nb_col):
        """ Create the complete channel matrix from a given set of transmit antenna channel vectors. """
        nb_rows = nb_col + self.multipaths - 1
        nb_columns = nb_col
        for _ in range(0, self.multipaths):
            # Number of rows and columns of each sub-matrix is N_r and N_t.
            self.sub_matrices[_] = self.channel_matrix[_ * self.n_r : _ * self.n_r + self.n_r, :  self.n_t]
            #print(self.sub_matrices[_])
        # Create 4-dimensional matrix using the sub-matrices.
        self.channel_matrix = np.zeros((nb_rows, self.n_r, nb_columns, self.n_t),
                                       dtype=self.sub_matrices[0].dtype)
        for index, sub_matrix in self.sub_matrices.items():
            for element in range(nb_columns):
                self.channel_matrix[index + element, :, element, :] = sub_matrix
        # Flatten the 4-dimensional matrix.
        self.channel_matrix.shape = (nb_rows * self.n_r, nb_columns * self.n_t)

    def apply_rx_channel_without_awgn(self, signal_vector):
        """ Directly apply the frequency selective channel without AWGN on a signal vector. """
        return (self.channel_matrix).dot(signal_vector)

    def get_ta_channel(self, antenna):
        """ Return the channel impulse response for a specific transmit antenna. """
        if antenna == 0:
            response = np.array([1.0,
                                 0.5,
                                 0.0,
                                 0.0,
                                 0.3,
                                 0.3])
        else:
            response = np.array([0.5,
                                 0.5,
                                 0.0,
                                 0.0,
                                 0.2,
                                 0.4])
        return response
