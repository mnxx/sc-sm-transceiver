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
        #sub_matrices = dict()
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
        self.create_channel_matrix()
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

    def apply_frequency_offset(self, signal, samples, sample_rate, frequency_offset):
        """ Apply a frequency offset to a given signal. """
        for index, element in enumerate(signal):
            signal[index] = element * np.exp(1j * 2 * np.pi * frequency_offset * index / samples / sample_rate)
        return signal


class LTEChannel(MIMOChannel):
    """ Class implementing different LTE channel scenarios defined by 3GPP. """

    #def __init__(self)
    def create_channel_matrix(self):
        """ Create the corresponding Block-Toeplitz channel matrix. """
        nb_rows = self.frame_len + self.multipaths - 1
        nb_columns = self.frame_len
        # A dictionary containing the discrete delay and its corresponding power profile is used.
        profile = [(0, 0), (30, -1.5), (150, -1.4)]
        # Each sub-matrix has a dimension of N_r x N_t.
        sub_matrices = dict()
        for _ in range(0, self.multipaths):
            # Only a certain number of taps are used in the channel model.
            if _ > len(profile):
                break
            # Number of rows and columns of each sub-matrix is N_r and N_t.

        # Create 4-dimensional matrix using the sub-matrices.
        self.channel_matrix = np.zeros((nb_rows, self.n_r, nb_columns, self.n_t),
                                       dtype=sub_matrices[0].dtype)
        for index, sub_matrix in sub_matrices.items():
            for element in range(nb_columns):
                self.channel_matrix[index + element, :, element, :] = sub_matrix
        # Flatten the 4-dimensional matrix.
        self.channel_matrix.shape = (nb_rows * self.n_r, nb_columns * self.n_t)


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
