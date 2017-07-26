# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Implementation of different MIMO channels."""

class MIMOChannel:
    """ Super class for MIMO fading channels. Uniform frequency selective fading. """

    def __init__(self, frame_len, multipaths, antenna_setup, snr):
        self.frame_len = frame_len
        self.multipaths = multipaths
        self.n_t = antenna_setup[0]
        self.n_r = antenna_setup[1]
        self.snr = snr

    def create_channel_matrix(self):
        """ Create the corresponding Block-Toeplitz channel matrix. """
        nb_rows = self.frame_len + self.multipaths - 1
        nb_columns = self.frame_len
        # Each sub-matrix has a dimension of N_r x N_t.
        sub_matrices = dict()
        for _ in range(0, self.multipaths):
            # Number of rows and columns of each sub-matrix is N_r and N_t.
            sub_matrices[_] = (np.random.randn(self.n_r, self.n_t) / 2
                               + 1j * np.random.randn(self.n_r, self.n_t) / 2)
        # Create 4-dimensional matrix using the sub-matrices.
        channel_matrix = np.zeros((nb_rows, self.n_r, nb_columns, self.n_t),
                                  dtype=sub_matrices[0].dtype)
        for index, sub_matrix in sub_matrices.items():
            for element in range(nb_columns):
                channel_matrix[index + element, :, element, :] = sub_matrix
        # Flatten the 4-dimensional matrix.
        channel_matrix.shape = (nb_rows * self.n_r, nb_columns * self.n_t)
        # Return the resulting matrix.
        return channel_matrix

    def create_awgn_vector(self):
        """ Create an additive white Gaussian noise vector. """
        return (np.random.normal(0, np.sqrt(10**(-self.snr / 10) / 2), (self.frame_len, 1))
                + 1j * np.random.normal(0, np.sqrt(10**(-self.snr / 10) / 2), (self.frame_len, 1)))

    def apply_channel(self, signal_vector):
        """ Directly apply the effects of the frequency selective channel on a signal vector. """
        return self.create_channel_matrix().dot(signal_vector) + self.create_awgn_vector()
