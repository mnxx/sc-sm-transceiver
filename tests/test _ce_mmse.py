# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Test of the implementation of the Single Carrier Spatial Modulation scheme. """


import time
import numpy as np
import  numpy.linalg as lin
from modulation import BPSK as bpsk
from channel import MIMOChannel as h
from sc_sm_transceiver import Transceiver as tr
from sc_sm_transceiver import LSSDetector as det

def estimate_channel():
    """ Reduced-Complexity Joint Channel Estimation and Data Detection. """
    # Pilot length is supposed to be of the same length as a normal frame.
    t = 4
    n_t = 2
    p = np.zeros((n_t, t))
    p_h = p.conj().T
    weigths = lin.multi_dot(lin.inv((lin.mullit_dot([p_h, rh, p]) + n_0 * n * eye)), p_h, rh)
    estimated_channel = rx_training.dot(weigths)
    return


def main():
    """ Main function. """
    # Initiate constants used in this test.
    # Antenna setup: Number of transmit antennas, number of reception antennas (N_t, N_r).
    setup = (2, 1)
    n_t = setup[0]
    n_r = setup[1]
    # Frame length of the transmission - K symbols for each transmission.
    k = 4
    # Number of multipath links.
    p = 3
    # Length of the Zero-Prefix.
    zp_len = p - 1
    # Signal to Noise Ratio.
    #snr = int(sys.argv[1])
    snr = 0
    # M algorithm: breadth-first search with M survivors.
    #m = int(sys.argv[2])
    m = 4

    # Use a linear modulation scheme.
    modulation = bpsk()

    # Initiate the transmission: create a transmission frame.
    transceiver = tr(setup, modulation.get_symbols())

    # Simulate the influence of a frequency-selective fading channel.
    channel = h(k, p, setup, snr)

    # Detect the sent frame using the M-algorithm based LSS-ML detector.
    detector = det(setup, m)

    # LOOP FOR TESTING PURPOSES.
    rounds = 1
    # BER is measured for the following SNRs.
    #steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    steps = [10]
    # The resulting BER values are stored in a list.
    points = []
    for step in steps:
        start = time.time()
        count = 0
        channel.set_snr(step - np.log2(setup[1]))
        for _ in range(0, rounds):
            #tx_frame = transceiver.create_transmission_frame(k)
            #tx_frame = transceiver.transmit_frame(k, zp_len)
            tr_frame1 = np.reshape([1, 0, 1, 0, 1, 0, 0, 1], (8, 1))
            #tr_frame2 = [[0, 1], [0, 1], [0, 1], [0, 1]]
            rx_frame = channel.apply_channel(tr_frame1)
            print(rx_frame)
            # Channel estimation.
            estimated_sub_channels = []
            for n in range(0, n_t):
                for index in range(0, p):
                    # For now: Every coefficient is the received value.
                    estimated_sub_channels[index].append(rx_frame[index + n])
            # Hard-coded for now:
            estimated_sub_channels[1][0] = estimated_sub_channels[1][0] - estimated_sub_channels[0][0]
            estimated_sub_channels[2][0] = estimated_sub_channels[1][0] - estimated_sub_channels[1][0] - estimated_sub_channels[0][0]
            estimated_sub_channels[1][1] = estimated_sub_channels[1][0] - estimated_sub_channels[2][0]
            estimated_sub_channels[0][1] = estimated_sub_channels[0][1] - estimated_sub_channels[1][0] - estimated_sub_channels[2][0]
            # Create 4-dimensional matrix using the sub-matrices.
            nb_rows = n_r * (k + p - 1)
            nb_columns = n_t * k
            estimated_channel = np.zeros((nb_rows, n_r, nb_columns, n_t),
                                              dtype=estimated_sub_channels[0].dtype)
            for index, sub_matrix in estimated_sub_channels.items():
                for element in range(nb_columns):
                    estimated_channel[index + element, :, element, :] = sub_matrix
            # Flatten the 4-dimensional matrix.
            estimated_channel.shape = (nb_rows * n_r, nb_columns * n_t)
            # Detect the sent frame using the M-algorithm based LSS-ML detector.
            detected_frame = detector.detect(k,
                                             transceiver.get_symbol_list(),
                                             #channel.get_channel_matrix(),
                                             estimated_channel,
                                             rx_frame)

            # Show the number of bit errors which occurred.
            tx_frame = tr_frame1.flatten()
            detected_frame = [symbol for sm_symbol in detected_frame for symbol in sm_symbol]
            for index in range(0, k * setup[0]):
                if tx_frame[index] != detected_frame[index]:
                    count += 1

        # BER calculation: Take combining gain into account, i.e. multiply by N_r.
        ber = count / (k * setup[0] * rounds)
        #ber = count / (k * setup[0] * rounds) * np.sqrt(2)
        #ber = count / (k * setup[0] * rounds)
        # Measure the passed time.
        diff = time.time() - start
        # Write result in console.
        print(str(count) + " bits in " + str(rounds) + " tests were wrong!\n"
              + "> BER = " + str(ber) + "\n"
              + "> In " + str(diff) + " seconds.")
        # Append result to the list.
        points.append(ber)

    print(points)

if __name__ == '__main__':
    main()
