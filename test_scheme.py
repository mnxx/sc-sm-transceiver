# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Test of the implementation of the Single Carrier Spatial Modulation scheme. """


import sys
import time
import numpy as np
from modulation import BPSK as bpsk
from channel import MIMOChannel as h
from sc_sm_transceiver import Transceiver as tr
from sc_sm_transceiver import LSSDetector as det


def main():
    """ Main function. """
    # Initiate constants used in this test.
    # Antenna setup: Number of transmit antennas, number of reception antennas (N_t, N_r).
    n_r = int(sys.argv[1])
    setup = (2, n_r)
    # Frame length of the transmission - K symbols for each transmission.
    #k = 4
    k = int(sys.argv[2])
    # Number of multipath links.
    p = 7
    # Signal to Noise Ratio.
    snr = 0
    # M algorithm: breadth-first search with M survivors.
    #m = 2
    m = int(sys.argv[3])

    # Use a linear modulation scheme.
    modulation = bpsk()

    # Initiate the transmission: create a transmission frame.
    transceiver = tr(setup, k, 1, modulation.get_symbols())

    # Simulate the influence of a frequency-selective fading channel.
    channel = h(k, p, setup, snr)

    # Detect the sent frame using the M-algorithm based LSS-ML detector.
    detector = det(setup, m)

    # Length of an SM symbol.
    symbol_len = 1 + int(np.log2(setup[0]))

    # LOOP FOR TESTING PURPOSES.
    #rounds = 100000
    rounds = int(sys.argv[4])
    # BER is measured for the following SNRs.
    #steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    #steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    steps = [50]
    # The resulting BER values are stored in a list.
    points = []
    start = time.time()
    for step in steps:
        #start = time.time()
        count = 0
        # Adapt for diversity gain of 3 dB for each additional receive antenna.
        #channel.set_snr(step - 3 * (setup[1] - 1))
        channel.set_snr(step - 3 * int(np.log2(setup[1])))
        #channel.set_snr(step)
        for _ in range(0, rounds):
            channel.create_channel_matrix()
            # Test with random data bits.
            data_frame = np.random.randint(0, 2, k * symbol_len).tolist()
            # Convert bits into spatial modulation symbols.
            blocks = transceiver.data_to_blocks(data_frame)
            modulated_symbols = np.array(modulation.modulate([block[1] for block in blocks]))
            tx_frame = transceiver.frame_sm_modulation(blocks, modulated_symbols)
            # Apply the channel to the sent signal.
            rx_frame = channel.apply_channel(tx_frame)
            # RECEPTION:
            # Perfect Channel State Information is assumed.
            # Detect the sent frame using the M-algorithm based LSS-ML detector.
            detected_frame = detector.detect(k,
                                             transceiver.get_symbol_list(),
                                             channel.get_channel_matrix(),
                                             rx_frame)

            # Show the number of bit errors which occurred.
            detected_bits = []
            for sm_symbol in detected_frame:
                temp = transceiver.sm_demodulation(sm_symbol)
                # Demodulation gives list with antenna information.
                antenna_info = temp[: -1]
                # Last element of the list is the linearly modulated symbol.
                modulated_info = modulation.demodulate(temp[-1])
                # Append demodulated bits to the result.
                detected_bits = detected_bits + antenna_info + modulated_info
            for index in range(0, k * symbol_len):
                if data_frame[index] != detected_bits[index]:
                    count += 1
        # BER calculation.
        ber = count / (k * symbol_len * rounds)
        points.append(ber)
    # Print the results for different SNRs.
    diff = time.time() - start
    print("***** N_r = " + str(n_r))
    print("***** K = " + str(k))
    print("***** M = " + str(m))
    print("***** ROUNDS = " + str(rounds))
    print("> In " + str(diff) + " seconds.")
    print(points)

if __name__ == '__main__':
    main()
