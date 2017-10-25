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
    print("***** N_r = " + str(n_r))
    setup = (2, n_r)
    # Frame length of the transmission - K symbols for each transmission.
    #k = 4
    k = int(sys.argv[2])
    print("***** K = " + str(k))
    # Number of multipath links.
    p = 3
    # Length of the Zero-Prefix.
    zp_len = p - 1
    # Signal to Noise Ratio.
    snr = 0
    # M algorithm: breadth-first search with M survivors.
    #m = 2
    m = int(sys.argv[3])
    print("***** M = " + str(m))

    # Use a linear modulation scheme.
    modulation = bpsk()

    # Initiate the transmission: create a transmission frame.
    transceiver = tr(setup, k, 1, modulation.get_symbols())

    # Simulate the influence of a frequency-selective fading channel.
    channel = h(k, p, setup, snr)

    # Detect the sent frame using the M-algorithm based LSS-ML detector.
    detector = det(setup, m)

    # LOOP FOR TESTING PURPOSES.
    #rounds = 100000
    rounds = int(sys.argv[4])
    print("***** ROUNDS = " + str(rounds))
    # BER is measured for the following SNRs.
    #steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    #steps = [4]
    # The resulting BER values are stored in a list.
    points = []
    for step in steps:
        start = time.time()
        count = 0
        # Adapt for diversity gain of 3 dB for each additional receive antenna.
        #channel.set_snr(step - 3 * (setup[1] - 1))
        channel.set_snr(step)
        #channel.create_channel_matrix()
        for _ in range(0, rounds):
            channel.create_channel_matrix()
            #tx_frame = transceiver.create_transmission_frame(k)
            #tx_frame = transceiver.transmit_frame(k, zp_len)
            # Test with random data bits.
            data_frame = np.random.randint(0, 2, k * setup[0]).tolist()
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
            #tx_frame = tx_frame.flatten()
            #detected_frame = [symbol for sm_symbol in detected_frame for symbol in sm_symbol]
            #for index in range(0, k * setup[0]):
            #    if tx_frame[index] != detected_frame[index]:
            #        count += 1
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
            for index in range(0, k * setup[0]):
                if data_frame[index] != detected_bits[index]:
                    count += 1
        # BER calculation.
        ber = count / (k * setup[0] * rounds)
        # Measure the passed time.
        diff = time.time() - start
        # Write result in console.
        print(str(count) + " bits in " + str(rounds) + " tests were wrong!\n"
              + "> BER = " + str(ber) + "\n"
              + "> In " + str(diff) + " seconds.")
        # Append result to the list.
        points.append(ber)
    # Print the results for different SNRs.
    print(points)

if __name__ == '__main__':
    main()
