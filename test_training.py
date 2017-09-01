# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Test of the implementation of the Single Carrier Spatial Modulation scheme. """

import numpy as np
import time
from modulation import BPSK as bpsk
from channel import MIMOChannel as h
from sc_sm_transceiver import Transceiver as tr
from sc_sm_transceiver import LSSDetector as det
from sc_sm_transceiver import ChannelEstimator as ce


def main():
    """ Main function. """
    # Initiate constants used in this test.
    # Antenna setup: Number of transmit antennas, number of reception antennas (N_t, N_r).
    setup = (2, 2)
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
    m = 2

    # Use a linear modulation scheme.
    modulation = bpsk()

    # Initiate the transmission: create a transmission frame.
    transceiver = tr(setup, k, modulation.get_symbols())

    # Simulate the influence of a frequency-selective fading channel.
    channel = h(k, p, setup, snr)

    # Detect the sent frame using the M-algorithm based LSS-ML detector.
    detector = det(setup, m)

    # Simulate the channel estimation.
    channel_estimator = ce(setup)

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
        # Set channel parameters.
        channel.set_snr(step)
        for _ in range(0, rounds):
            # TRAINING: Send training frames and estimate the channel.
            seq = channel_estimator.generate_gold_sequence(5)
            training_blocks = transceiver.training_data_to_blocks(seq)
            symbols = modulation.modulate(training_blocks)
            tx_frames = transceiver.training_symbols_to_frames(symbols)
            channel.create_channel_matrix()
            for tx_frame in tx_frames:
                rx_frame = channel.apply_channel(tx_frame)
                # Detect with predefined channel estimation.
                detected_frame = detector.detect(k,
                                                 transceiver.get_symbol_list(),
                                                 channel.get_ce_error_matrix(0),
                                                 rx_frame)
                detected_frame = [symbol for sm_symbol in detected_frame for symbol in sm_symbol]
                #print(detected_frame)
                bits = modulation.demodulate(detected_frame)
                print(bits)
            break
            #est_channel = channel_estimator.find_channel_matrix(detected_frame, bits)
            est_channel = channel.get_ce_error_matrix(17)
            # SENDING DATA: Send data using the estimated channel in the receiver.
            data_frame = transceiver.transmit_frame(k, zp_len)
            rx_data_frame = channel.apply_channel_without_awgn(data_frame)

            # Detect the sent frame using the M-algorithm based LSS-ML detector.
            detected_data_frame = detector.detect(k,
                                                  transceiver.get_symbol_list(),
                                                  est_channel,
                                                  rx_data_frame)

            # Show the number of bit errors which occurred.
            tx_frame = data_frame.flatten()
            detected_data_frame = [symbol for sm_symbol in detected_data_frame for symbol in sm_symbol]
            for index in range(0, k * setup[0]):
                if data_frame[index] != detected_data_frame[index]:
                    count += 1

        # BER calculation: Take combining gain into account, i.e. multiply by N_r.
        ber = count / (k * setup[0] * rounds)
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
