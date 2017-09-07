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
from modulation import BPSK as bpsk
from channel import MIMOChannel as h
from sc_sm_transceiver import Transceiver as tr
from sc_sm_transceiver import LSSDetector as det


def main():
    """ Main function. """
    # Initiate constants used in this test.
    # Antenna setup: Number of transmit antennas, number of reception antennas (N_t, N_r).
    setup = (2, 2)
    # Frame length of the transmission - K symbols for each transmission.
    k = 1024
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
    transceiver = tr(setup, k, modulation.get_symbols())

    # Simulate the influence of a frequency-selective fading channel.
    channel = h(k, p, setup, snr)

    # Detect the sent frame using the M-algorithm based LSS-ML detector.
    detector = det(setup, m)

    # LOOP FOR TESTING PURPOSES.
    rounds = 10
    # BER is measured for the following SNRs.
    steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    #steps = [10]
    # The resulting BER values are stored in a list.
    points = []
    for step in steps:
        start = time.time()
        count = 0
        channel.set_snr(step)
        channel.create_channel_matrix()
        for _ in range(0, rounds):
            #tx_frame = transceiver.create_transmission_frame(k)
            tx_frame = transceiver.transmit_frame(k, zp_len)
            #rx_frame = channel.apply_channel_without_awgn(tx_frame)
            rx_frame = channel.apply_channel(tx_frame)

            # Remove Zero-Pad from received frame.
            #rx_frame = transceiver.remove_zero_pad(rx_frame, zp_len)
            # Channel estimation: For now, perfect Channel State Information is assumed.

            # Detect the sent frame using the M-algorithm based LSS-ML detector.
            detected_frame = detector.detect(k,
                                             transceiver.get_symbol_list(),
                                             channel.get_channel_matrix(),
                                             #channel.get_ce_error_matrix(10),
                                             rx_frame)

            # Show the number of bit errors which occurred.
            tx_frame = tx_frame.flatten()
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
