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
import matplotlib.pyplot as plt
from modulation import BPSK as bpsk
#from channel import MIMOChannel as h
from channel import HardCodedChannel as h
from sc_sm_transceiver import Transceiver as tr
from sc_sm_transceiver import LSSDetector as det
from sc_sm_transceiver import ChannelEstimator as ce


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
    channel_estimator = ce(setup, k)

    # LOOP FOR TESTING PURPOSES.
    rounds = 1
    # BER is measured for the following SNRs.
    steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    #steps = [20]
    # The resulting BER values are stored in a list.
    points = []
    for step in steps:
        start = time.time()
        count = 0
        data_rounds = 10
        # Set channel parameters.
        channel.set_snr(step)
        for _ in range(0, rounds):
            bit_list = []
            rx_frames = []
            # TRAINING: Send training frames and estimate the channel.
            seq_order = 8
            seq_len = 2**seq_order - 1
            seq = channel_estimator.generate_gold_sequence(seq_order)
            training_blocks = transceiver.training_data_to_blocks(seq)
            symbols = modulation.modulate(training_blocks)
            tx_frames = transceiver.training_symbols_to_frames(symbols)
            channel.create_channel_matrix()
            for tx_frame in tx_frames:
                rx_frame = channel.apply_channel(tx_frame)
                # Detect with predefined channel estimation.
                #detected_frame = detector.detect(k,
                 #                                transceiver.get_symbol_list(),
                 #                                channel.get_ce_error_matrix(-10),
                 #                                rx_frame)
                #detected_frame = [symbol for sm_symbol in detected_frame for symbol in sm_symbol]
                #print(detected_frame)
                # Find where the modulated symbols are in the SM symbol and find antenna index.
                #modulated_symbol_list = []
                #for sm_symbol in detected_frame:
                    #modulated_symbol_list.append(transceiver.sm_demodulation(sm_symbol))
                #for symbol in modulated_symbol_list:
                    # Add the antenna index information bits.
                    # NOT NEEDED FOR CHANNEL ESTIMATION.
                    #bit_list.append(symbol[0])
                    # Add the information bits of the modulated signal.
                    #bit_list.append(modulation.demodulate(symbol[1]))
                rx_frame.flatten()
                rx_frames += [symbol for sub_list in rx_frame.tolist() for symbol in sub_list]
            #print(bit_list)
            channel_response = []
            for index, frame in enumerate(tx_frames):
                tx_frames[index] = frame.flatten().tolist()
            flat_tx = [symbol for sub_list in tx_frames for symbol in sub_list]
            nb_frames = int(np.ceil(seq_len / k))
            for antenna in range(0, setup[0]):
                #gold = np.where(seq[antenna + 2], 1.0, -1.0)
                #rx_gold = np.where(bit_list[antenna * seq_len : antenna * seq_len + seq_len], 1, 0)
                # EDGE CASE: We needed an additional frame for a sequence. Test w/ modulo?
                gold = flat_tx[antenna * setup[0] * seq_len : antenna * setup[0] * seq_len + setup[0] * seq_len]
                # If we have multiple frames, we have additional symbols for each frame.
                #rx_gold = rx_frames[antenna * (k + p - 1) * setup[0] * nb_frames : antenna * (k + p - 1) * setup[0] * int(np.ceil(seq_len / k)) + (k + p - 1) * setup[0] * int(np.ceil(seq_len / k))]
                rx_gold = rx_frames[antenna * setup[0] * seq_len : antenna * setup[0] * seq_len + setup[0] * seq_len]
                #print(str(len(gold)) + "  ~  " + str(len(rx_gold)))
                #print(gold[0 : 32])
                #print(rx_gold[0 : 64])
                #print(seq[antenna + 2][100 : 110])
                #print(rx_gold[100 : 110])
                #channel_response.append(np.roll(np.correlate(modulation.modulate(seq[antenna + 2]), rx_gold).real, int(len(seq[antenna + 2]) / 2 - 1)))
                channel_response.append(np.absolute(np.correlate(rx_gold, gold, mode='same').real))
                #channel_response.append(channel_estimator.ccorr(gold, gold[: 2048]))
                #(np.roll(ccorr(g0, g0).real, int(len(g0)/2-1))))
            #print(channel.get_channel_matrix())
            plt.figure()
            plt.subplot(2,2,1)
            plt.title('Cross-correlation: Antenna [0]')
            plt.plot(channel_response[0])
            #plt.axis([-100, 1100, -100, 1100])
            plt.subplot(2,2,2)
            plt.title('Cross-correlation: Antenna [1]')
            plt.plot(channel_response[1])
            #plt.axis([-100, 1100, -100, 1100])
            plt.subplot(2,2,3)
            plt.title('Cross-correlation: Antenna [0]')
            plt.plot(channel_response[0][250 : 265])
            plt.subplot(2,2,4)
            plt.title('Cross-correlation: Antenna [1]')
            plt.plot(channel_response[1][250 : 265])
            plt.show()
            est_channel = channel_estimator.extract_channel(channel_response)
            #est_channel = channel.get_ce_error_matrix(17)
            #print(est_channel)
            #print(est_channel.shape)

            # SENDING DATA: Send data using the estimated channel in the receiver.
            #data = ...
            #symbol_blocks = transceiver.data_to_blocks(data)
            #data_symbols = []
            #for symbol in symbol_blocks:
            #    data_symbols.append(transceiver.sm_modulation(bits_to_index(symbol[0]), modulation.modulate(symbol[1])))
            #data_frames = transceiver.data_symbols_to_frames(data_symbols)
            #data_frames = 1000
            for _ in range(0, data_rounds):
                data_frame = transceiver.transmit_frame(k, zp_len)
                rx_data_frame = channel.apply_channel(data_frame)

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
        ber = count / (k * setup[0] * rounds * data_rounds)
        # Measure the passed time.
        diff = time.time() - start
        # Write result in console.
        print(str(count) + " bits in " + str(data_rounds) + " data frames in " + str(rounds) + " tests were wrong!\n"
              + "> BER = " + str(ber) + "\n"
              + "> In " + str(diff) + " seconds.")
        # Append result to the list.
        points.append(ber)

    print(points)

if __name__ == '__main__':
    main()
