# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Test of the implementation of the transmission scheme sans pulse shaping filters. """


import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from modulation import BPSK as bpsk
#from channel import HardCodedChannel as h
from channel import CustomChannel as h
from sc_sm_transceiver import Transceiver as tr
from sc_sm_transceiver import LSSDetector as det
from sc_sm_transceiver import ChannelEstimator as ce


def main():
    """ Main function. """
    # Initiate constants used in the transmission.
    # Antenna setup: Number of transmit antennas, number of reception antennas (N_t, N_r).
    #n_r = int(sys.argv[1])
    setup = (2, 2)
    # Frame length of the transmission - K symbols for each transmission.
    #k = 1024
    k = int(sys.argv[1])
    # Number of multipath links.
    p = 3
    # Length of the Zero-Prefix.
    #zp_len = p - 1
    # Signal to Noise Ratio.
    #snr = int(sys.argv[1])
    snr = 0
    # M algorithm: breadth-first search with M survivors.
    #m = int(sys.argv[2])
    #m = 4
    m = int(sys.argv[2])
    # Sample rate.
    sample_rate = 1e7
    # Samples per symbol.
    sps = 4
    # Filter span.
    #span = 8

    # Initiate objects used in the transmission.
    # Use a linear modulation scheme.
    modulation = bpsk()
    # Initiate the transmission: create a transmission frame.
    transceiver = tr(setup, k, sample_rate, modulation.get_symbols())
    # Simulate the influence of a frequency-selective fading channel.
    channel = h(k, p, setup, snr)
    # Detect the sent frame using the M-algorithm based LSS-ML detector.
    detector = det(setup, m)
    # Simulate the channel estimation.
    channel_estimator = ce(setup, k, sample_rate, sps)

    # Initiate possible offsets.
    # Frequency offset in Hz.
    f_off = 0
    estimated_f_off = 0
    # Frame offset in samples (compare with samples per symbol).
    frame_off = 0
    estimated_frame_off = 0
    # Phase offset in rad.
    phi_off = np.pi

    # Channel choice for testing.
    channel_choice  = int(sys.argv[3])

    # Loops.
    nb_channels = 100
    #nb_channels = 1
    rounds = 1
    #rounds = 1
    # BER is measured for the following SNRs.
    steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    #steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    #steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    #steps = [90]
    # The resulting BER values are stored in a list.
    points = []
    start = time.time()
    for step in steps:
        count = 0
        # Adapt for diversity gain of 3 dB for each additional receive antenna.
        channel.set_snr(step - 3 * (setup[1] - 1))
        for realization  in range(0, nb_channels):
            # Create LTE channel model.
            if channel_choice == 1:
                channel.create_flat(sample_rate, sps)
            elif channel_choice == 2:
                channel.create_uniform(sample_rate, sps)
            elif channel_choice == 3:
                channel.create_linear_1(sample_rate, sps)
            elif channel_choice == 4:
                channel.create_linear_2(sample_rate, sps)
            elif channel_choice == 5:
                channel.create_EPA(sample_rate, sps)
            elif channel_choice == 6:
                channel.create_EVA(sample_rate, sps)
            else:
                channel.create_ETU(sample_rate, sps)
            # TRAINING PHASE:
            channel_response_list = []
            for transmit_antenna in range(0, setup[0]):
                # TRANSMISSION:
                # Generate training frame for each transmit antenna.
                c = channel_estimator.generate_zadoffchu_sequence(1, int(k / 2))
                # Upsampling of the training frame as well as the corresponding prime frame.
                #tx_frame = transceiver.upsampling(sps, channel_estimator.create_flc_frame(c))
                tx_frame = channel_estimator.create_flc_frame(c)
                #print(tx_frame[: 4])
                c_prime = transceiver.upsampling(sps, channel_estimator.create_flc_prime(c))
                if transmit_antenna == 0:
                    blocks = [[[0],[0]]] * k
                else:
                    blocks =  [[[1],[1]]] * k
                tx_frame = transceiver.frame_sm_modulation(blocks, tx_frame)
                upsampled_frame = transceiver.upsampling(sps, tx_frame)
                #print(upsampled_frame[: 16])
                # Apply a frequency offset.
                #rx_frame = channel.apply_frequency_offset(upsampled_frame, sample_rate, f_off)
                rx_frame = upsampled_frame
                # Create and apply channel matrix for this transmit antenna.
                rx_frame = channel.apply_channel_without_awgn(rx_frame)
                # Add AWGN to the signal.
                rx_frame = rx_frame + channel.add_awgn(rx_frame.size)
                # RECEPTION:
                y = []
                # Analyze received signal for each receive antenna.
                for receive_antenna in range(0, setup[1]):
                    # Split the result for each receive antenna.
                    path = np.zeros((int(rx_frame.size / setup[1])), dtype=complex)
                    for index in range(0, int(path.size / sps)):
                        position = (index * setup[1] + receive_antenna) * sps
                        path[index * sps : index * sps + sps] = rx_frame[position : position + sps]
                    y.append(path)
                    #print(path[: 16])
                    # Estimate the frequency offset.
                    #estimated_f_off = 2 * channel_estimator.estimate_frequency_offset(y[receive_antenna])
                    #print("f_off : " + str(estimated_f_off))
                    # Get rid of frequency-offset.
                    #y[receive_antenna] = channel_estimator.sync_frequency_offset(y[receive_antenna], estimated_f_off)
                    # Analyze the channel impulse response for the particular path by correlation.
                    y[receive_antenna] = np.correlate(y[receive_antenna], c_prime, mode='full')
                    y[receive_antenna] = y[receive_antenna] / np.sqrt(np.sum(np.abs(y[receive_antenna])**2)) * sps / setup[1]
                    """
                    plt.figure()
                    plt.title('Phase of FLC for link RA: ' + str(receive_antenna + 1) + ', TA: ' + str(transmit_antenna + 1))
                    #plt.plot(np.abs(y[receive_antenna]), 'm-<')
                    plt.plot(np.abs(y[receive_antenna]), 'm-<')
                    plt.show()
                    """
                    zone =int((y[receive_antenna].size - np.mod(y[receive_antenna].size, sps)) / 2) - 20 * sps - 3
                    y[receive_antenna] = y[receive_antenna][zone : zone + 50 * sps]
                    y[receive_antenna] = np.reshape(y[receive_antenna], (sps, int(y[receive_antenna].size / sps)), 'F')
                    """
                    plt.figure()
                    plt.title('Polyphase-cross-correlation: RA: ' + str(receive_antenna) + ', TA: ' + str(transmit_antenna))
                    plt.plot(np.abs(y[receive_antenna][0][:]), 'k-<')
                    plt.plot(np.abs(y[receive_antenna][1][:]), 'b-<')
                    plt.plot(np.abs(y[receive_antenna][2][:]), 'g-<')
                    plt.plot(np.abs(y[receive_antenna][3][:]), 'r-<')
                    plt.show()
                    """
                # Estimate frame using the channel impulse response with the most energy.
                # TO IMPROVE: USE THE BEST OVERALL CHOICE.
                samples_to_use = channel_estimator.estimate_frame(y[0])
                #print(samples_to_use)
                channel_response_list.append(channel_estimator.extract_channel_response(y, samples_to_use))
            # Recreate the channel matrix from the channel impulse vector for each transmit antenna.
            #estimated_channel = channel_estimator.recreate_channel(channel_response_list)
            estimated_channel = channel.recreate_channel_matrix(sps)
            #print(estimated_channel[: 8, : 2])
            # DATA TRANSMISSION PHASE:
            for _ in range(0, rounds):
                # TRANSMISSION:
                # Test with random data bits.
                data_frame = np.random.randint(0, 2, k * setup[0]).tolist()
                blocks = transceiver.data_to_blocks(data_frame)
                modulated_symbols = modulation.modulate([block[1] for block in blocks])
                # Add antenna information for channel simulation.
                tx_data_frame = transceiver.frame_sm_modulation(blocks, modulated_symbols)
                upsampled_data_frame = transceiver.upsampling(sps, tx_data_frame)
                # Apply a frequency offset on pulse.
                #rx_data_pulse = channel.apply_frequency_offset(upsampled_data_frame, sample_rate, f_off)
                rx_data_pulse = upsampled_data_frame
                # Apply fading channel.
                rx_data_pulse = channel.apply_channel_without_awgn(rx_data_pulse)
                # Apply AWGN.
                rx_data_pulse = rx_data_pulse + channel.add_awgn(rx_data_pulse.size)
                # RECEPTION:
                # Split into receive antennas.
                rx_data_frame = np.zeros((setup[1], int(rx_data_pulse.size / setup[1] / sps)), dtype=complex)
                # Synchronization and downsampling for each receive antenna.
                for receive_antenna in range(0, setup[1]):
                    data_path = np.zeros((int(rx_data_pulse.size / setup[1])), dtype=complex)
                    for index in range(0, int(data_path.size / sps)):
                        position = (index * setup[1] + receive_antenna) * sps
                        data_path[index * sps : index * sps + sps] = rx_data_pulse[position : position + sps]
                    # Get rid of frequency-offset.
                    #data_path = channel_estimator.sync_frequency_offset(data_path, estimated_f_off)
                    # Synchronize frames while downsampling.
                    rx_data_frame[receive_antenna] = channel_estimator.sync_frame_offset(data_path, samples_to_use)
                rx_data_frame = rx_data_frame.flatten('F')
                # Detect the sent frame using the M-algorithm based LSS-ML detector.
                detected_data_frame = detector.detect(k,
                                                      transceiver.get_symbol_list(),
                                                      estimated_channel,
                                                      rx_data_frame)
                # Show the number of bit errors which occurred.
                detected_bits = []
                for sm_symbol in detected_data_frame:
                    temp = transceiver.sm_demodulation(sm_symbol)
                    # Demodulation gives list with antenna information.
                    antenna_info = temp[: -1]
                    # Last element of the list is the linearly modulated symbol.
                    modulated_info = modulation.demodulate(temp[-1])
                    # Append demodulated bits to the result.
                    detected_bits = detected_bits + antenna_info + modulated_info
                #print(data_frame[-10 : -1])
                #print(detected_bits[-10 : -1])
                for index in range(0, k * setup[0]):
                    if data_frame[index] != detected_bits[index]:
                        count += 1
        # BER calculation.
        ber = count / (k * setup[0] * rounds * nb_channels)
        # Measure the passed time.
        #diff = time.time() - start
        # Write result in console.
        #print(str(count) + " bits in " + str(rounds) + " tests were wrong!\n"
        #      + "> BER = " + str(ber))
        #      + "> In " + str(diff) + " seconds.")
        # Append result to the list.
        points.append(ber)
    # Print the results for different SNRs.
    print("***** K = " + str(k))
    print("***** M = " + str(m))
    print("***** CHANNEL = " + str(channel_choice))
    diff = time.time() - start
    print("> In " + str(diff) + " seconds:")
    print(points)

if __name__ == '__main__':
    main()
