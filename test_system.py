# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Test of the complete implementation of the transmission scheme. """


import time
import numpy as np
import matplotlib.pyplot as plt
from modulation import BPSK as bpsk
from channel import HardCodedChannel as h
from sc_sm_transceiver import Transceiver as tr
from sc_sm_transceiver import LSSDetector as det
from sc_sm_transceiver import ChannelEstimator as ce


def main():
    """ Main function. """
    # Initiate constants used in the transmission.
    # Antenna setup: Number of transmit antennas, number of reception antennas (N_t, N_r).
    setup = (2, 2)
    # Frame length of the transmission - K symbols for each transmission.
    k = 1024
    # Number of multipath links.
    p = 3
    # Length of the Zero-Prefix.
    #zp_len = p - 1
    # Signal to Noise Ratio.
    #snr = int(sys.argv[1])
    snr = 0
    # M algorithm: breadth-first search with M survivors.
    #m = int(sys.argv[2])
    m = 2
    # Sample rate.
    sample_rate = 1e6
    # Samples per symbol.
    sps = 4
    # Filter span.
    span = 8

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
    channel_estimator = ce(setup, k, sample_rate)

    # Initiate possible offsets.
    # Frequency offset in Hz.
    f_off = 200
    estimated_f_off = 0
    # Phase offset in rad.
    phi_off = np.pi
    estimated_phi_off = 0
    # Frame offset in samples (compare with sps).
    frame_off = 0
    estimated_frame_off = 0

    # Loops.
    rounds = 1
    # BER is measured for the following SNRs.
    #steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    steps = [50]
    # The resulting BER values are stored in a list.
    points = []
    for step in steps:
        start = time.time()
        count = 0
        channel.set_snr(step)
        # TRAINING PHASE:
        channel_response_list = []
        for transmit_antenna in range(0, setup[0]):
            # TRANSMISSION:
            # Generate training frame for each transmit antenna.
            c = channel_estimator.generate_zadoffchu_sequence(1, int(k / 2))
            # Upsampling of the training frame as well as the corresponding prime frame.
            tx_frame = transceiver.upsampling(sps, channel_estimator.create_flc_frame(c))
            c_prime = transceiver.upsampling(sps, channel_estimator.create_flc_prime(c))
            # Pulse shape the training frame using an RRC-Filter.
            pulse = transceiver.rrc_filter(1, span, sps, tx_frame)
            # Apply a frequency offset.
            rx_frame = channel.apply_frequency_offset(pulse, sample_rate, f_off)
            # TO IMPROVE:
            # Create and apply channel matrix for this transmit antenna.
            channel.create_ta_channel_matrix(sps, transmit_antenna)
            rx_frame = channel.apply_ta_channel_without_awgn(rx_frame)
            #rx_frame = channel.apply_composed_channel(sps, pulse)
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
                # Matched filtering of the path from a transmit antenna to a receive antenna.
                y.append(transceiver.rrc_filter(1, span, sps, path))
                # Estimate the frequency offset.
                estimated_f_off = channel_estimator.estimate_frequency_offset(y[receive_antenna])
                # Get rid of frequency-offset.
                y[receive_antenna] = channel_estimator.sync_frequency_offset(y[receive_antenna],
                                                                             estimated_f_off)
                # Analyze the channel impulse response for the particular path by correlation.
                y[receive_antenna] = np.correlate(y[receive_antenna], c_prime, mode='full')
                zone =int((y[receive_antenna].size - np.mod(y[receive_antenna].size, sps)) / 2) - sps - 2
                y[receive_antenna] = y[receive_antenna][zone : zone + 10 * sps]
                y[receive_antenna] = np.reshape(y[receive_antenna], (sps, int(y[receive_antenna].size / sps)), 'F')
                """
                plt.figure()
                plt.title('Polyphase-cross-correlation: RA: ' + str(ra) + ', TA: ' + str(ta))
                plt.plot(np.abs(y[ra][0][:]), 'k-<')
                plt.plot(np.abs(y[ra][1][:]), 'b-<')
                plt.plot(np.abs(y[ra][2][:]), 'g-<')
                plt.plot(np.abs(y[ra][3][:]), 'r-<')
                plt.show()
                """
            # Estimate frame using the channel impulse response with the most energy.
            # TO IMPROVE: USE THE BEST OVERALL CHOICE.
            samples_to_use = channel_estimator.estimate_frame(y[0])
            # TO IMPROVE: CALCULATE STRONGEST PATH.
            strongest_path = 200
            channel_response_list.append(channel_estimator.extract_channel_response(y, samples_to_use, strongest_path))
        # Recreate the channel matrix from the channel impulse vector for each transmit antenna.
        # Channel matrix is 'deformed' because it includes the filters' impulse responses.
        estimated_channel = channel_estimator.recreate_channel(channel_response_list)

        # DATA TRANSMISSION PHASE:
        for _ in range(0, rounds):
            # TRANSMISSION:
            # Test with random data bits (ONE FRAME / PULSE SHAPING IS NEEDED FOR EACH FRAME).
            blocks = transceiver.data_to_blocks(np.random.randint(0, 2, 2048).tolist())
            modulated_symbols = modulation.modulate([block[1] for block in blocks])
            pulse = transceiver.rrc_filter(1, span, sps, transceiver.upsampling(sps, modulated_symbols))
            # Add antenna information.
            for index, block in enumerate(blocks):
                data_pulse = upsampled_sm_symbol_creation(block[0], pulse[index * sps : index * sps + sps], sps)
            """
            data_pulse = np.zeros(pulsed_info.size * 2, dtype=complex)
            for sample in range(0, 1024):
                index = sample * sps
                for _ in range(0, sps):
                    data_pulse[index * 2 + _] = pulsed_info[index + _]
            """
            # Apply a frequency offset.
            rx_data_pulse = channel.apply_frequency_offset(data_pulse, sample_rate, 200)
            # Apply fading channel.
            rx_data_pulse = channel.apply_composed_channel(sps, rx_data_pulse)
            # Apply AWGN.
            rx_data_pulse = rx_data_pulse + channel.add_awgn(rx_data_pulse.size)
            # RECEPTION:
            # Get rid of frequency-offset.
            rx_data_pulse = channel_estimator.sync_frequency_offset(rx_data_pulse,
                                                                    estimated_f_off)
            # Synchronize frames while downsampling.
            rx_data_frame = channel_estimator.sync_frame_offset(rx_data_pulse,
                                                                samples_to_use)
            # TO IMPROVE: APPLY PHASE OFFSET.
            rx_data_frame = channel_estimator.sync_phase_offset(rx_data_frame,
                                                                estimated_phi_off)
            # Matched filtering of the synchronized frame.
            rx_data_frame = transceiver.rrc_filter(1, span, sps, rx_data_frame)
            # Detect the sent frame using the M-algorithm based LSS-ML detector.
            detected_data_frame = detector.detect(k,
                                                  transceiver.get_symbol_list(),
                                                  estimated_channel,
                                                  rx_data_frame)
            # TO IMPROVE: DEMODULATE TO BITS.
            # Show the number of bit errors which occurred.
            detected_data_frame = [symbol for sm_symbol in detected_data_frame for symbol in sm_symbol]
            print(str(data_frame[: 20].tolist()) + " ~ ")
            print(str(detected_data_frame[: 20]))
            for index in range(0, k * setup[0]):
                if data_frame[index] != detected_data_frame[index]:
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
