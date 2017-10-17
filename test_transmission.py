# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Test of the implementation of the transmission scheme. """


import time
import numpy as np
import matplotlib.pyplot as plt
from modulation import BPSK as bpsk
from channel import HardCodedChannel as h
from sc_sm_transceiver import Transceiver as tr
from sc_sm_transceiver import LSSDetector as det
from sc_sm_transceiver import ChannelEstimator as ce


def main():
    """
    plt.figure()
    plt.subplot(2,2,1)
    plt.title('Autocorrelation gold[0]')
    g0 = np.where(seq[0], 1.0, -1.0)
    #g0 = seq[0]
    print(g0)
    #print(int((len(g0)/2-1)))
    plt.plot((np.roll(ccorr(g0, g0).real, int(len(g0)/2-1))))
    #plt.axis([-100, 1100, -100, 1100])
    plt.subplot(2,2,2)
    plt.title('Autocorrelation gold[30]')
    g30 = np.where(seq[-1], 1.0, -1.0)
    plt.plot((np.roll(ccorr(g30, g30).real, int(len(g30)/2-1))))
    #plt.axis([-100, 1100, -100, 1100])
    plt.subplot(2,2,3)
    plt.title('Crosscorrelation gold[0] gold[1]')
    g1 = np.where(seq[1], 1.0, -1.0)
    plt.plot((np.roll(ccorr(g0, g1).real, int(len(g0)/2-1))))
    #plt.axis([-100, 1100, -100, 1100])
    plt.subplot(2,2,4)
    plt.title('Crosscorrelation gold[0] gold[1024]')
    plt.plot((np.roll(ccorr(g0, g30).real, int(len(g0)/2-1))))
    #plt.axis([-100, 1100, -100, 1100])
    plt.show()
    """

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

    # Sample rate.
    sample_rate = 1e6

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

    # LOOP FOR TESTING PURPOSES.
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

        # TRAINING FOR EACH TRANSMIT ANTENNA.
        channel_response_list = []
        for ta in range(0, setup[0]):
            # TRAINING TRANSMISSIONS:
            sps = 4
            span = 8
            c = channel_estimator.generate_zadoffchu_sequence(1, int(k / 2))
            #tx_frame = transceiver.upsampling(sps, channel_estimator.create_flc_frame(c, ta))
            tx_frame = transceiver.upsampling(sps, channel_estimator.create_flc_frame(c))
            c_prime = transceiver.upsampling(sps, channel_estimator.create_flc_prime(c))
            pulse = transceiver.rrc_filter(1, span, sps, tx_frame)
            #pulse = tx_frame
            # Apply a frequency offset.
            rx_frame = channel.apply_frequency_offset(pulse, sample_rate, 200)
            #rx_frame = pulse
            channel.create_ta_channel_matrix(sps, ta)
            rx_frame = channel.apply_ta_channel_without_awgn(rx_frame)
            #rx_frame = channel.apply_composed_channel(sps, pulse)
            #print(rx_frame[12 : 24])
            #print(rx_frame[12 : 24])
            rn = rx_frame + channel.add_awgn(rx_frame.size)
            #rn = rx_frame

            #aka = np.ones((1))
            #aka_pulsed = transceiver.rrc_filter(1, span, sps, aka)
            #daka = transceiver.rrc_filter(1, span, sps, aka_pulsed)
            #print(daka)

            #plt.figure()
            #plt.subplot(211)
            #plt.title('Pulse shaped Zadoff-Chu sequence.')
            #plt.plot(daka, 'k-<')
            #plt.subplot(212)
            #plt.title('Upsampled C\'.')
            #plt.plot(c_prime)
            #plt.plot(tx_frame)
            #plt.show()

            #rx_frame = np.convolve(pulse, h_s)
            #rx_frame = np.reshape(rx_frame, (rx_frame.size, 1))
            #print(channel.add_awgn(4141).shape)
            # Shape needed for correlation is (size,).
            #print(rn[: 12])
            #rn_psplit = np.reshape(rn, (sps, int(rn.size / sps)))
            #rn_split = np.reshape(rn, (setup[1], int(rx_frame.size / setup[1])), 'F')
            #rn_first_split = np.reshape(rn, (setup[1], int(rx_frame.size / setup[1])), 'F')
            #rn_split = np.reshape(rn_first_split[0], (setup[1], int(rn_first_split[0].size / setup[1])), 'F')
            #print(rn_split[0][: 12])
            #print(rn_split[1][: 12])
            # Splitting for each reception antenna.
            #block_len = int(rn_split.shape[1] / setup[0])
            y = []
            for ra in range(0, setup[1]):
                path_rn = np.zeros((int(rn.size / setup[1])), dtype=complex)
                for index in range(0, int(path_rn.size / sps)):
                    position = (index * setup[1] + ra) * sps
                    path_rn[index * sps : index * sps + sps] = rn[position : position + sps]
                y.append(transceiver.rrc_filter(1, span, sps, path_rn))
                #y.append(rn_split[ra])
                #y = transceiver.rrc_filter(1, 8, 4, rn)

                plt.figure()
                mid = int(y[ra].size / 2)
                print(mid)
                #sc = np.correlate(y[ra][: mid], y[ra][mid :], 'full')
                #phase = np.arctan2(sc.imag, sc.real)
                plt.title('Phase of the received frame.')
                plt.plot(-np.angle(y[ra]), 'm-<')
                #plt.plot(phase, 'c-<')
                #pzone =int((y[ra].size - np.mod(y[ra].size, sps)) / 2) - sps
                #point = y[ra][1025] * np.conj(y[ra][pzone + 1025])
                #point = y[ra][1021] * np.conj(y[ra][3069])
                #point2 = y[ra][1021] * np.conj(y[ra][3077])
                more_points = y[ra][ : 1200] * np.conj(y[ra][ 2048 : 1200 + 2048])
                #more_points = np.correlate(y[ra][: 2048], y[ra][2048 :], 'same')
                #p_phase = (np.pi + np.arctan2(point.imag, point.real)) / 2 / np.pi / 2
                #p_phase2 = -np.angle(point2) / 2 / np.pi
                #x_point = np.mean(more_points)
                #print(x_point)
                more_phase = -np.angle(more_points)
                #x_phase = np.angle(x_point) / 2 / np.pi
                #print(more_phase[330 : 340])
                #print("#+* " + str(np.mean(more_phase)))
                #print("TOPS: " + str(p_phase * 1e5 * sps / 2 / np.pi / 10))
                #print("TOPS: " + str(point) + " *** " + str(point2) + " ~ " + str(p_phase)+ " *** " + str(p_phase2) + " --- " + str(np.mean(more_phase)))
                f_off = np.mean(more_phase) / (2 * k)  * sample_rate / 2 / np.pi
                print(f_off)
                #print(x_phase / (2 * k) * 1e5)
                #print(p_phase / 2 / mid  * 1e5 * k * sps / 2 / np.pi)
                #print(str(p_phase / (2 * k) * 1e5) + " ## " + str(p_phase2 / (2 * k) * 1e5) + " ## " + str(np.mean(more_phase) / (2 * k) * 1e5))

                # Get rid of frequency-offset.
                for index, element in enumerate(y[ra]):
                    y[ra][index] = element * np.exp(-2j * np.pi * f_off * index / sample_rate)

                y[ra] = np.correlate(y[ra], c_prime, mode='full')
                #y[ra] = np.correlate(c_prime, y[ra], mode='same')
                #print(y[ra][0 : 4])
                #plt.figure()
                #plt.subplot(211)
                #plt.title('Crosscorrelation: rx before filtering.')
                #plt.plot(rcorr)
                #plt.subplot(212)
                #plt.title('Crosscorrelation: rx after filtering.')
                #plt.plot(c_prime)
                #plt.plot(ycorr)
                #plt.show()

                #plt.figure()
                #plt.subplot(211)
                #plt.title('Cross-correlation: rx after filtering - rx-antenna 1.')
                #plt.plot(y[ra].real, 'c-<')
                #plt.plot(np.arctan2(y[ra].imag, y[ra].real), 'c-<')
                #plt.plot(np.fft.fftshift(np.fft.fft(y[ra])), 'm-<')

                #plt.subplot(212)
                #plt.title('Cross-correlation: rx after filtering - rx-antenna 2.')
                #plt.plot(c_prime)
                #plt.plot(y[1])

                # Downsampling & poly-phase correlation.
                #yycorr = ycorr[: ycorr.size - np.mod(ycorr.size, sps)]
                #yycorr = np.reshape(yycorr, (sps, int(yycorr.size / sps)), 'F')
                #pycorr = np.zeros((sps, int(yycorr.size / sps)))
                #for index, value in enumerate(yycorr):
                #    place = np.mod(index, sps)
                #    pycorr[place][int(index / 4)] = value
                #print(pycorr.shape)

                #zone = int(k * sps / 2)
                #y[ra] = y[ra][zone + 2 : zone + 100]
                #y[ra] = y[ra][zone : zone + 20]
                #y[ra] = y[ra][zone - sps - 1 : zone + 100]
                #y[ra] = y[ra][1000 : 1040]
                zone =int((y[ra].size - np.mod(y[ra].size, sps)) / 2) - sps - 2
                y[ra] = y[ra][zone : zone + 10 * sps]
                y[ra] = np.reshape(y[ra], (sps, int(y[ra].size / sps)), 'F')
                #zone = int(int(samples_len / sps) / 2)
                #for index in range(0, sps):
                #    y[ra][index] = y[ra][index][zone : zone + 10 * sps]
                #print(y[ra][0].size)
                plt.figure()
                plt.title('Polyphase-cross-correlation: RA: ' + str(ra) + ', TA: ' + str(ta))
                #start = int(k / 2) - sps
                #stop = int(k / 2) + sps
                plt.plot(np.abs(y[ra][0][:]), 'k-<')
                plt.plot(np.abs(y[ra][1][:]), 'b-<')
                plt.plot(np.abs(y[ra][2][:]), 'g-<')
                plt.plot(np.abs(y[ra][3][:]), 'r-<')

                #phase = np.arctan2(y[ra][0].imag, y[ra][0].real)
                #plt.figure()
                #plt.plot(phase, 'c-<')
                #plt.plot(np.fft.fft(y[ra][0]), 'm-<')
                #print(str(max(phase)) + " ~~~ " + str(phase[1]))

            #plt.figure()
            #plt.title('Poly-Crosscorrelation.')
            #plt.plot(pycorr[0][999 : 1080], 'k-<')
            #plt.plot(pycorr[1][999 : 1080], 'b-<')
            #plt.plot(pycorr[2][999 : 1080], 'g-<')
            #plt.plot(pycorr[3][999 : 1080], 'r-<')

            plt.show()

            sum_energy = []
            # Find sample moment with the maximum energy.
            for _ in range(0, sps):
                sum_energy.append((np.sum(np.absolute(y[0][_][:]**2)), _))
            samples_to_use = max(sum_energy)[1]
            #print("SAMPLES TO USE: " + str(samples_to_use))
            # Extract a channel impulse response vector.
            #strongest_path = max([channel_response.max() for channel_response in y])
            strongest_path = 200
            #print(strongest_path)
            #samples_to_use = 0
            channel_response_list.append(channel_estimator.extract_channel_response(y, samples_to_use, strongest_path))
        # Recreate the channel matrix from the channel impulse vector for each transmit antenna.
        # Channel matrix is 'deformed' because it includes the filters' impulse responses.
        estimated_channel = channel_estimator.recreate_channel(channel_response_list)
        #print(estimated_channel[: 8, : 2])
        #print(estimated_channel[-8 :, : 2])
        #print(estimated_channel.shape)
        # Recreate the channel matrix influencing the transmission.
        #channel.create_channel_matrix_from_ta_vectors(sps * k + ((sps * span) - 1) / setup[0])
        #exit()
        # START TRANSMITTING DATA USING THE ESTIMATED CHANNEL.
        #sps = 4
        #span = 8
        #transceiver.set_symbols_per_frame(k_data)
        for _ in range(0, rounds):
            # Send random data for now.
            #data_frame = transceiver.transmit_frame(k, zp_len)
            # Test with random data bits (ONE FRAME / PULSE SHAPING IS NEEDED FOR EACH FRAME).
            blocks = transceiver.data_to_blocks(np.random.randint(0, 2, 2048).tolist())
            modulated_symbols = modulation.modulate([block[1] for block in blocks])
            pulse = transceiver.rrc_filter(1, span, sps, transceiver.upsampling(sps, modulated_symbols))
            # Add antenna information.
            for index, block in enumerate(blocks):
                data_pulse = upsampled_sm_symbol_creation(block[0], pulse[index * sps : index * sps + sps], sps)
            #test = np.array([1, 0, 1, 0])
            #for l in range(0, 9):
            #    test = np.concatenate((test, test))
            #data_frame = test
            #print(data_frame.shape)
            info_bits = np.ones((1024))
            up_info = transceiver.upsampling(sps, info_bits)
            #tx_data_frame = transceiver.upsampling(sps, data_frame)
            #pulsed_info = transceiver.rrc_filter(1, span, sps, tx_data_frame)
            pulsed_info = transceiver.rrc_filter(1, span, sps, up_info)
            # ADD ANTENNA INFORMATION.
            data_pulse = np.zeros(pulsed_info.size * 2, dtype=complex)
            for sample in range(0, 1024):
                index = sample * sps
                for _ in range(0, sps):
                    data_pulse[index * 2 + _] = pulsed_info[index + _]
            #print(data_pulse[0 : 12])
            #pulsed_info = tx_data_frame
            #data_frame_split = np.reshape(data_frame, (setup[0], int(data_frame.size / setup[0])), 'F')
            #print(pulsed_info[: 4])
            #print(transceiver.rrc_filter(1, span, sps, transceiver.upsampling(sps, info_bits))[: 4])
            #rn_split = np.empty((setup[0], (sps * k + p - 1) * setup[1]), dtype=complex)
            #for ta in range(0, setup[0]):
                #tx_frame = transceiver.upsampling(sps, data_frame_split[ta])
                #pulse = transceiver.rrc_filter(1, span, sps, tx_frame)
                # Apply the channel on the pulse.
                #print(pulse.shape)
                #channel.create_ta_channel_matrix(sps * k, ta)
                #rx_frame = channel.apply_ta_channel_without_awgn(pulse)
                #rx_frame = np.convolve(pulse, channel.get_ta_channel(ta), 'same')
                #print(rx_frame.shape)
                #rn_split[ta] = rx_frame #+ channel.add_awgn(rx_frame.size)
                #print(rx_frame.shape)
            #rx_data_pulse = channel.apply_composed_channel(sps, pulsed_info)
            # Apply a frequency offset.
            rx_data_pulse = channel.apply_frequency_offset(data_pulse, sample_rate, 200)
            rx_data_pulse = channel.apply_composed_channel(sps, rx_data_pulse)
            rx_data_pulse = rx_data_pulse + channel.add_awgn(rx_data_pulse.size)
            #print(rx_pulse.shape)
            #rx_pulse = np.reshape(rx_pulse, (2, int(rx_pulse.size / 2)), 'F')
            #rx_pulse[0] = transceiver.rrc_filter(1, span, sps, rx_pulse[0])
            #rx_pulse[1] = transceiver.rrc_filter(1, span, sps, rx_pulse[1])
            #rx_frame = np.reshape(rx_pulse, (rx_pulse.size), 'F')
            #print(rx_data_pulse[0 : 12])
            #rx_filtered_frame = transceiver.rrc_filter(1, span, sps, rx_data_pulse)
            #rx_filtered_frame = rx_data_pulse
            #print(rx_filtered_frame[: 12])
            #print(rx_filtered_frame[0 : 4])
            #rx_data_frame = np.zeros((int(rx_filtered_frame.size / sps)), dtype=complex)
            # Get rid of frequency-offset.
            for index, element in enumerate(rx_data_pulse):
                rx_data_pulse[index] = element * np.exp(-2j * np.pi * f_off * index / sample_rate)
            rx_data_frame = np.zeros((int(rx_data_pulse.size / sps)), dtype=complex)
            #print(samples_to_use)
            for index in range(0, int(rx_data_pulse.size / sps)):
                rx_data_frame[index] = rx_data_pulse[index * sps + samples_to_use]
            #print(rx_data_frame[: 12])
            rx_data_frame = transceiver.rrc_filter(1, span, sps, rx_data_frame)
            #print(rx_data_frame[: 12])
            #test_rx = estimated_channel.dot(data_frame)
            #print(test_rx[: 4])
            # Detect the sent frame using the M-algorithm based LSS-ML detector.
            detected_data_frame = detector.detect(k,
                                                  transceiver.get_symbol_list(),
                                                  estimated_channel,
                                                  rx_data_frame)
                                                  #test_rx)

            # Show the number of bit errors which occurred.
            #tx_frame = data_frame.flatten()
            #tx_frame = data_frame
            #print(detected_data_frame[: 10])
            detected_data_frame = [symbol for sm_symbol in detected_data_frame for symbol in sm_symbol]
            print(str(data_frame[: 20].tolist()) + " ~ ")
            print(str(detected_data_frame[: 20]))
            for index in range(0, k * setup[0]):
                if data_frame[index] != detected_data_frame[index]:
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
