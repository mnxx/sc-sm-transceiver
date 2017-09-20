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
    k_data = k
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

    # Simulate the channel estimation.
    channel_estimator = ce(setup, k)

    # LOOP FOR TESTING PURPOSES.
    rounds = 10
    # BER is measured for the following SNRs.
    #steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    steps = [20]
    # The resulting BER values are stored in a list.
    points = []
    for step in steps:
        start = time.time()
        count = 0
        channel.set_snr(step)
        # TRAINING FOR EACH TRANSMIT ANTENNA.
        channel_response_list = []
        samples_to_use = 0
        for ta in range(0, setup[0]):
            # TRAINING TRANSMISSIONS:
            sps = 4
            span = 8
            c = channel_estimator.generate_zadoffchu_sequence(1, int(k / 2))
            tx_frame = transceiver.upsampling(sps, channel_estimator.create_flc_frame(c))
            c_prime = transceiver.upsampling(sps, channel_estimator.create_flc_prime(c))
            pulse = transceiver.rrc_filter(1, span, sps, tx_frame)
            channel.create_ta_channel_matrix(sps * k + (sps * span) - 1, ta)

            # FOR NOW: USE SISO CHANNEL MODEL
            #h_s = np.zeros(100, dtype=complex)
            #h_s[0] = 1
            #h_s[3] = 1
            #h_s[19] = 0.5
            #h_s[99] = 0.2 + 0.2j
            #h_s[int(sps/2)] = -2
            #h_s[1*sps+1] = 1j
            #h_s[2*sps+1] = .7+.7j
            #h_s[3*sps+1] = -2-2j

            #plt.figure()
            #plt.subplot(211)
            #plt.title('Pulse shaped Zadoff-Chu sequence.')
            #plt.plot(pulse)
            #plt.subplot(212)
            #plt.title('Upsampled C\'.')
            #plt.plot(c_prime)
            #plt.plot(tx_frame)
            #plt.show()

            rx_frame = channel.apply_ta_channel_without_awgn(pulse)
            rn = rx_frame + channel.add_awgn(rx_frame.size)

            #rx_frame = np.convolve(pulse, h_s)
            #rx_frame = np.reshape(rx_frame, (rx_frame.size, 1))
            #print(channel.add_awgn(4141).shape)
            # Shape needed for correlation is (size,).
            #print(rn.shape)

            rn_split = np.reshape(rn, (setup[1], int(rx_frame.size / setup[1])), 'F')
            # Splitting for each reception antenna.
            y = []
            for ra in range(0, setup[1]):
                y.append(transceiver.rrc_filter(1, span, sps, rn_split[ra]))
                #y = transceiver.rrc_filter(1, 8, 4, rn)

                y[ra] = np.correlate(y[ra], c_prime, mode='same')

                #rcorr = np.correlate(rn, c_prime, mode='same').real
                #ycorr = np.correlate(y, c_prime, mode='same').real


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
                #plt.plot(ycorr1)
                #plt.subplot(212)
                #plt.title('Cross-correlation: rx after filtering - rx-antenna 2.')
                #plt.plot(c_prime)
                #plt.plot(ycorr2)

                # Downsampling & poly-phase correlation.
                #yycorr = ycorr[: ycorr.size - np.mod(ycorr.size, sps)]
                #yycorr = np.reshape(yycorr, (sps, int(yycorr.size / sps)), 'F')
                #pycorr = np.zeros((sps, int(yycorr.size / sps)))
                #for index, value in enumerate(yycorr):
                #    place = np.mod(index, sps)
                #    pycorr[place][int(index / 4)] = value
                #print(pycorr.shape)

                y[ra] = y[ra][: y[ra].size - np.mod(y[ra].size, sps)]
                y[ra] = np.reshape(y[ra], (sps, int(y[ra].size / sps)), 'F')

                plt.figure()
                plt.title('Polyphase-cross-correlation: RA: ' + str(ra) + ', TA: ' + str(ta))
                start = int(k / 2)
                stop = int(k / 2) + sps * p
                plt.plot(y[ra][0][start : stop].real, 'k-<')
                plt.plot(y[ra][1][start : stop].real, 'b-<')
                plt.plot(y[ra][2][start : stop].real, 'g-<')
                plt.plot(y[ra][3][start : stop].real, 'r-<')

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
                sum_energy.append((np.sum(np.absolute(y[0][_][int(k / 2) : int(k / 2) + sps * p])**2), _))
            samples_to_use = max(sum_energy)[1]
            #print(samples_to_use)
            # Extract a channel impulse response vector.
            strongest_path = max([channel_response.max() for channel_response in y])
            #print(strongest_path)
            channel_response_list.append(channel_estimator.extract_channel_response(y, samples_to_use, strongest_path))
        # Recreate the channel matrix from the channel impulse vector for each transmit antenna.
        # Channel matrix is 'deformed' because it includes the filters' impulse responses.
        estimated_channel = channel_estimator.recreate_channel(channel_response_list)
        #print(estimated_channel[: 8, : 2])
        # Recreate the channel matrix influencing the transmission.
        channel.create_channel_matrix_from_ta_vectors()
        # START TRANSMITTING DATA USING THE ESTIMATED CHANNEL.
        sps = 4
        span = 8
        #transceiver.set_symbols_per_frame(k_data)
        for _ in range(0, rounds):
            # Send random data for now.
            data_frame = transceiver.transmit_frame(k_data, zp_len)
            #print(data_frame.shape)
            tx_frame = transceiver.upsampling(sps, data_frame)
            pulse = transceiver.rrc_filter(1, span, sps, tx_frame)
            # Apply the channel on the pulse.
            # PULSE IS 2 TIMES TOO LARGE: WE HAVE 2 BITS PER SYMBOL IN THIS SIMULATION!
            rx_frame = channel.apply_ta_channel_without_awgn(pulse)
            rn = rx_frame + channel.add_awgn(rx_frame.size)
            # Detect the sent frame using the M-algorithm based LSS-ML detector.
            rn_split = np.reshape(rn, (setup[1], int(rx_frame.size / setup[1])), 'F')
            detected_data_frame = detector.detect(k_data,
                                                  transceiver.get_symbol_list(),
                                                  estimated_channel,
                                                  rn)

            # Show the number of bit errors which occurred.
            tx_frame = data_frame.flatten()
            detected_data_frame = [symbol for sm_symbol in detected_data_frame for symbol in sm_symbol]
            for index in range(0, k_data * setup[0]):
                if tx_frame[index] != detected_data_frame[index]:
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
