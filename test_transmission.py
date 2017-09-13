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
    rounds = 1
    # BER is measured for the following SNRs.
    #steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    steps = [10]
    # The resulting BER values are stored in a list.
    points = []
    for step in steps:
        start = time.time()
        count = 0
        channel.set_snr(step)
        channel.create_channel_matrix()
        sps = 4
        for _ in range(0, rounds):
            c = channel_estimator.generate_zadoffchu_sequence(1, 512)
            tx_frame = transceiver.upsampling(sps, channel_estimator.create_flc_frame(c))
            #print(tx_frame.shape)
            c_prime = transceiver.upsampling(sps, channel_estimator.create_flc_prime(c))
            #print(c_prime.shape)
            pulse = transceiver.rrc_filter(1, 8, 4, tx_frame)
            print(pulse.shape)

            # FOR NOW: USE SISO CHANNEL MODEL
            h_s = np.zeros(15, dtype=complex)
            h_s[0] = 1
            h_s[int(sps/2)] = -2
            h_s[1*sps+1] = 1j
            h_s[2*sps+1] = .7+.7j
            h_s[3*sps+1] = -2-2j

            plt.figure()
            plt.subplot(211)
            plt.title('Pulse shaped Zadoff-Chu sequence.')
            plt.plot(pulse)
            plt.subplot(212)
            plt.title('Pulse shaped C\'.')
            #plt.plot(c_prime)
            plt.plot(tx_frame)
            #plt.show()

            #rx_frame = channel.apply_channel(pulse)
            rx_frame = np.convolve(pulse, h_s)
            print(rx_frame.shape)
            rn = rx_frame + (np.random.normal(0, np.sqrt(10**(-snr / 10) / 2),
                                              4141)
                + 1j * np.random.normal(0, np.sqrt(10**(-snr / 10) / 2),
                                        4141))
            print(rn.shape)
            y = transceiver.rrc_filter(1, 8, 4, rn)

            rcorr = np.correlate(rn, c_prime, mode='same').real
            ycorr = np.correlate(y, c_prime, mode='same').real

            plt.figure()
            plt.subplot(211)
            plt.title('Crosscorrelation: rx before filtering.')
            plt.plot(rcorr)
            plt.subplot(212)
            plt.title('Crosscorrelation: rx after filtering.')
            #plt.plot(c_prime)
            plt.plot(ycorr)
            #plt.show()

            # Downsampling & poly-phase correlation.
            yycorr = ycorr[: ycorr.size - np.mod(ycorr.size, sps)]
            yycorr = np.reshape(yycorr, (sps, int(yycorr.size / 4)), 'F')
            print(yycorr.shape)

            plt.figure()
            plt.title('Poly-Crosscorrelation.')
            plt.plot(yycorr[0], 'k-<')
            plt.plot(yycorr[1], 'b-<')
            plt.plot(yycorr[2], 'g-<')
            plt.plot(yycorr[3], 'r-<')
            plt.show()
            sum_energy = []
            for _ in range(0, sps):
                sum_energy.append(np.sum(np.absolute(yycorr[_][515 : 530])**2))
            print(sum_energy)
            break
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
