# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Test of the implementation of the Single Carrier Spatial Modulation scheme. """


import sys
from modulation import BPSK as bpsk
from channel import MIMOChannel as h
from sc_sm_transceiver import Transceiver as tr
from sc_sm_transceiver import LSSDetector as det


def main():
    """ Main function. """
    # Initiate constants used in this test.
    # Antenna setup: Number of transmit antennas, number of reception antennas (N_t, N_r).
    setup = (2, 1)
    # Frame length of the transmission - K symbols for each transmission.
    k = 4
    # Number of multipath links.
    p = 3
    # Length of the Zero-Prefix.
    zp_len = p - 1
    # Signal to Noise Ratio.
    snr = int(sys.argv[1])
    # M algorithm: breadth-first search with M survivors.
    m = int(sys.argv[2])

    # Use a linear modulation scheme.
    modulation = bpsk()

    # Initiate the transmission: create a transmission frame.
    transceiver = tr(setup, modulation.get_symbols())
    #tx_frame = transceiver.create_transmission_frame(k)
    tx_frame = transceiver.transmit_frame(k, zp_len)

    # Simulate the influence of a frequency-selective fading channel.
    channel = h(k, p, setup, snr)
    rx_frame = channel.apply_channel(tx_frame)

    # Remove Zero-Pad from received frame.
    #rx_frame = transceiver.remove_zero_pad(rx_frame, zp_len)
    # Channel estimation: For now, perfect Channel State Information is assumed.

    # Detect the sent frame using the M-algorithm based LSS-ML detector.
    detector = det(setup, m)
    detected_frame = detector.detect(k,
                                     transceiver.get_symbol_list(),
                                     channel.get_channel_matrix(),
                                     rx_frame)

    # Show the number of bit errors which occurred.
    count = 0
    tx_frame = tx_frame.flatten()
    detected_frame = [symbol for sm_symbol in detected_frame for symbol in sm_symbol]
    for index in range(0, k * setup[0]):
        if tx_frame[index] != detected_frame[index]:
            count += 1
    print(count)

if __name__ == '__main__':
    main()
