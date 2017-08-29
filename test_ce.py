# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Test of the implementation of the channel estimation scheme. """


import numpy as np
import matplotlib.pyplot as plt
from sc_sm_transceiver import ChannelEstimator


def cconv(x, y):
    """Calculate the circular convolution of 1-D input numpy arrays using DFT
    """
    return np.fft.ifft(np.fft.fft(x)*np.fft.fft(y))

def ccorr(x, y):
    """Calculate the circular correlation of 1-D input numpy arrays using DFT
    """
    return np.fft.ifft(np.fft.fft(x)*np.fft.fft(y).conj())


def main():
    ce = ChannelEstimator((2,1))
    seq =ce.generate_gold_sequence(10)
    print(len(seq[-1]))

    plt.figure()
    plt.subplot(2,2,1)
    plt.title('Autocorrelation gold[0]')
    g0 = np.where(seq[0], 1.0, -1.0)
    #g0 = seq[0]
    #print(len(g0))
    #print(int((len(g0)/2-1)))
    plt.plot((np.roll(ccorr(g0, g0).real, int(len(g0)/2-1))))
    plt.subplot(2,2,2)
    plt.title('Autocorrelation gold[30]')
    g30 = np.where(seq[-1], 1.0, -1.0)
    plt.plot((np.roll(ccorr(g30, g30).real, int(len(g30)/2-1))))
    plt.subplot(2,2,3)
    plt.title('Crosscorrelation gold[0] gold[1]')
    g1 = np.where(seq[1], 1.0, -1.0)
    plt.plot((np.roll(ccorr(g0, g1).real, int(len(g0)/2-1))))
    plt.subplot(2,2,4)
    plt.title('Crosscorrelation gold[0] gold[1024]')
    plt.plot((np.roll(ccorr(g0, g30).real, int(len(g0)/2-1))))
    plt.show()

if __name__ == '__main__':
    main()
