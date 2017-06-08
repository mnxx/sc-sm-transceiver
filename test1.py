# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.signal as sig
import scipy.linalg as lin
import matplotlib.pyplot as plt

class QPSKModulation:
    """ Generate frames with QPSK modulated signals. """
    
    def __init__(self, symbols_per_frame):
        self.symbols_per_frame = symbols_per_frame
        # For now: create two frames - training sequence and data frame.
        qpsk_map = np.exp(1j * np.array([0, np.pi/2, np.pi, -np.pi/2]) + 1j * np.pi / 4)
        symbol_indices_train = np.random.randint(0, qpsk_map.size, symbols_per_frame)
        symbol_indices_data = np.random.randint(0, qpsk_map.size, symbols_per_frame)
        s_train = qpsk_map[symbol_indices_train]
        s_data = qpsk_map[symbol_indices_data]
        self.frameList = [s_train, s_data]

    def plotModulation(self):
        # Plot the data in the given frames.
        for index, frame in enumerate(self.frameList):
            plt.stem(frame.real[:20])
            plt.xlabel("Symbol index")
            plt.ylabel("Real part")
            plt.margins(0.1)
            plt.title("Frame" + str(index + 1))
            plt.show()
            

class Rest:

    def __init__(self):
        # add cyclic prefix
        N_CP = N // 4
        s = np.concatenate((s_train[-N_CP:], s_train, s_data[-N_CP:], s_data))
        
        plt.semilogy(np.linspace(-0.5, 0.5, s.size), np.fft.fftshift(np.abs(np.fft.fft(s))**2)); plt.xlabel("normalized frequency"); plt.ylabel("power"); plt.margins(y=0.1, x=0); plt.title("|S|^2"); plt.show();

        # convolve with the channel and add AWGN
        h = np.linspace(1, 0, N // 8) * np.exp(1j * 2 * np.pi * np.random.random(N // 8))
        #h = np.array([1,])
        n = np.sqrt(10**(-20/10) / 2) * (np.random.randn(s.size + h.size - 1) + 1j * np.random.randn(s.size + h.size - 1))
        r = np.convolve(h, s) + n

        # extract training sequence
        r_train = r[N_CP: N_CP + N]  # right after the CP. Valid values are [h.size, ..., N_CP].
        
        plt.stem(r_train.real[:20]); plt.xlabel("symbol index"); plt.ylabel("real part"); plt.margins(0.1); plt.title("r_train"); plt.show();

        # transform training sequence to frequency domain and estimate the channel with known training sequence
        R_train = np.fft.fft(r_train)
        H = R_train / np.fft.fft(s_train)
        #H = np.fft.fft(h, N)

        plt.semilogy(np.linspace(-0.5, 0.5, N), np.fft.fftshift(np.abs(R_train)**2)); plt.xlabel("normalized frequency"); plt.ylabel("power"); plt.margins(y=0.1, x=0); plt.title("|R_train|^2"); plt.show();
        plt.semilogy(np.linspace(-0.5, 0.5, N), np.fft.fftshift(np.abs(H)**2)); plt.xlabel("normalized frequency"); plt.ylabel("power"); plt.margins(y=0.1, x=0); plt.title("|H|^2"); plt.show();
        plt.stem(np.fft.ifft(H).real); plt.xlabel("sample index"); plt.ylabel("real part"); plt.margins(0.1); plt.plot(np.fft.ifft(H).imag); plt.title("h"); plt.show();
        
        # extract second frame and use Zero-Forcing equalization
        r_data = r[2 * N_CP + N: 2 * N_CP + 2 * N]
        R_data = np.fft.fft(r_data)
        R_data_eq = R_data / H
        r_data_eq = np.fft.ifft(R_data_eq)

        plt.stem(r_data.real[:40]); plt.xlabel("sample index"); plt.ylabel("real part"); plt.margins(y=0.1, x=0.1); plt.title("r_data"); plt.show();
        plt.semilogy(np.linspace(-0.5, 0.5, N), np.fft.fftshift(np.abs(R_data)**2)); plt.xlabel("normalized frequency"); plt.ylabel("power"); plt.margins(y=0.1, x=0); plt.title("|R_data|^2"); plt.show();
        plt.stem(r_data_eq.real[:40]); plt.xlabel("sample index"); plt.ylabel("real part"); plt.margins(y=0.1, x=0.1); plt.title("r_data_eq"); plt.show();
        plt.semilogy(np.linspace(-0.5, 0.5, N), np.fft.fftshift(np.abs(R_data_eq)**2)); plt.xlabel("normalized frequency"); plt.ylabel("power"); plt.margins(y=0.1, x=0); plt.title("|R_data_eq|^2"); plt.show();

        # compare symbol error rates
        def qpsk_detector(x):
            index = np.zeros(x.size, dtype=int)
            for i, sym in enumerate(x):
                if sym.real >= 0 and sym.imag >= 0:
                    index[i] = 0
                elif sym.real < 0 and sym.imag >= 0:
                    index[i] = 1
                elif sym.real < 0 and sym.imag < 0:
                    index[i] = 2
                else:
                    index[i] = 3
                    return index
                
                rx_symbol_indices_data_noeq = qpsk_detector(r_data)
                SER_noeq = sum(rx_symbol_indices_data_noeq != symbol_indices_data) / symbol_indices_data.size
                print("SER without equalization:", SER_noeq)

                rx_symbol_indices_data_eq = qpsk_detector(r_data_eq)
                SER_eq = sum(rx_symbol_indices_data_eq != symbol_indices_data) / symbol_indices_data.size
                print("SER with equalization:", SER_eq)

def main():
    # Number of symbols per frame.
    N = 256
    mod = QPSKModulation(N)
    mod.plotModulation()
    

if __name__ == '__main__':
    main()
