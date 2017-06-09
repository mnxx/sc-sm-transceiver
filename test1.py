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
        self.symbol_indices_train = np.random.randint(0, qpsk_map.size, symbols_per_frame)
        self.symbol_indices_data = np.random.randint(0, qpsk_map.size, symbols_per_frame)
        s_train = qpsk_map[self.symbol_indices_train]
        s_data = qpsk_map[self.symbol_indices_data]
        self.frameList = [s_train, s_data]

    def getTrainingIndices(self):
        return self.symbol_indices_train

    def getDataIndices(self):
        return self.symbol_indices_data

    def plotModulation(self):
        # Plot the data in the given frames.
        for index, frame in enumerate(self.frameList):
            plt.stem(frame.real[:20])
            plt.xlabel("Symbol Index")
            plt.ylabel("Real Part")
            plt.margins(0.1)
            plt.title("Frame" + str(index + 1))
            plt.show()

    def addCyclicPrefix(self, cp_length):
        # Put the last cp_length bits of the individual frame in front of the frame.
        for index, frame in enumerate(self.frameList):
            frame = np.concatenate((frame[-cp_length:], frame))
            self.frameList[index] = frame
            #self.frameListCP.append(frame)
            
    def concatenateFrames(self):
        # Concatenate the frames to a single list.
        #if not self.frameListCP:
        return np.concatenate(self.frameList)

    def plotSignalEnergy(self):
        # Plot the energy of the signal by using concatenateFrames.
        signal = self.concatenateFrames()
        plt.semilogy(np.linspace(-0.5, 0.5, signal.size), np.fft.fftshift(np.abs(np.fft.fft(signal))**2))
        plt.xlabel("Normalized Frequency")
        plt.ylabel("Power")
        plt.margins(y=0.1, x=0)
        plt.title("|S|^2")
        plt.show()


class Channel:
    """ Simulate the effects of a fading channel. """

    def __init__(self, signal):
        self.signal = signal
        
    def convolveChannel(self, channel_length):
        # Convolve the signal with the channel.
        h = np.linspace(1, 0, channel_length) * np.exp(1j * 2 * np.pi * np.random.random(channel_length))
        self.signal = np.convolve(h, self.signal)

    def addAWGN(self, snr):
        # Apply Additive White Gaussian Noise.
        #n = np.sqrt(10**(-20/10) / 2) * (np.random.randn(s.size + h.size - 1) + 1j * np.random.randn(s.size + h.size - 1))
        n = np.sqrt(10**(-snr/10) / 2) * (np.random.randn(self.signal.size) + 1j * np.random.randn(self.signal.size))
        self.signal = self.signal + n


class Detector:
    """ Detect the sent symbols by synchronizating/equalizing as well as finding the correct antenna index. """

    def __init__(self, signal):
        self.signal = signal

    def extractTrainingSeq(self, symbols_per_frame, cp_length):
        # We expect the training sequence to be the first frame.
        trainingSeq = self.signal[cp_length : cp_length + symbols_per_frame]

        #plt.stem(trainingSeq[:20])
        #plt.xlabel("Symbol Index")
        #plt.ylabel("Real Part")
        #plt.margins(0.1)
        #plt.title("Received Training Sequence")
        #plt.show();

        self.signal = self.signal[cp_length + symbols_per_frame :]
        return trainingSeq

    def extractData(self, symbols_per_frame, cp_length):
        #
        frameList = []
        indexer = np.arange(symbols_per_frame)
        #print(self.signal.size)
        self.signal = self.signal[cp_length :]
        while self.signal.size > symbols_per_frame:
            frameList.append(self.signal[indexer])
            self.signal = self.signal[cp_length + symbols_per_frame :]
        return np.concatenate(frameList)
        
    def estimateChannel(self, txTrainingSeq, rxTrainingSeq):
        # Estimate H
        self.H = np.fft.fft(rxTrainingSeq) / np.fft.fft(txTrainingSeq)

    def frequencyDomainEq(self, cp_length, frameList):
        # Equalize in the frequency domain using a Zero-Forcing approach.
        #r_data = r[2 * N_CP + N: 2 * N_CP + 2 * N]
        #frame_data = signal
        rx_Data = np.fft.fft(frameList)
        rx_Data_Eq = rx_Data / self.H
        return np.fft.ifft(rx_Data_Eq)

    def qpsk_detector(self, x):
    # Detect the corresponding QPSK-symbol.     
        index = np.zeros(len(x), dtype=int)
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
             


             
class Rest:

    def __init__(self):
        
        # convolve with the channel and add AWGN
        #h = np.array([1,])
        r = np.convolve(h, s) + n

        # extract training sequence
        r_train = r[N_CP: N_CP + N]  # right after the CP. Valid values are [h.size, ..., N_CP].
        
        

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

       

def main():
    
    N = 256
    N_CP = 256 // 4
    SNR = 20
    CHANNEL_LEN = 256 // 8
    
    mod = QPSKModulation(N)
    #mod.plotModulation()

    mod.addCyclicPrefix(N_CP)
    
    signal = mod.concatenateFrames()
    #mod.plotSignalEnergy()

    channel = Channel(signal)
    channel.convolveChannel(CHANNEL_LEN)
    channel.addAWGN(SNR)

    detector = Detector(channel.signal)
    rxTrainingSeq = detector.extractTrainingSeq(N, N_CP)
    rxData = detector.extractData(N, N_CP)
    detector.estimateChannel(mod.getTrainingIndices(),rxTrainingSeq)
    rxDataEq = detector.frequencyDomainEq(N_CP, rxData)
    
    rx_symbol_indices_data_noeq = detector.qpsk_detector(rxData)
    SER_noeq = sum(rx_symbol_indices_data_noeq != mod.getDataIndices()) / len(mod.getDataIndices())
    print("SER without equalization:", SER_noeq)

    print(rx_symbol_indices_data_noeq)
    print(mod.getDataIndices())
    
    rx_symbol_indices_data_eq = detector.qpsk_detector(rxDataEq)
    SER_eq = sum(rx_symbol_indices_data_eq != mod.getDataIndices()) / len(mod.getDataIndices())
    print("SER with equalization:", SER_eq)
    

if __name__ == '__main__':
    main()
