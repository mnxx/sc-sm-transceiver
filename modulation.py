# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Implementation of different linear modulation schemes. """


import numpy as np


class Modulation:
    """ Super class for the modulation schemes. """

    def __init__(self):
        self.symbol_list = []
        self._index = 0

    def get_symbols(self):
        """ Return the corresponding symbols of a modulation scheme. """
        return self.symbol_list

    def get_modulation_index(self):
        """ Return the corresponding modulation index of the modulation. """
        return self._index

    def modulate(self, blocks):
        """ Abstract modulation method. """
        raise NotImplementedError("Method implemented by specific linear modulation.")

    def demodulate(self, symbol):
        """ Abstract demodulation method. """
        raise NotImplementedError("Method implemented by specific linear modulation.")

    def demodulate_list(self, symbols):
        """ Demodulate blocks of symbols to bits. """
        bit_list = []
        for symbol in symbols:
            bit_list = bit_list + self.demodulate(symbol)
        return bit_list


class BPSK(Modulation):
    """ Implementation of the BPSK modulation scheme. """

    def __init__(self):
        self.symbol_list = [[1], [-1]]
        self._index = 1

    def modulate(self, blocks):
        """ Modulate blocks of bits to symbols. """
        try:
            for index, block in enumerate(blocks):
                if block == [0]:
                    blocks[index] = -1
                elif block == [1]:
                    blocks[index] = 1
                else:
                    raise Exception
        except Exception:
            print("Block did not match bit criteria: " + str(block))
        return blocks

    def demodulate(self, symbol):
        """ Demodulate symbol to bits. """
        try:
            if symbol == -1:
                return [0]
            elif symbol == 1:
                return [1]
            else:
                raise Exception
        except Exception:
            print("Symbol did not match BPSK-symbol criteria: " + str(symbol))


class QPSK(Modulation):
    """ Implementation of the QPSK modulation scheme. """

    def __init__(self):
        self.symbol_list = [[np.exp(1j * np.pi / 4)], [np.exp(3j * np.pi / 4)],
                            [np.exp(-3j * np.pi / 4)], [np.exp(-1j * np.pi / 4)]]
        self._index = 2

    def modulate(self, blocks):
        """ Modulate blocks of bits to symbols. """
        try:
            for index, block in enumerate(blocks):
                if block == [0, 0]:
                    blocks[index] = np.exp(1j * np.pi / 4)
                elif block == [0, 1]:
                    blocks[index] = np.exp(3j * np.pi / 4)
                elif block == [1, 1]:
                    blocks[index] = np.exp(-3j * np.pi / 4)
                elif block == [1, 0]:
                    blocks[index] = np.exp(-1j * np.pi / 4)
                else:
                    raise Exception
        except Exception:
            print("Block did not match bit criteria: " + str(block))
        return blocks

    def demodulate(self, symbol):
        """ Demodulate symbol to bits. """
        try:
            if symbol.real >= 0 and symbol.imag >= 0:
                return [0, 0]
            elif symbol.real < 0 and symbol.imag >= 0:
                return [0, 1]
            elif symbol.real < 0 and symbol.imag < 0:
                return [1, 1]
            elif symbol.real >= 0 and symbol.imag < 0:
                return [1, 0]
            else:
                raise Exception
        except Exception:
            print("Symbol did not match QPSK-symbol criteria: " + str(symbol))
