# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# GNU General Public License v3.0 is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

""" Implementation of different linear modulation schemes. """


class Modulation:
    """ Super class for the modulation schemes. """

    def __init__(self):
        self.symbol_list = []

    def get_symbols(self):
        """ Return the corresponding symbols of a modulation scheme. """
        return self.symbol_list


class BPSK(Modulation):
    """ Implementation of the BPSK modulation scheme. """

    def __init__(self):
        self.symbol_list = [[1], [-1]]

    def modulate(self, blocks):
        """ Modulate blocks of bits to symbols. """
        for index, block in enumerate(blocks):
            if block == [0]:
                blocks[index] = -1
            else:
                blocks[index] = 1
        return blocks

    def demodulate(self, symbol):
        """ Demodulate blocks of symbols to bits. """
        if symbol == -1:
            return 0
        else:
            return 1

    def demodulate_list(self, symbols):
        """ Demodulate blocks of symbols to bits. """
        for index, symbol in enumerate(symbols):
            if symbol == [-1]:
                symbols[index] = 0
            else:
                symbols[index] = 1
        return symbols
