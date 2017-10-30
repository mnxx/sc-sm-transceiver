# Copyright 2017 Communications Engineering Lab - KIT.
# All Rights Reserved.
# Author: Manuel Roth.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the GNU General Public License v3.0 is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Create BER/SNR plots using a list of calculated points.

"""


import numpy as np
import matplotlib.pyplot as plt

# Flat-fading:
#flat2x2_1024_2 = [0.446484375, 0.3939453125, 0.385888671875, 0.33056640625, 0.2677734375, 0.32841796875, 0.0755859375, 0.04951171875, 0.039892578125, 0.0404296875, 0.0474609375]

flat2x2_1024_4 = [0.425341796875, 0.43642578125, 0.4, 0.342138671875, 0.339208984375, 0.235791015625, 0.198046875, 0.17451171875, 0.098828125, 0.05908203125, 0.04111328125, 0.015380859375, 0.0171875, 0.0056640625, 0.003759765625, 0.05]

unifrom2x2_1024_4 = [0.447119140625, 0.41669921875, 0.3720703125, 0.335693359375, 0.229150390625, 0.21328125, 0.1865234375, 0.1017578125, 0.11865234375, 0.037158203125, 0.007421875, 0.021044921875, 0.059521484375, 0.07470703125, 0.04296875, 0.06474609375]

linea1r2x2_1024_4 = [0.443408203125, 0.37197265625, 0.373486328125, 0.3427734375, 0.28408203125, 0.209521484375, 0.152001953125, 0.111376953125, 0.0791015625, 0.03408203125, 0.070751953125, 0.00849609375, 0.043017578125, 0.021044921875, 0.02880859375, 0.02412109375]

linea2r2x2_1024_4 = [0.43037109375, 0.40283203125, 0.398193359375, 0.312255859375, 0.300048828125, 0.1994140625, 0.18271484375, 0.149951171875, 0.118212890625, 0.087744140625, 0.053125, 0.020703125, 0.03603515625, 0.0017578125, 0.045458984375, 0.037841796875]

epa2x2_1024_4 = [0.446240234375, 0.416748046875, 0.40009765625, 0.35048828125, 0.35986328125, 0.265771484375, 0.208740234375, 0.169921875, 0.114892578125, 0.10341796875, 0.046044921875, 0.0296875, 0.0388671875, 0.057080078125, 0.04384765625, 0.016259765625]



# BER is measured for the following SNRs.
steps = np.arange(0, 32, 2)

# Plot the results in a BER over SNR plot.
plt.figure(1)

plt.plot(steps, flat2x2_1024_4, 'r-<', label='Flat fading')
plt.plot(steps, unifrom2x2_1024_4, 'b-<', label='Rayleigh, P=3')
plt.plot(steps, linea1r2x2_1024_4, 'g-<', label='Rayleigh, long linear CIR')
plt.plot(steps, linea2r2x2_1024_4, 'm-<', label='Rayleigh, long linear CIR')
plt.plot(steps, epa2x2_1024_4, 'k-<', label='LTE, EPA')
plt.axis([0, 30, 0.00001, 1])
plt.xlabel('SNR / dB')
plt.yscale('log')
plt.grid()
plt.legend()

#plt.show()

from matplotlib2tikz import save as tikz_save
tikz_save('../master-thesis/figures/custom_flat.tex', figureheight='9.5cm', figurewidth='15cm');
