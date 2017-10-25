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
Plotting: - 2x2
          - K = 1024
          - M = 2/4/8/16
          - Hard Coded Channel

"""


import numpy as np
import matplotlib.pyplot as plt


# Testing the 2x2 transmission scheme.
points2x2_2_hard = 0
points2x2_2_hard_ce_x = [0.426904296875, 0.40830078125, 0.377734375, 0.349755859375, 0.3208984375, 0.28291015625, 0.23994140625, 0.19921875, 0.164453125, 0.139501953125, 0.107421875, 0.08798828125, 0.0638671875, 0.039208984375, 0.028515625, 0.01962890625, 0.015966796875, 0.007177734375, 0.002099609375, 0.00078125, 0.00126953125]

points2x2_4_hard = 0
points2x2_4_hard_ce = [0.393505859375, 0.35390625, 0.3283203125, 0.29599609375, 0.253857421875, 0.22373046875, 0.18037109375, 0.140673828125, 0.1025390625, 0.075, 0.055517578125, 0.03857421875, 0.02431640625, 0.01494140625, 0.005908203125, 0.002490234375, 0.001826171875, 0.0009814453125, 0.0005029296875, 0.00029296875, 0.0002099609375]
points2x2_4_hard_ce_x = [0.42412109375, 0.402392578125, 0.379541015625, 0.3484375, 0.305078125, 0.275927734375, 0.236865234375, 0.1888671875, 0.155908203125, 0.12021484375, 0.091650390625, 0.079541015625, 0.044921875, 0.03193359375, 0.017724609375, 0.011474609375, 0.00751953125, 0.001806640625, 0.002685546875, 0.0005859375, 0.0005859375]

points2x2_8_hard_ce_x = [0.441552734375, 0.3986328125, 0.37041015625, 0.3458984375, 0.31171875, 0.27685546875, 0.240234375, 0.184912109375, 0.150390625, 0.127685546875, 0.083935546875, 0.056787109375, 0.04248046875, 0.023828125, 0.017529296875, 0.007958984375, 0.007275390625, 0.002197265625, 0.001171875, 0.00048828125, 0.0001953125]

points2x2_16_hard_ce_x = [0.4291015625, 0.407275390625, 0.3734375, 0.3447265625, 0.31298828125, 0.269970703125, 0.23359375, 0.191796875, 0.15009765625, 0.107373046875, 0.08984375, 0.067138671875, 0.050390625, 0.02255859375, 0.015185546875, 0.00703125, 0.00478515625, 0.001904296875, 0.002001953125, 0.00078125, 0.000244140625]

# BER is measured for the following SNRs.
steps = np.arange(0, 42, 2)

# Plot the results in a BER over SNR plot.
plt.figure()

plt.title('Antenna-Setup=(2x2), K=1024, 10/100 Rounds/Point, f_off=100Hz.')
plt.plot(steps, points2x2_2_hard_ce_x, 'm-<', label='LSS, M=2, CE, HardCodedChannel')
plt.plot(steps, points2x2_4_hard_ce, 'c-o', label='LSS, M=4, CE, HardCodedChannel')
plt.plot(steps, points2x2_4_hard_ce_x, 'c-<', label='LSS, M=4, CE, HardCodedChannel')
plt.plot(steps, points2x2_8_hard_ce_x, 'b-<', label='LSS, M=8, CE, HardCodedChannel')
plt.plot(steps, points2x2_16_hard_ce_x, 'k-<', label='LSS, M=16, CE, HardCodedChannel')
plt.axis([0, 40, 0.00001, 1])
plt.xlabel('SNR / dB')
plt.ylabel('BER')
plt.yscale('log')
plt.legend()

plt.show()

#from matplotlib2tikz import save as tikz_save
#tikz_save('../master-thesis/figures/points_plot_hc_1024.tex', figureheight='7cm', figurewidth='7cm');
