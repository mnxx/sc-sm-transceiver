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


# Testing the 2x1 transmission scheme.

points2x1_2_ce = [7.75e-05, 5.625e-05, 9e-05, 8.125e-05, 8.25e-05, 9.125e-05, 9.125e-05, 7.25e-05, 5.5e-05, 8.75e-05, 4.625e-05]

points2x1_2 = [0.28773625, 0.21225625, 0.14205375, 0.08651625, 0.04656, 0.0232975, 0.01064, 0.00462625, 0.00199, 0.0008125, 0.00031625]

points2x1_4 = [0.20399875, 0.12980875, 0.06934875, 0.03154375, 0.01171875, 0.004145, 0.0011425, 0.00035, 0.000105, 3.5e-05, 1e-05]

# Testing the 2x2 transmission scheme.
#
points2x2_2_ce = [0.43369140625, 0.39375, 0.347998046875, 0.31005859375, 0.2572265625, 0.21298828125, 0.16875, 0.11806640625, 0.06337890625, 0.02763671875, 0.0078125]
#
points2x2_2_hard = [0.434716796875, 0.385205078125, 0.345654296875, 0.311962890625, 0.269580078125, 0.23115234375, 0.17578125, 0.11591796875, 0.06025390625, 0.02265625, 0.00556640625]
#
points2x2_4_hard = [0.419873046875, 0.375244140625, 0.320458984375, 0.286181640625, 0.23837890625, 0.17744140625, 0.12265625, 0.06474609375, 0.02001953125, 0.0017578125, 0.0]
#
points2x2_4 = [0.069580078125, 0.006103515625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# BER is measured for the following SNRs.
steps = np.arange(0, 22, 2)

# Plot the results in a BER over SNR plot.
plt.figure(1)
plt.subplot(121)
plt.title('2x1')
plt.plot(steps, points2x1_2_ce, 'k-<', label='LSS, M=2, CE=27dB')
plt.plot(steps, points2x1_2, 'r-<', label='LSS, M=2')
plt.plot(steps, points2x1_4, 'b-o', label='LSS, M=4')
plt.axis([0, 20, 0.00001, 1])
plt.xlabel('SNR / dB')
plt.ylabel('BER')
plt.yscale('log')
plt.legend()
plt.subplot(122)
plt.title('2x2')
plt.plot(steps, points2x2_2_ce, 'k-<', label='LSS, K=1024, M=2, CE_seq = 10')
plt.plot(steps, points2x2_2_hard, 'r-<', label='LSS, K=1024, M=2, PERF_CSI / HARD')
plt.plot(steps, points2x2_4_hard, 'b-o', label='LSS, K=1024, M=4, PERF_CSI / HARD')
plt.plot(steps, points2x2_4, 'g-o', label='LSS, K=1024, M=4, PERF_CSI / UNI')
plt.axis([0, 20, 0.00001, 1])
plt.xlabel('SNR / dB')
plt.yscale('log')
plt.legend()
plt.show()
