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
          - M = 2/4
          - Hard Coded Channel

"""


import numpy as np
import matplotlib.pyplot as plt


# Testing the 2x2 transmission scheme.
points2x2_2_hard = 0
points2x2_2_hard_ce = [0.39560546875, 0.361865234375, 0.333642578125, 0.301708984375, 0.26123046875, 0.221728515625, 0.184765625, 0.150732421875, 0.11923828125, 0.098828125, 0.0689453125, 0.052880859375, 0.042041015625, 0.029248046875, 0.017333984375, 0.009326171875]
points2x2_4_hard = 0
points2x2_4_hard_ce = [0.393505859375, 0.35390625, 0.3283203125, 0.29599609375, 0.253857421875, 0.22373046875, 0.18037109375, 0.140673828125, 0.1025390625, 0.075, 0.055517578125, 0.03857421875, 0.02431640625, 0.01494140625, 0.005908203125, 0.002490234375, 0.001826171875, 0.0009814453125, 0.0005029296875, 0.00029296875, 0.0002099609375]

# BER is measured for the following SNRs.
steps = np.arange(0, 42, 2)

# Plot the results in a BER over SNR plot.
plt.figure()

#plt.subplot(121)
plt.title('Antenna-Setup=(2x2) -- K=1024 -- 10 Rounds/Point -- f_off=200Hz.')
#plt.plot(steps, points2x2_2_hard, 'r-<', label='LSS, M=2, perfect CSI, HardCodedChannel')
#plt.plot(steps, points2x2_2_hard_ce, 'm-<', label='LSS, M=2, CE, HardCodedChannel')
#plt.plot(steps, points2x2_4_hard, 'b-o', label='LSS, M=4, perfect CSI, HardCodedChannel')
plt.plot(steps, points2x2_4_hard_ce, 'k-o', label='LSS, M=4, CE, HardCodedChannel')
plt.axis([0, 40, 0.00001, 1])
plt.xlabel('SNR / dB')
plt.ylabel('BER')
plt.yscale('log')
plt.legend()

#plt.subplot(122)
#plt.title('2x2')
#plt.plot(steps, points2x2_2_ce, 'k-<', label='LSS, K=1024, M=2, CE_seq = 10')
#plt.plot(steps, points2x2_2_hard, 'r-<', label='LSS, K=1024, M=2, PERF_CSI / HARD')
#plt.plot(steps, points2x2_4_hard, 'b-o', label='LSS, K=1024, M=4, PERF_CSI / HARD')
#plt.plot(steps, points2x2_4, 'g-o', label='LSS, K=1024, M=4, PERF_CSI / UNI')
#plt.axis([0, 20, 0.00001, 1])
#plt.xlabel('SNR / dB')
#plt.yscale('log')
#plt.legend()

plt.show()
