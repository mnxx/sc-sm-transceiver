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
# 10 dB CE error:
#points2x1_2_ce = [0.32937125, 0.27197375, 0.2212925, 0.17606125, 0.141915, 0.1179575, 0.102585, 0.093465, 0.08688625, 0.08168625, 0.08045125]
# 20 dB CE error:
points2x1_2_ce = [7.75e-05, 5.625e-05, 9e-05, 8.125e-05, 8.25e-05, 9.125e-05, 9.125e-05, 7.25e-05, 5.5e-05, 8.75e-05, 4.625e-05] #[0.00012875, 7.75e-05, 0.00013375, 8e-05, 0.0001225, 0.000105, 0.00010375, 0.00013375, 8.875e-05, 8.75e-05, 0.0001175] #[0.000145, 0.00017625, 0.00019125, 0.00021, 0.00015625, 0.0001625, 0.0001825, 0.00018375, 0.00020375, 0.0001875, 0.0001575]
#[0.0020625, 0.00135, 0.0017375, 0.002075, 0.00165, 0.001575, 0.00185, 0.0017625, 0.0020625, 0.00195, 0.0013375]
#[0.29092, 0.219445, 0.1516325, 0.09639375, 0.05667875, 0.031735, 0.01760875, 0.01035, 0.0063575, 0.00438875, 0.00313375]
points2x1_2 = [0.28773625, 0.21225625, 0.14205375, 0.08651625, 0.04656, 0.0232975, 0.01064, 0.00462625, 0.00199, 0.0008125, 0.00031625]
points2x1_4 = [0.20399875, 0.12980875, 0.06934875, 0.03154375, 0.01171875, 0.004145, 0.0011425, 0.00035, 0.000105, 3.5e-05, 1e-05]

# Testing the 2x2 transmission scheme.
# 16 dB CE SNR.
#points2x2_2_ce = [0.0001, 0.00015, 0.00015, 0.0001375, 7.5e-05, 7.5e-05, 0.0001, 0.000225, 8.75e-05, 7.5e-05, 0.00015]
# 17 dB CE SNR.
points2x2_2_ce = [0.00015, 5e-05, 2.5e-05, 2.5e-05, 0.000125, 0.000125, 7.5e-05, 2.5e-05, 0.0, 0.0001125, 2.5e-05]

points2x2_2 = [0.111095, 0.05419375, 0.02167125, 0.00706625, 0.001975, 0.0004925, 8.25e-05, 7.5e-06, 0.0, 0.0, 0.0]
#[0.2244675, 0.108935, 0.0427425, 0.013775, 0.003555, 0.0007975, 0.00019, 3.95e-05, 7e-06, 0.0, 0.0]
gp_2 = [0.14705875, 0.07981875, 0.0358475, 0.012775, 0.00368875, 0.001025, 0.00027625, 3.5e-05, 1e-05, 0.0, 0.0]

points2x2_4 = [0.0503625, 0.0167825, 0.00410625, 0.00059375, 9.125e-05, 2.5e-06, 2.5e-06, 0.0, 0.0, 0.0, 0.0]
#[0.10201, 0.0341025, 0.00795, 0.0014675, 0.0002125, 1.3e-05, 0.0, 0.0, 0.0, 0.0, 0.0]
gp_4 = [0.0788675, 0.0303975, 0.00855875, 0.00169375, 0.00024375, 4e-05, 5e-06, 0.0, 0.0, 0.0, 0.0]

# BER is measured for the following SNRs.
steps = np.arange(0, 22, 2)

# Plot the results in a BER over SNR plot.
plt.figure(1)
#plt.plot(steps, points2x1_2, 'r-<', label='LSS, M=2')
#plt.plot(steps, points2x1_4, 'b-o', label='LSS, M=4')
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
plt.plot(steps, points2x2_2_ce, 'k-<', label='LSS, M=2, CE=17dB')
plt.plot(steps, gp_2, 'k-')
plt.plot(steps, points2x2_2, 'r-<', label='LSS, M=2')
plt.plot(steps, gp_4, 'g-')
plt.plot(steps, points2x2_4, 'b-o', label='LSS, M=4')
plt.axis([0, 20, 0.00001, 1])
plt.xlabel('SNR / dB')
plt.yscale('log')
plt.legend()
plt.show()
