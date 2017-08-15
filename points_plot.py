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
points2x1_2 = [0.28773625, 0.21225625, 0.14205375, 0.08651625, 0.04656, 0.0232975, 0.01064, 0.00462625, 0.00199, 0.0008125, 0.00031625]
points2x1_4 = [0.20399875, 0.12980875, 0.06934875, 0.03154375, 0.01171875, 0.004145, 0.0011425, 0.00035, 0.000105, 3.5e-05, 1e-05]

# Testing the 2x2 transmission scheme.
points2x2_2 = [0.111095, 0.05419375, 0.02167125, 0.00706625, 0.001975, 0.0004925, 8.25e-05, 7.5e-06, 0.0, 0.0, 0.0]
points2x2_4 = [0.0503625, 0.0167825, 0.00410625, 0.00059375, 9.125e-05, 2.5e-06, 2.5e-06, 0.0, 0.0, 0.0, 0.0]

# BER is measured for the following SNRs.
steps = np.arange(0, 22, 2)

# Plot the results in a BER over SNR plot.
plt.title('Performance Analysis - BER/SNR')
#plt.plot(steps, points2x1_2, 'r-<', label='LSS, M=2')
#plt.plot(steps, points2x1_4, 'b-o', label='LSS, M=4')
plt.plot(steps, points2x2_2, 'r-<', label='LSS, M=2')
plt.plot(steps, points2x2_4, 'b-o', label='LSS, M=4')
plt.axis([0, 20, 0.00001, 1])
plt.xlabel('SNR')
plt.ylabel('BER')
plt.yscale('log')
plt.legend()
plt.show()
