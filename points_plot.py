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
points2x1_2 = [0.38358625, 0.32032875, 0.2498675, 0.17789625, 0.1153075, 0.06456375, 0.032965, 0.0156525, 0.0067, 0.00285875, 0.00122875]
points2x1_4 = [0.3185475, 0.244315, 0.167215, 0.097645, 0.047265, 0.01987875, 0.00742625, 0.002215, 0.00056125, 0.00014, 2.25e-05]

# Testing the 2x2 transmission scheme.
points2x2_2 = [0.2286025, 0.14755625, 0.07947625, 0.03591875, 0.01272125, 0.0036375, 0.0009675, 0.00016375, 2e-05, 8.125e-06, 0.0]
points2x2_4 = [0.152605, 0.0790525, 0.03022125, 0.00879875, 0.0017375, 0.00026875, 3.5e-05, 2.25e-06, 0.0, 0.0, 0.0]

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
