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
#points2 = [0.396, 0.32075, 0.24975, 0.168375, 0.115225, 0.066375, 0.036375,  0.0156, 0.00715, 0.0029, 0.00115]
#points4 = [0.30975, 0.2535, 0.1565, 0.101125, 0.0395, 0.02, 0.007, 0.0023875, 0.00055, 0.0002125,  0.000075]

# Testing the 2x2 transmission scheme.
points2 = [0.23425, 0.151, 0.085125, 0.031625, 0.0115, 0.003125, 0.001,  0.0002375, 0.0000625, 0.00001375, 0.0]
points4 = [0.152605, 0.0790525, 0.03022125, 0.00879875, 0.0017375, 0.00026875, 3.5e-05, 0.0, 0.0, 0.0, 0.0] 

# BER is measured for the following SNRs.
steps = np.arange(0, 22, 2)

# Plot the results in a BER over SNR plot.
plt.title('Performance Analysis - BER/SNR')
plt.plot(steps, points2, 'r-<', label='LSS, M=2')
plt.plot(steps, points4, 'b-o', label='LSS, M=4')
plt.axis([0, 20, 0.00001, 1])
plt.xlabel('SNR')
plt.ylabel('BER')
plt.yscale('log')
plt.legend()
plt.show()
