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


#points = [0.30975, 0.2535, 0.1565, 0.101125, 0.0395, 0.02, 0.007, 0.001, 0.00055, 0.0002125,  0.000075]
points = [0.110375, 0.069875, 0.04125, 0.022375, 0.015625, 0.008125, 0.00425, 0.003625, 0.000625, 0.000375, 0.0]

# BER is measured for the following SNRs.
steps = np.arange(0, 22, 2)

# Plot the results in a BER over SNR plot.
plt.plot(steps, points, 'bo', steps, points, 'k')
plt.axis([0, 20, 0.00001, 1])
plt.yscale('log')
plt.show()
