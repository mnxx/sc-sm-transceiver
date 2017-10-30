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
flat2x2_1024_2 = [0.446484375, 0.3939453125, 0.385888671875, 0.33056640625, 0.2677734375, 0.32841796875, 0.0755859375, 0.04951171875, 0.039892578125, 0.0404296875, 0.0474609375]

# BER is measured for the following SNRs.
steps = np.arange(0, 22, 2)

# Plot the results in a BER over SNR plot.
plt.figure(1)

plt.plot(steps, flat2x2_1024_2, 'r-<', label='K=1024, M=2, FLAT')
plt.axis([0, 20, 0.00001, 1])
plt.xlabel('SNR / dB')
plt.yscale('log')
plt.grid()
plt.legend()

plt.show()

#from matplotlib2tikz import save as tikz_save
#tikz_save('../master-thesis/figures/custom_flat.tex', figureheight='9.5cm', figurewidth='7.5cm');
