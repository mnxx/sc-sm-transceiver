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

# 2x2

new_points2x1_2 = [0.22123, 0.1628425, 0.1103525, 0.06604375, 0.03643375, 0.01784625, 0.00793625, 0.00333625, 0.00137875, 0.00055625, 0.0002255]
new_points2x1_4 = [0.15721, 0.099085, 0.05289375, 0.02367625, 0.0092025, 0.00295875, 0.000889375, 0.000245, 6.875e-05, 1.75e-05, 3.875e-06]
new_points2x2_2 = [0.17318125, 0.11204625, 0.06107875, 0.0267575, 0.00963875, 0.00265375, 0.0005975, 0.0001535, 2.9375e-05, 6.5e-06, 0.0]
new_points2x2_4 = [0.11534125, 0.059455, 0.0227725, 0.00637625, 0.00140125, 0.0002275, 1.75e-05, 1.5e-06, 0.0, 0.0, 0.0]

# 8x8
mimo_4_4_8x8 = [0.25340625, 0.1699875, 0.08541875, 0.02695625, 0.00460625, 0.00053125, 1.875e-05, 0.0, 0.0, 0.0, 0.0]

# 8x16
mimo_4_4_8x16 = [0.2368, 0.1522, 0.0696125, 0.0195375, 0.00298125, 0.00013125, 0.0, 0.0, 0.0, 0.0, 0.0]

# BER is measured for the following SNRs.
steps = np.arange(0, 22, 2)

# Plot the results in a BER over SNR plot.
plt.figure(1)

plt.title('Various MIMO-setups, K=4, P=3')
#plt.plot(steps, new_points2x2_2, 'r-<', label='K=4, M=2')
plt.plot(steps, new_points2x2_4, 'b-o', label='MIMO: 2x2')
plt.plot(steps, mimo_4_4_8x8, 'g->', label='MIMO: 8x8')
plt.plot(steps, mimo_4_4_8x16, 'm->', label='MIMO: 8x16')
plt.axis([0, 20, 0.00001, 1])
plt.xlabel('SNR / dB')
plt.yscale('log')
plt.grid()
plt.legend()

plt.show()

#from matplotlib2tikz import save as tikz_save
#tikz_save('../master-thesis/figures/detector_lsmimo.tex', figureheight='9.5cm', figurewidth='7.5cm');
