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
This test investigates the flat fading scenario.

"""


import numpy as np
import matplotlib.pyplot as plt


# Testing the 2x1 transmission scheme.
# Previous & model results.
points2x1_2 = [0.28773625, 0.21225625, 0.14205375, 0.08651625, 0.04656, 0.0232975, 0.01064, 0.00462625, 0.00199, 0.0008125, 0.00031625]
points2x1_4 = [0.20399875, 0.12980875, 0.06934875, 0.03154375, 0.01171875, 0.004145, 0.0011425, 0.00035, 0.000105, 3.5e-05, 1e-05]

# Testing the 2x2 transmission scheme.
# Model results.
gp_2 = [0.14705875, 0.07981875, 0.0358475, 0.012775, 0.00368875, 0.001025, 0.00027625, 3.5e-05, 1e-05, 0.0, 0.0]
gp_4 = [0.0788675, 0.0303975, 0.00855875, 0.00169375, 0.00024375, 4e-05, 5e-06, 0.0, 0.0, 0.0, 0.0]

# K = 4
new_points2x1_2 = [0.22123, 0.1628425, 0.1103525, 0.06604375, 0.03643375, 0.01784625, 0.00793625, 0.00333625, 0.00137875, 0.00055625, 0.0002255]
new_points2x1_4 = [0.15721, 0.099085, 0.05289375, 0.02367625, 0.0092025, 0.00295875, 0.000889375, 0.000245, 6.875e-05, 1.75e-05, 3.875e-06] 
new_points2x2_2 = [0.17318125, 0.11204625, 0.06107875, 0.0267575, 0.00963875, 0.00265375, 0.0005975, 0.0001535, 2.9375e-05, 6.5e-06, 0.0]
new_points2x2_4 = [0.11534125, 0.059455, 0.0227725, 0.00637625, 0.00140125, 0.0002275, 1.75e-05, 1.5e-06, 0.0, 0.0, 0.0]

# K = 128
new_points2x1_2_128 = [0.2794140625, 0.2244921875, 0.166796875, 0.108125, 0.0545703125, 0.036171875, 0.014609375, 0.005, 0.0019140625, 0.0005859375, 0.0001953125]
new_points2x1_4_128 = [0.2165234375, 0.155625, 0.0868359375, 0.0399609375, 0.018671875, 0.00203125, 0.0009375, 0.000625, 7.8125e-05, 0.0, 0.0]
new_points2x2_2_128 = [0.2109375, 0.14578125, 0.0829296875, 0.0448046875, 0.021015625, 0.005859375, 0.000859375, 7.8125e-05, 0.0, 0.0, 0.0]
new_points2x2_4_128 = [0.1599609375, 0.0898046875, 0.0350390625, 0.0118359375, 0.0040625, 0.00015625, 0.0, 0.0, 0.0, 0.0, 0.0]

# P = 1
flat2x1_2 = [0.28026, 0.23334875, 0.185045, 0.14105125, 0.101215, 0.070685, 0.04788875, 0.03219375, 0.02112375, 0.0134225, 0.00846875]
flat2x1_4 = [0.2791075, 0.23346875, 0.1852575, 0.13964875, 0.1014475, 0.07070375, 0.04817, 0.03157375, 0.020635, 0.01342125, 0.0086325]
flat2x2_2 = [0.24791875, 0.19249, 0.13634, 0.08725, 0.05143, 0.027035, 0.013355, 0.0060875, 0.0026175, 0.00107, 0.0005225]
flat2x2_4 = [0.2476025, 0.19248, 0.1359375, 0.08752, 0.05121875, 0.026965, 0.01335375, 0.00586125, 0.00253875, 0.0011125, 0.0004775]

# P = 3

# P = 5
p5_2x1_2 = [0.2163075, 0.1592075, 0.10751125, 0.06552375, 0.03498, 0.0175075, 0.00810125, 0.00379875, 0.00165125, 0.000675, 0.0002525]
p5_2x1_4 = [0.12865125, 0.075585, 0.03718875, 0.01590625, 0.00571625, 0.0018475, 0.0004975, 0.00013, 4.5e-05, 6.25e-06, 0.0]
p5_2x2_2 = [0.166755, 0.10724125, 0.05843, 0.02601625, 0.00957125, 0.00267875, 0.0006725, 0.000155, 4.625e-05, 3.75e-06, 0.0]
p5_2x2_4 = [0.08777125, 0.040395, 0.0139875, 0.00338, 0.00061875, 0.0001, 1e-05, 1.25e-06, 0.0, 0.0, 0.0]

# P = 7
p7_2x1_2 = [0.21339625, 0.1584575, 0.10601, 0.064175, 0.035405, 0.01743375, 0.008585, 0.003615, 0.00142, 0.00066375, 0.00028]
p7_2x1_4 = [0.1182075, 0.069815, 0.03505875, 0.01521625, 0.00561125, 0.00180625, 0.00054, 0.00014, 3.375e-05, 1e-05, 6.25e-06]
p7_2x2_2 = [0.16441375, 0.1072275, 0.05830625, 0.02590625, 0.009155, 0.00295625, 0.0007575, 0.00013625, 2e-05, 0.0, 0.0]
p7_2x2_4 = [0.0780925, 0.03629625, 0.01267, 0.00326625, 0.000575, 6.875e-05, 6.25e-06, 5e-06, 0.0, 0.0, 0.0]

# P = 9
p9_2x1_2 = [0.21099375, 0.1578225, 0.10551125, 0.06457625, 0.0349775, 0.0174275, 0.0082975, 0.0035025, 0.001435, 0.00062, 0.00030875]
p9_2x1_4 = [0.113915, 0.067285, 0.03384375, 0.01485625, 0.005395, 0.0017775, 0.000475, 0.000175, 3.75e-05, 1.75e-05, 2.5e-06]
p9_2x2_2 = [0.1620925, 0.1056225, 0.05804125, 0.026085, 0.0092025, 0.00277625, 0.00070625, 0.000125, 2.875e-05, 0.0, 0.0]
p9_2x2_4 = [0.076605, 0.03575, 0.01219375, 0.003305, 0.0005875, 0.00011, 8.75e-06, 0.0, 0.0, 0.0, 0.0]

# BER is measured for the following SNRs.
steps = np.arange(0, 22, 2)

# Plot the results in a BER over SNR plot.
plt.figure(1)

plt.subplot(121)
plt.title('MIMO-setup: 2x1')
#plt.plot(steps, points2x1_2, 'r-', label='LSS, M=2')
#plt.plot(steps, points2x1_4, 'g-', label='LSS, M=4')
plt.plot(steps, new_points2x1_2, 'r-<', label='P=3, M=2')
plt.plot(steps, new_points2x1_4, 'b-o', label='P=3, M=4')
#plt.plot(steps, new_points2x1_2_128 , 'g-<', label='K=128, M=2')
#plt.plot(steps, new_points2x1_4_128, 'g-o', label='K=128, M=4')
#plt.plot(steps, flat2x1_2, 'k-<', label='P=1, M=2')
#plt.plot(steps, flat2x1_4, 'g--o', label='P=1, M=4')
plt.plot(steps, p5_2x1_2, 'g-<', label='P=7, M=2')
plt.plot(steps, p5_2x1_4, 'g-o', label='P=7, M=4')
plt.plot(steps, p9_2x1_2, 'm-<', label='P=9, M=2')
plt.plot(steps, p9_2x1_4, 'm-o', label='P=9, M=4')
plt.axis([0, 20, 0.00001, 1])
plt.xlabel('SNR / dB')
plt.ylabel('BER')
plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot(122)
plt.title('MIMO-setup: 2x2')
#plt.plot(steps, gp_2, 'r-')
#plt.plot(steps, gp_4, 'g-')
plt.plot(steps, new_points2x2_2, 'r-<', label='P=3, M=2')
plt.plot(steps, new_points2x2_4, 'b-o', label='P=3, M=4')
#plt.plot(steps, new_points2x2_2_128 , 'g-<', label='K=128, M=2')
#plt.plot(steps, new_points2x2_4_128, 'g-o', label='K=128, M=4')
#plt.plot(steps, flat2x2_2, 'k-<', label='P=1, M=2')
#plt.plot(steps, flat2x2_4, 'g--o', label='P=1, M=4')
plt.plot(steps, p5_2x2_2, 'g-<', label='P=7, M=2')
plt.plot(steps, p5_2x2_4, 'g-o', label='P=7, M=4')
plt.plot(steps, p9_2x2_2, 'm-<', label='P=9, M=2')
plt.plot(steps, p9_2x2_4, 'm-o', label='P=9, M=4')
plt.axis([0, 20, 0.00001, 1])
plt.xlabel('SNR / dB')
plt.yscale('log')
plt.grid()
plt.legend()

plt.show()

#from matplotlib2tikz import save as tikz_save
#tikz_save('../master-thesis/figures/detector_p7.tex', figureheight='9.5cm', figurewidth='7.5cm');
