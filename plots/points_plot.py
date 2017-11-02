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
# 20 dB CE error:
points2x1_2_ce = [7.75e-05, 5.625e-05, 9e-05, 8.125e-05, 8.25e-05, 9.125e-05, 9.125e-05, 7.25e-05, 5.5e-05, 8.75e-05, 4.625e-05]
# Previous & model results.
points2x1_2 = [0.28773625, 0.21225625, 0.14205375, 0.08651625, 0.04656, 0.0232975, 0.01064, 0.00462625, 0.00199, 0.0008125, 0.00031625]
points2x1_4 = [0.20399875, 0.12980875, 0.06934875, 0.03154375, 0.01171875, 0.004145, 0.0011425, 0.00035, 0.000105, 3.5e-05, 1e-05]

# Testing the 2x2 transmission scheme.
# 17 dB CE SNR.
points2x2_2_ce = [0.00015, 5e-05, 2.5e-05, 2.5e-05, 0.000125, 0.000125, 7.5e-05, 2.5e-05, 0.0, 0.0001125, 2.5e-05]
# Previous results.
points2x2_2 = [0.111095, 0.05419375, 0.02167125, 0.00706625, 0.001975, 0.0004925, 8.25e-05, 7.5e-06, 0.0, 0.0, 0.0]
points2x2_4 = [0.0503625, 0.0167825, 0.00410625, 0.00059375, 9.125e-05, 2.5e-06, 2.5e-06, 0.0, 0.0, 0.0, 0.0]
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

# K = 1024
new_points2x1_2_1024 = [0.2603125, 0.2405859375, 0.15779296875, 0.091220703125, 0.064287109375, 0.0260498046875, 0.027275390625, 0.006025390625, 0.0044482421875, 0.0011376953125, 0.0012060546875]
new_points2x1_4_1024 = [0.2222216796875, 0.151513671875, 0.090869140625, 0.039345703125, 0.0292626953125, 0.003740234375, 0.000546875, 0.0001318359375, 3.90625e-05, 0.0, 0.0]
new_points2x2_2_1024 = [0.222734375, 0.1414404296875, 0.08794921875, 0.048681640625, 0.0182470703125, 0.00736328125, 0.0018017578125, 0.000556640625, 0.0002294921875, 0.0, 0.0]
new_points2x2_4_1024 = [0.1597802734375, 0.105732421875, 0.0496923828125, 0.0148291015625, 0.002275390625, 0.0002685546875, 9.27734375e-05, 0.0, 0.0, 0.0, 0.0]

# OFDM-SM
ofdm_sm2x1 = [0.232254, 0.184291, 0.137450, 0.099697, 0.069683, 0.048027, 0.031876, 0.020667, 0.013600, 0.008385, 0.005842]
"""
0	0.232254
1	0.208437
2	0.184291
3	0.161436
4	0.137450
5	0.118708
6	0.099697
7	0.084043
8	0.069683
9	0.058973
10	0.048027
11	0.038166
12	0.031876
13	0.026397
14	0.020667
15	0.016176
16	0.013600
17	0.010684
18	0.008385
19	0.007139
20	0.005842
21	0.004245
22	0.003228
23	0.002829
24	0.002355
25	0.001629
26	0.001424
27	0.001118
28	0.000767
29	0.000570
30	0.000366
"""
ofdm_sm2x2 = [0.175181, 0.129666, 0.086101, 0.049967, 0.026723, 0.012924, 0.005945, 0.002302, 0.000930, 0.000466, 0.000200]
#[0.109666, 0.066076, 0.036543, 0.018635, 0.008719, 0.003908, 0.001843, 0.000713, 0.000329, 0.000149, 0.000055]
"""
0	0.109666
1	0.086101
2	0.066076
3	0.049967
4	0.036543
5	0.026723
6	0.018635
7	0.012924
8	0.008719
9	0.005945
10	0.003908
11	0.002302
12	0.001843
13	0.000930
14	0.000713
15	0.000466
16	0.000329
17	0.000200
18	0.000149
19	0.000077
20	0.000055
21	0.000033
22	0.000018
23	0.000014
24	0.000003
25	0.000001
26	0.000013
27	0.000003
28	0.000000
29	0.000000
30	0.000000
"""

# BER is measured for the following SNRs.
steps = np.arange(0, 22, 2)

# Plot the results in a BER over SNR plot.
plt.figure(1)

plt.subplot(121)
plt.title('MIMO-setup: 2x1')
#plt.plot(steps, points2x1_2, 'r-<', label='LSS, M=2')
#plt.plot(steps, points2x1_4, 'b-o', label='LSS, M=4')
plt.plot(steps, new_points2x1_2, 'r-<', label='K=4, M=2')
plt.plot(steps, new_points2x1_4, 'b-o', label='K=4, M=4')
#plt.plot(steps, new_points2x1_2_128 , 'g-<', label='K=128, M=2')
#plt.plot(steps, new_points2x1_4_128, 'g-o', label='K=128, M=4')
#plt.plot(steps, new_points2x1_2_1024 , 'm-<', label='K=1024, M=2')
#plt.plot(steps, new_points2x1_4_1024 , 'c-o', label='K=1024, M=4')
plt.plot(steps, ofdm_sm2x1, 'm-*', label='OFDM-SM')
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
#plt.plot(steps, points2x2_2, 'r-<', label='LSS, M=2')
#plt.plot(steps, points2x2_4, 'b-o', label='LSS, M=4')
plt.plot(steps, new_points2x2_2, 'r-<', label='K=4, M=2')
plt.plot(steps, new_points2x2_4, 'b-o', label='K=4, M=4')
#plt.plot(steps, new_points2x2_2_128 , 'g-<', label='K=128, M=2')
#plt.plot(steps, new_points2x2_4_128, 'g-o', label='K=128, M=4')
#plt.plot(steps, new_points2x2_2_1024 , 'm-<', label='K=1024, M=2')
#plt.plot(steps, new_points2x2_4_1024 , 'c-o', label='K=1024, M=4')
plt.plot(steps, ofdm_sm2x2, 'm-*', label='OFDM-SM')
plt.axis([0, 20, 0.00001, 1])
plt.xlabel('SNR / dB')
plt.yscale('log')
plt.grid()
plt.legend()

#plt.show()

from matplotlib2tikz import save as tikz_save
tikz_save('../master-thesis/figures/ofdm_sm.tex', figureheight='9.5cm', figurewidth='7.5cm');
