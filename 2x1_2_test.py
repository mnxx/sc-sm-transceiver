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
Create BER/SNR plots for the Single Carrier Spatial Modulation scheme using 2 TAs, 1 RAs and M = 2.
This test also requires: - The number of symbols per frame.
                         - The size of each symbol in bits.
                         - Number of rounds (precision/duration).
"""


import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt


symbols = 8
symbol_size = 2
rounds = 1000


# BER is measured for the following SNRs.
steps = np.arange(0, 20, 2)

# The resulting BER values are stored in a list.
points = []

for step in steps:
    # Start measuring the time.
    # As check_output has to wait for the console, the process takes relatively long.
    start = time.time()

    # Sum the erroneus bits for each iteration.
    count = 0
    for i in range(0, rounds):
        errors = subprocess.check_output(["python3", "proposed_detector.py", str(step)])
        count += int(errors)

    # Measure the passed time.
    diff = time.time() - start

    ber = count / (symbols * symbol_size * rounds)
    
    # Print the results.
    print("\n" + str(count) + " bits in " + str(rounds) + " tests were wrong! \n > BER = " + str(ber * 1000) + " * 10^-3."+ "\n > In " + str(diff) + " seconds.\n")

    # Append result to the list.
    points.append(ber)

print(points)
    
# Plot the results in a BER over SNR plot.
plt.plot(steps, points, 'bo', steps, points, 'k')
plt.axis([0,20,0.0001, 1])
plt.yscale('log')
plt.show()
