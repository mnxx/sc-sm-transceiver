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
Test the output of the proposed detector scheme, to see how reliable the system is. 

"""


import sys
import time
import subprocess

symbols = 8
symbol_size = 2
rounds = 1000
count = 0

# Start measuring the time.
# As check_output has to wait for the console, the process takes relatively long.
start = time.time()

# Sum the erroneus bits for each iteration.
for i in range(0, rounds):
    errors = subprocess.check_output(["python3", "proposed_detector.py", "10"])
    count += int(errors)

# Measure the passed time.
diff = time.time() - start

# Print the results.
print(str(count) + " bits in " + str(rounds) + " tests were wrong! \n > BER = " + str(count / (symbols * symbol_size * rounds) * 1000) + " * 10^-3."+ "\n > In " + str(diff) + " seconds.")
