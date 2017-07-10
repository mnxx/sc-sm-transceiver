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


# Count the successful reconstructions of the sent signals.
count = 0
start = time.time()
for i in range(1,100):
    errors = subprocess.check_output(["python3", "proposed_detector.py"])
    count += int(errors)
diff = time.time() - start
print(str(count) + " bits in 100 tests were wrong! \n > BER = " + str(count / (8 * 100)) + "\n > In " + str(diff) + " seconds.")
