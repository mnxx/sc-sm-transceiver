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


import subprocess


# Count the successful reconstructions of the sent signals.
count = 0
for i in range(1,100):
    val = subprocess.check_output(["python3", "proposed_detector.py"])
    #print(val)
    if val == b'True\n':
        count = count + 1

print(str(count) + " of 100 tests were correct! \n > " + str(count) + "%")
