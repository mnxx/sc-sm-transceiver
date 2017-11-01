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

import time
import subprocess


# Start measuring the time. This might take a while.
start = time.time()

# Commands to be run.
#commands = [
#    "python3 test_scheme.py 1 4 2 100000",
#    "python3 test_scheme.py 1 4 4 100000",
#    "python3 test_scheme.py 2 4 2 100000",
#    "python3 test_scheme.py 2 4 4 100000",
#]
# Run in parallel.
#processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]
# Wait until all processes have finished.
#for p in processes: p.wait()

# Commands to be run.
commands = [
    "python3 test_system.py 128 4 1",
    "python3 test_system.py 128 4 2",
    "python3 test_system.py 128 4 3",
    "python3 test_system.py 128 4 4",
#    "python3 test_system.py 1024 4 5",
#    "python3 test_system.py 1024 4 6",
#    "python3 test_system.py 1024 4 7",
]
# Run in parallel.
processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]
# Wait until all processes have finished.
for p in processes: p.wait()

# Test detector: 2x1, K=4, M=2.
#subprocess.run(["python3", "test_scheme.py", "1", "4", "2", "10000"])
# Test detector: 2x1, K=4, M=4.
#subprocess.run(["python3", "test_scheme.py", "1", "4", "4", "10000"])
# Test detector: 2x2, K=4, M=2.
#subprocess.run(["python3", "test_scheme.py", "2", "4", "2", "10000"])
# Test detector: 2x2, K=4, M=4.
#subprocess.run(["python3", "test_scheme.py", "2", "4", "4", "10000"])

# Test detector: 2x1, K=1024, M=2.
#subprocess.check_call(["python3", "test_scheme.py", "1", "1024", "2", "1000"])
# Test detector: 2x1, K=1024, M=4.
#subprocess.check_call(["python3", "test_scheme.py", "1", "1024", "4", "1000"])
# Test detector: 2x2, K=1024, M=2.
#subprocess.check_call(["python3", "test_scheme.py", "2", "1024", "2", "1000"])
# Test detector: 2x2, K=1024, M=4.
#subprocess.check_call(["python3", "test_scheme.py", "2", "1024", "4", "1000"])

# Test w/ CE: 2x1, K=1024, M=4.
#subprocess.check_call(["python3", "test_system.py", "1", "1024", "4"])
# Test w/ CE: 2x1, K=1024, M=8.
#subprocess.check_call(["python3", "test_system.py", "1", "1024", "8"])
# Test detector: 2x1, K=1024, M=16.
#subprocess.check_call(["python3", "test_system.py", "1", "1024", "16"])
# Test detector: 2x2, K=1024, M=4.
#subprocess.check_call(["python3", "test_system.py", "2", "1024", "4"])
# Test detector: 2x2, K=1024, M=8.
#subprocess.check_call(["python3", "test_system.py", "2", "1024", "8"])
# Test detector: 2x2, K=1024, M=16.
#subprocess.check_call(["python3", "test_system.py", "2", "1024", "16"])

# Test w/ CE: 2x2, K=1024, M=4.
#subprocess.run(["python3", "test_system.py", "2", "1024", "4", ""])
# Test w/ CE: 2x1, K=1024, M=8.
#subprocess.check_call(["python3", "test_system.py", "1", "1024", "8"])
# Test detector: 2x1, K=1024, M=16.
#subprocess.check_call(["python3", "test_system.py", "1", "1024", "16"])
# Test detector: 2x2, K=1024, M=4.
#subprocess.check_call(["python3", "test_system.py", "2", "1024", "4"])
# Test detector: 2x2, K=1024, M=8.
#subprocess.check_call(["python3", "test_system.py", "2", "1024", "8"])
# Test detector: 2x2, K=1024, M=16.
#subprocess.check_call(["python3", "test_system.py", "2", "1024", "16"])

# Test w/ CE: 2x1, K=1024, M=4.
#subprocess.check_call(["python3", "test_system.py", "1", "1024", "4"])
# Test w/ CE: 2x1, K=1024, M=8.
#subprocess.check_call(["python3", "test_system.py", "1", "1024", "8"])
# Test detector: 2x1, K=1024, M=16.
#subprocess.check_call(["python3", "test_system.py", "1", "1024", "16"])
# Test detector: 2x2, K=1024, M=4.
#subprocess.check_call(["python3", "test_system.py", "2", "1024", "4"])
# Test detector: 2x2, K=1024, M=8.
#subprocess.check_call(["python3", "test_system.py", "2", "1024", "8"])
# Test detector: 2x2, K=1024, M=16.
#subprocess.check_call(["python3", "test_system.py", "2", "1024", "16"])

# Test w/ CE: 2x1, K=1024, M=4.
#subprocess.check_call(["python3", "test_system.py", "1", "1024", "4"])
# Test w/ CE: 2x1, K=1024, M=8.
#subprocess.check_call(["python3", "test_system.py", "1", "1024", "8"])
# Test detector: 2x1, K=1024, M=16.
#subprocess.check_call(["python3", "test_system.py", "1", "1024", "16"])
# Test detector: 2x2, K=1024, M=4.
#subprocess.check_call(["python3", "test_system.py", "2", "1024", "4"])
# Test detector: 2x2, K=1024, M=8.
#subprocess.check_call(["python3", "test_system.py", "2", "1024", "8"])
# Test detector: 2x2, K=1024, M=16.
#subprocess.check_call(["python3", "test_system.py", "2", "1024", "16"])

# Measure the passed time.
diff = time.time() - start
print("# R U N T I M E : " + str(diff) + "s #")
