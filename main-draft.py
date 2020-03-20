from pytoolkit.utils import ProgressBar
import time, os
pb = ProgressBar(100)

for t in range(10):
    cmd = 'start python x.py'
    os.system(cmd)