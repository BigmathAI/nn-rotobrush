#from pytoolkit.utils import ProgressBar
import time, os
#pb = ProgressBar(100)

for t in range(3):
    cmd = 'screen -S {} echo {}'.format('SESSION'+str(t), str(t))
    print(cmd)
    os.system(cmd)

