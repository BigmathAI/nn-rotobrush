from pytoolkit.utils import ProgressBar
import time, datetime
pb = ProgressBar(1000)

def foo():
    for i in range(pb.GetTotal() * 2):
        if i == pb.GetTotal():
            pb.Reset()
        pb.Update()
        s = pb.GetBar()
        print(s)
        time.sleep(0.01)

foo()

import tqdm

for i in tqdm.tqdm(range(100000)):
    a = 1
    time.sleep(0.3)