from pytoolkit.utils import ProgressBar
import time, datetime
pb = ProgressBar(100)
#pb.Restore(10)

def foo():
    last_line = None
    for i in range(pb.GetTotal()):
        pb.Update(1)
        line = '{:02d} {}'.format(i, pb.GetBar())
        if last_line is None:
            print('',  end='\r{}'.format(line))
            last_line = line
        else:
            print('', end='\r{}'.format(line))

        time.sleep(0.1)

foo()