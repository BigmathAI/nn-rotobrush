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

import tensorflow as tf, numpy as np

a = np.random.random([5,2])
b = np.random.random([5])
b = (b > 0.5).astype(np.int32)

ea = np.exp(a)
s = np.sum(ea, axis=1)
ea /= s[:,np.newaxis]
la = np.log(ea)


xb = np.zeros([b.shape[0], 2], np.float32)
row = 0
for t in b:
    xb[row][t] = 1
    row += 1
out = -np.sum(la * xb, axis=1)

ta = tf.constant(a)
tb = tf.constant(b)

xa = tf.cast(tf.greater(ta, 0.5), tf.int32)

#tc = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tb, logits=ta)
tc = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tb, logits=ta)

ss = tf.Session()
c = ss.run(xa)

print(a)
print(b)

print(c.shape)
print(c)

print(out)
