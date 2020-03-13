from easydict import EasyDict as edict
import pytoolkit.utils as pyutils, os
import pytoolkit.files as fp

FLAGS = edict({
    'data_path':                            r'/data/totaltext',
    'result_path':                          r'/data/results/totaltext',

    'batch_size':                           48,
    'num_gpus':                             1,

    'USE_MULTI_THREADS':                    True,
    'NUM_THREADS':                          8,
})

fp.mkdir(FLAGS['result_path'])

tmp = ['log', 'output', 'figures', 'tb']
for s in tmp:
    fd = FLAGS[s + '_path'] = os.path.join(FLAGS['result_path'], s)
    fp.mkdir(fd)

logger = pyutils.create_logger(os.path.join(FLAGS.log_path, 'logger.txt'))