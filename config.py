from easydict import EasyDict as edict
import pytoolkit.utils as pyutils, os
import pytoolkit.files as fp
import tensorflow as tf
import platform

def INIT_EXP_ENV(FLAGS, TFCONFIG):
    fp.mkdir(FLAGS.result_path)
    tmp = ['log', 'output', 'figures', 'tb']
    for s in tmp:
        fd = os.path.join(FLAGS.result_path, s)
        tf.flags.DEFINE_string(s + '_path',             fd,         '')
        fp.mkdir(fd)
    TFCONFIG.gpu_options.allow_growth = True

#-----------------------------------------------------------------------------------------------------------------------
tf.flags.DEFINE_integer('batch_size',                   16,         'batch size in single GPU')
tf.flags.DEFINE_list('image_size',                      [256,256],  'size')

tf.flags.DEFINE_integer('num_gpus',                     1,          'number of GPUs')
tf.flags.DEFINE_integer('epoches',                      500,        'number of epoches')
tf.flags.DEFINE_integer('eval_nums',                    2,         'the number of batch evaluated, -1 means all')

tf.flags.DEFINE_float('lr',                             1e-5,       'learning rate')

tf.flags.DEFINE_string('mode',                          'finetune', 'train, valid, finetune or test')

if platform.system() == 'Windows':
    tf.flags.DEFINE_string('data_path',                 r'F:\text-seg\totaltext', '')
    tf.flags.DEFINE_string('result_path',               r'F:\text-seg\results', '')
else:
    tf.flags.DEFINE_string('data_path',                 r'/data/totaltext', '')
    tf.flags.DEFINE_string('result_path',               r'/data/results/totaltext', '')

tf.flags.DEFINE_integer('NUM_THREADS',                  8,          '')
tf.flags.DEFINE_bool('USE_MULTI_THREADS',               True,       '')
#-----------------------------------------------------------------------------------------------------------------------

FLAGS = tf.flags.FLAGS
TFCONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
INIT_EXP_ENV(FLAGS, TFCONFIG)
logger = pyutils.create_logger(os.path.join(FLAGS.log_path, 'logger.txt'))
pyutils.print_params(logger, FLAGS)