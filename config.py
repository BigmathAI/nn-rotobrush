from easydict import EasyDict as edict
import pytoolkit.utils as pyutils, os
import pytoolkit.files as fp
import tensorflow as tf
import platform

def INIT_EXP_ENV(FLAGS, TFCONFIG):
    #fd = '{}_[lr_{:3.1e}]_[weight_ce_{:>04.2f}]_[weight_decay_{:>06.4f}]'.format(FLAGS.exp_id,
    #                                                                             FLAGS.lr,
    #                                                                             FLAGS.weight_ce,
    #                                                                             FLAGS.weight_decay)
    fd = '{}'.format(FLAGS.exp_id)
    FLAGS.result_path = os.path.join(FLAGS.result_path, fd)
    fp.mkdir(FLAGS.result_path)
    tmp = ['log', 'output', 'figures', 'tb']
    for s in tmp:
        fd = os.path.join(FLAGS.result_path, s)
        tf.flags.DEFINE_string(s + '_path',             fd,         '')
        fp.mkdir(fd)
    TFCONFIG.gpu_options.allow_growth = True

#-----------------------------------------------------------------------------------------------------------------------
tf.flags.DEFINE_string('exp_id',                        'ID000',         '')

tf.flags.DEFINE_integer('batch_size',                   8,          'batch size in single GPU')
tf.flags.DEFINE_list('image_size',                      [256,256],  'size')

tf.flags.DEFINE_integer('num_gpus',                     1,          'number of GPUs')
tf.flags.DEFINE_integer('epoches',                      500,        'number of epoches')
tf.flags.DEFINE_integer('eval_nums',                    2,         'the number of batch evaluated, -1 means all')

tf.flags.DEFINE_float('weight_decay',                   1e-3,       'weight decay')
tf.flags.DEFINE_float('weight_ce',                      0.7,        'weight ce, range: [0,1)')
tf.flags.DEFINE_float('lr',                             1e-5,       'learning rate')

tf.flags.DEFINE_string('version',                       'v4',       'version')

tf.flags.DEFINE_string('mode',                          'finetune', 'train, valid, finetune, or findbest')

if platform.system() == 'Windows':
    tf.flags.DEFINE_string('data_path',                 r'F:\text-seg\totaltext', '')
else:
    tf.flags.DEFINE_string('data_path',                 r'/data/totaltext', '')

if platform.system() == 'Windows':
    tf.flags.DEFINE_string('result_path',               r'F:\text-seg\results\totaltext-ce4', '')
else:
    tf.flags.DEFINE_string('result_path',               r'/data/results/totaltext-ce4', '')

tf.flags.DEFINE_integer('NUM_THREADS',                  8,          '')
tf.flags.DEFINE_bool('USE_MULTI_THREADS',               True,       '')
#-----------------------------------------------------------------------------------------------------------------------

FLAGS = tf.flags.FLAGS
TFCONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
INIT_EXP_ENV(FLAGS, TFCONFIG)
logger = pyutils.create_logger(os.path.join(FLAGS.log_path, 'logger.txt'))
pyutils.print_params(logger, FLAGS)