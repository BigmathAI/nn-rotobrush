from data_layer import data_layer_2d
from config import FLAGS, TFCONFIG, logger
from network import CENet
from solver import solver_wrapper
import pytoolkit.utils as pyutils

import tensorflow as tf
import os, cv2, time

def main():
    dl_train = data_layer_2d(FLAGS, 'train') if FLAGS.mode in ['train', 'finetune'] else None
    dl_valid = data_layer_2d(FLAGS, 'valid')

    net = CENet(FLAGS)

    with tf.Session(config=TFCONFIG) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        sw = solver_wrapper(net, (dl_train, dl_valid), sess, FLAGS)
        if FLAGS.mode == 'train':
            sw.Train()
        elif FLAGS.mode == 'finetune':
            sw.Finetune()
        elif FLAGS.mode == 'valid':
            total_loss_vals = sw.Evaluate()
            [logger.info(line) for line in pyutils.dict_to_string(total_loss_vals, 3)]
        elif FLAGS.mode == 'findbest':
            sw.FindBestModel()
        else:
            print('Do Nothing')

def test():
    FLAGS.num_gpus = 1
    FLAGS.batch_size = 1
    dl_valid = data_layer_2d(FLAGS, 'valid')
    k = 0
    while dl_valid.status.epoch < 1:
        rst = dl_valid.next_batch()
        im = rst[0][0]
        print(im.shape)
        cv2.imwrite('%05d.jpg' % k, im)
        k += 1

def debug():
    net = CENet(FLAGS)
    fname_meta = 'F:\\results\\rotobrush-0430\\XXXXX\\log\\temp_model.ckpt-0.meta'
    fname_ckpt = 'F:\\results\\rotobrush-0430\\XXXXX\\log\\temp_model.ckpt-0'
    with tf.Session(config=TFCONFIG) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver(net.var.all_vars, max_to_keep=None)
        saver.restore(sess, fname_ckpt)

if __name__ == '__main__':
    logger.info('Start Program: ...')
    debug()