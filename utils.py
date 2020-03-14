import tensorflow as tf
import pytoolkit.tf_funcs as tfx

def to_float_tensor(batch):
    batch = tf.cast(batch, tf.float32)
    return batch / 127.5 - 1.0

def to_uint8_tensor(batch):
    batch = (batch + 1.0) * 127.5
    return tf.cast(batch, tf.uint8)

def conv_layer_2d(im, chnls, rate, knls, stps, name,
                  isdeconv=False,
                  usebn=False,
                  is_train=True,
                  userelu=False):
    with tf.variable_scope(name) as scope:
        if rate != 1:
            lconv = tfx.dilated_conv2d(im, chnls, rate, (knls, knls), name='dilated_conv')
        else:
            func, name = (tfx.deconv2d, 'deconv') if isdeconv else (tfx.conv2d, 'conv')
            lconv = func(im, chnls, (knls, knls), (stps, stps), name=name)
        lconv = tfx.bn_new(lconv, phase_train=is_train) if usebn else lconv
        lconv = tfx.relu(lconv) if userelu else lconv
        return lconv

def compute_lxloss(mask, rs, gt, name, mode):
    assert mode == 'l2' or mode == 'l1', 'mode must be l2 or l1'
    func = tf.square if mode == 'l2' else tf.abs
    if mask is not None:
        gt *= mask
        rs *= mask
        rst = tf.reduce_sum(func(gt - rs)) / (tf.reduce_sum(mask) * 3 + 1e-8)
        return tf.identity(rst, name=name)
    else:
        return tf.reduce_mean(func(gt - rs), name=name)

def get_valid_batch(batch, valid_len, name):
    shape = batch.get_shape().as_list()
    ndims = len(shape)
    out = tf.slice(batch, [0] * ndims, [valid_len] + [-1] * (ndims - 1), name=name)
    return out