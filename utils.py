import tensorflow as tf
import pytoolkit.tf_funcs as tfx
import pytoolkit.files as fp
import pypaper.html as pyhtml
import os, cv2
import numpy as np

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

def compute_lxloss_old(mask, rs, gt, name, mode):
    assert mode == 'l2' or mode == 'l1', 'mode must be l2 or l1'
    func = tf.square if mode == 'l2' else tf.abs
    if mask is not None:
        gt *= mask
        rs *= mask
        rst = tf.reduce_sum(func(gt - rs)) / (tf.reduce_sum(mask) * 3 + 1e-8)
        return tf.identity(rst, name=name)
    else:
        return tf.reduce_mean(func(gt - rs), name=name)

def weighted(loss, wt):
    return tf.reduce_sum(loss * wt) / (tf.reduce_sum(wt) + 1e-8)

def compute_lxloss(wt, rs, gt, name, mode):
    assert mode == 'l2' or mode == 'l1', 'mode must be l2 or l1'
    func = tf.square if mode == 'l2' else tf.abs
    ndims = len(rs.get_shape().as_list())
    axis = np.arange(1, ndims)
    loss_datum = tf.reduce_mean(func(gt - rs), axis=axis)
    loss = weighted(loss_datum, wt)
    return tf.identity(loss, name=name), tf.identity(loss_datum, name=name+'_datum')

def compute_celoss(wt, logits, gt, name):
    ndims = len(logits.get_shape().as_list())
    axis = np.arange(1, ndims - 1)
    gt = tf.cast(tf.greater(gt, 0), tf.int32)
    gt = tf.squeeze(gt)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=logits)
    loss_datum = tf.reduce_mean(loss, axis=axis)
    loss = weighted(loss_datum, wt)
    return tf.identity(loss, name=name), tf.identity(loss_datum, name=name+'_datum')

def compute_weighted_celoss(wt, logits, gt, ratio, name):
    ratio = min(max(ratio, 0), 1-(1e-8))
    ratio = ratio / (1 - ratio)
    ndims = len(logits.get_shape().as_list())
    axis = np.arange(1, ndims)
    gt = tf.cast(tf.greater(gt, 0), tf.float32)
    gt = tf.concat([1 - gt, gt], axis=3)
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=gt, logits=logits, pos_weight=ratio)
    loss_datum = tf.reduce_mean(loss, axis=axis)
    loss = weighted(loss_datum, wt)
    return tf.identity(loss, name=name), tf.identity(loss_datum, name=name+'_datum')

def compute_IoU(wt, rs, gt, thres, name):
    ndims = len(rs.get_shape().as_list())
    axis = np.arange(1, ndims)
    gt = tf.greater(gt, thres)
    rs = tf.greater(rs, thres)
    TP = tf.reduce_sum(tf.cast(tf.logical_and(rs, gt), tf.float32), axis=axis)
    FP = tf.reduce_sum(tf.cast(tf.logical_and(rs, tf.logical_not(gt)), tf.float32), axis=axis)
    TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(rs), tf.logical_not(gt)), tf.float32), axis=axis)
    FN = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(rs), gt), tf.float32), axis=axis)
    In = tf.identity(TP)
    Un = tf.reduce_sum(tf.cast(tf.logical_or(rs, gt), tf.float32), axis=axis)
    P_datum = TP / (TP + FP + 1e-8)
    R_datum = TP / (TP + FN + 1e-8)
    IoU_datum = In / (Un + 1e-8)
    FPR_datum = FP / (FP + TN + 1e-8)
    F1_datum = 2.0 * P_datum * R_datum / (P_datum + R_datum + 1e-8)
    P = weighted(P_datum, wt)
    R = weighted(R_datum, wt)
    IoU = weighted(IoU_datum, wt)
    FPR = weighted(FPR_datum, wt)
    F1 = weighted(F1_datum, wt)
    return (P, R, IoU, FPR, F1), (P_datum, R_datum, IoU_datum, FPR_datum, F1_datum)

def compute_weight_decay(vars):
    return tf.add_n([tf.nn.l2_loss(v) for v in vars])

def get_valid_batch(batch, valid_len, name):
    shape = batch.get_shape().as_list()
    ndims = len(shape)
    out = tf.slice(batch, [0] * ndims, [valid_len] + [-1] * (ndims - 1), name=name)
    return out



#-----------------------------------------------------------------------------------------------------------------------
def draw_ims(draw_vals, fdout, batch_id=0):
    title_to_path = {}
    for title, ims in draw_vals.items():
        sub_fd = fp.mkdir(os.path.join(fdout, title))
        for i, im in enumerate(ims):
            fname = os.path.join(sub_fd, 'batch{:04d}_{:04d}_[{}].jpg'.format(batch_id, i, title))
            cv2.imwrite(fname, im, [cv2.IMWRITE_JPEG_QUALITY, 100])
        title_to_path[title] = title
    hw = pyhtml.HtmlWriter(fdout, title_to_path)
    hw.Run()

