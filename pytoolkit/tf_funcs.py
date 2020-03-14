import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow

# 2D OPERATIONS:
def _shape_2d(input):
    shape = tf.shape(input)
    n, h0, w0, c = input.get_shape().as_list()
    _, h1, w1, _ = [shape[i] for i in range(len(input.get_shape()))]
    return n, h1 if h0 is None else h0, w1 if w0 is None else w0, c

def _conv2d_weights(x, out_dim, kernel_size, stddev, is_deconv=False, with_b=True):
    c = _shape_2d(x)[-1]
    w_shape  = [kernel_size[0], kernel_size[1]]
    w_shape += [out_dim, c] if is_deconv else [c, out_dim]
    W = tf.get_variable('w', w_shape,   initializer=tf.random_normal_initializer(stddev=stddev))
    b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0)) if with_b else None
    return W, b

def conv2d(x, output_dim, kernel_size=(3,3), stride_size=(1,1), padding='SAME', stddev=0.02, name='conv2d', with_b=True):
    with tf.variable_scope(name):
        W, b = _conv2d_weights(x, output_dim, kernel_size, stddev, False, with_b)
        conv = tf.nn.conv2d(x, W, strides=[1, stride_size[0], stride_size[1], 1], padding=padding)
        return conv if b is None else tf.nn.bias_add(conv, b)

def dilated_conv2d(x, output_dim, rate, kernel_size=(3,3), padding='SAME', stddev=0.02, name='dilated_conv2d'):
    with tf.variable_scope(name):
        W, b = _conv2d_weights(x, output_dim, kernel_size, stddev, False)
        conv = tf.nn.atrous_conv2d(x, W, rate=rate, padding=padding)
        return tf.nn.bias_add(conv, b)

def deconv2d(x, output_dim, kernel_size, stride_size, stddev=0.02, name='deconv2d', with_b=True):
    with tf.variable_scope(name):
        W, b = _conv2d_weights(x, output_dim, kernel_size, stddev, True, with_b)
        n, h, w, c = _shape_2d(x)
        output_shape = [n, h * stride_size[0], w * stride_size[1], output_dim]
        output_shape_tensor = tf.stack(output_shape)
        strides = [1, stride_size[0], stride_size[1], 1]
        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape_tensor, strides=strides)
        deconv = deconv if b is None else tf.nn.bias_add(deconv, b)
        return tf.reshape(deconv, output_shape)

def max_pool2d(x, kernel_size=(2,2), stride_size=(2,2), padding='SAME', name='max_pool'):
    ksize = [1, kernel_size[0], kernel_size[1], 1]
    strides = [1, stride_size[0], stride_size[1], 1]
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding, name=name)

# 3D OPERATIONS:
def _shape_3d(input):
    shape = tf.shape(input)
    n, f, h0, w0, c = input.get_shape().as_list()
    _, _, h1, w1, _ = [shape[i] for i in range(len(input.get_shape()))]
    return n, f, h1 if h0 is None else h0, w1 if w0 is None else w0, c

def _conv3d_weights(x, out_dim, kernel_size, stddev, is_deconv=False, with_b=True):
    c = _shape_3d(x)[-1]
    w_shape  = [kernel_size[0], kernel_size[1], kernel_size[2]]
    w_shape += [out_dim, c] if is_deconv else [c, out_dim]
    W = tf.get_variable('w', w_shape,   initializer=tf.random_normal_initializer(stddev=stddev))
    b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0)) if with_b else None
    return W, b

def conv3d(x, output_dim, kernel_size=(3,3,3), stride_size=(1,1,1), padding='SAME', stddev=0.02, name='conv3d', with_b=True):
    with tf.variable_scope(name):
        W, b = _conv3d_weights(x, output_dim, kernel_size, stddev, False, with_b)
        conv = tf.nn.conv3d(x, W, strides=[1, stride_size[0], stride_size[1], stride_size[2], 1], padding=padding)
        return conv if b is None else tf.nn.bias_add(conv, b)

def deconv3d(x, output_dim, kernel_size, stride_size, stddev=0.02, name='deconv3d', with_b=True):
    with tf.variable_scope(name):
        W, b = _conv3d_weights(x, output_dim, kernel_size, stddev, True, with_b)
        n, f, h, w, c = _shape_3d(x)
        output_shape = [n, f * stride_size[0], h * stride_size[1], w * stride_size[2], output_dim]
        output_shape_tensor = tf.stack(output_shape)
        strides = [1, stride_size[0], stride_size[1], stride_size[2], 1]
        deconv = tf.nn.conv3d_transpose(x, W, output_shape=output_shape_tensor, strides=strides)
        deconv = deconv if b is None else tf.nn.bias_add(deconv, b)
        return tf.reshape(deconv, output_shape)

# OTHERS:
def relu(x, name='relu'):
    with tf.variable_scope(name):
        out = tf.nn.relu(x)
        return out

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        out = tf.maximum(x, leak * x)
        return out

def fc(x, output_dim, stddev=0.02, bias_start=0.0, name='fc'):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], output_dim], tf.float32, tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(bias_start))
        return tf.matmul(x, w) + b

def bn_new(x, phase_train=True, name='batch_norm'):
    with tf.variable_scope(name):
        x_shape = x.get_shape()
        params_shape = x.shape[-1:]
        axis = list(range(len(x_shape) - 1))
        beta  = tf.get_variable('beta',  params_shape, initializer=tf.zeros_initializer(), trainable=phase_train)
        gamma = tf.get_variable('gamma', params_shape, initializer=tf.random_normal_initializer(1., 0.02), trainable=phase_train)
        pop_mean = tf.get_variable('pop_mean', params_shape, initializer=tf.zeros_initializer(),      trainable=False)
        pop_var  = tf.get_variable('pop_var',  params_shape, initializer=tf.constant_initializer(1.), trainable=False)
        mean, variance = tf.nn.moments(x, axis)
        from tensorflow.python.training import moving_averages
        decay = 0.9
        update_pop_mean = moving_averages.assign_moving_average(pop_mean, mean, decay, zero_debias=False)
        update_pop_var  = moving_averages.assign_moving_average(pop_var,  variance,  decay, zero_debias=False)
        if phase_train:
            with tf.control_dependencies([update_pop_mean, update_pop_var]):
                normed = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-5)
        else:
            normed = tf.nn.batch_normalization(x, pop_mean, pop_var, beta, gamma, 1e-5)
        return normed

def bn(x, name='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, 
            updates_collections=None, epsilon=self.epsilon, 
            scale=True, scope=self.name)

def atrous_conv3d(input, filter, rate, padding, strides=None, name=None):
    return tf.nn.convolution(input=input, filter=filter, padding=padding, dilation_rate=np.broadcast_to(rate, (3, )), strides=strides, name=name)

def tensors_in_checkpoint_file(filename, need_value=False):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return [(key, shape, reader.get_tensor(key) if need_value else None) for key, shape in sorted(var_to_shape_map.items())]

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        if grads[0] is not None:
            grad = tf.stack(grads, 0)
            grad = tf.reduce_mean(grad, 0)
        else:
            grad = None
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def average_gradients_with_valid_lens(tower_grads, valid_lens):
    average_grads = []
    valid_lens = [tf.cast(vl, tf.float32) for vl in valid_lens]
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        if grads[0] is not None:
            grads = [g * vl for g, vl in zip(grads, valid_lens)]
            grad = tf.stack(grads, 0)
            grad = tf.reduce_sum(grad, 0)
            grad /= tf.reduce_sum(valid_lens)
        else:
            grad = None
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads