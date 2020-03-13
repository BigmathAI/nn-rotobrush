import pytoolkit.tf_funcs as tfx
import tensorflow as tf, numpy as np, os

def conv_layer_2d(im, chnls, knls, stps, name,
                  isdeconv=False,
                  with_b=True,
                  usebn=False,
                  is_train=True,
                  userelu=False,
                  uselrelu=False):
    assert not (userelu and uselrelu), 'userelu and uselrelu cannot be both true!!!'
    with tf.variable_scope(name) as scope:
        func = tfx.deconv2d if isdeconv else tfx.conv2d
        conv_name = 'deconv' if isdeconv else 'conv'
        lconv = func(im, chnls, (knls, knls), (stps, stps), name=conv_name, with_b=with_b)
        lconv = tfx.bn_new(lconv, phase_train=is_train) if usebn else lconv
        lconv = tfx.relu(lconv) if userelu else lconv
        lconv = tfx.lrelu(lconv) if uselrelu else lconv
        return lconv

class vgg(object):
    def __init__(self, vgg_path):
        self.path = vgg_path
        self.vgg_vars = []

    def net(self, im, reuse):
        VGG_MEAN = [103.939, 116.779, 123.68]  # b / g / r

        def _vgg_conv_a(input, chns, idx):
            layers  = [conv_layer_2d(input,      chns, 3, 1, 'conv' + str(idx) + '_1', False, True, False, False, True, False)]
            layers += [conv_layer_2d(layers[-1], chns, 3, 1, 'conv' + str(idx) + '_2', False, True, False, False, True, False)]
            layers += [tfx.max_pool2d(layers[-1], (2, 2), (2, 2), name='pool' + str(idx))]
            return layers

        def _vgg_conv_b(input, chns, idx):
            layers  = [conv_layer_2d(input,      chns, 3, 1, 'conv' + str(idx) + '_1', False, True, False, False, True, False)]
            layers += [conv_layer_2d(layers[-1], chns, 3, 1, 'conv' + str(idx) + '_2', False, True, False, False, True, False)]
            layers += [conv_layer_2d(layers[-1], chns, 3, 1, 'conv' + str(idx) + '_3', False, True, False, False, True, False)]
            layers += [conv_layer_2d(layers[-1], chns, 3, 1, 'conv' + str(idx) + '_4', False, True, False, False, True, False)]
            layers += [tfx.max_pool2d(layers[-1], (2, 2), (2, 2), name='pool' + str(idx))]
            return layers

        bs, h, w, c = tfx._shape_2d(im)
        assert h == 224 and w == 224 and c == 3, 'INPUT OF VGG MUST BE 224x224x3'

        with tf.variable_scope('VGG19', reuse=reuse) as scope:
            im_scaled = (im + 1.0) * 127.5
            b, g, r = tf.split(im_scaled, 3, 3)
            assert b.get_shape().as_list()[1:] == [224, 224, 1]
            assert g.get_shape().as_list()[1:] == [224, 224, 1]
            assert r.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)
            assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
            layers, feats = [], {}
            # conv1, conv2
            layers += _vgg_conv_a(bgr, 64, 1)
            layers += _vgg_conv_a(layers[-1], 128, 2)
            feats['conv2_4'] = layers[-1]
            # conv3, conv4, conv5
            layers += _vgg_conv_b(layers[-1], 256, 3)
            layers += _vgg_conv_b(layers[-1], 512, 4)
            feats['conv4_4'] = layers[-1]
            layers += _vgg_conv_b(layers[-1], 512, 5)
            # fc6, fc7, fc8
            rs = tf.reshape(layers[-1], shape=[bs, -1])
            layers += [tfx.relu(tfx.fc(rs, 4096, name='fc6'), name='fc6')]
            layers += [tfx.relu(tfx.fc(layers[-1], 4096, name='fc7'), name='fc7')]
            layers += [tfx.fc(layers[-1], 1000, name='fc8')]

        if reuse is False:
            self.vgg_vars = [v for v in tf.global_variables() if 'VGG19' in v.name]

        return layers, feats

    def restore(self, sess):
        vgg_vars = self.vgg_vars
        vgg_fname = self.path
        def pairwise(iterable):
            a = iter(iterable)
            return zip(a, a)
        if not os.path.exists(vgg_fname):
            return False
        npz = np.load(vgg_fname, encoding='latin1').item()
        vgg_vars = [x for x in pairwise(vgg_vars)]
        vgg_wgts = sorted(npz.items())
        ops = []
        for var, (name, (W, b)) in zip(vgg_vars, vgg_wgts):
            ops += [tf.assign(var[0], W), tf.assign(var[1], b)]
        sess.run(ops)
        return True