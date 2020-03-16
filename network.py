from config import FLAGS, logger
from easydict import EasyDict as edict
import tensorflow as tf
import pytoolkit.tf_funcs as tfx
import utils

class CENet(object):
    def __init__(self, FLAGS):
        self.batch_size = bs        = FLAGS.batch_size # bs: batch size in single GPU
        self.image_size = [szh,szw] = FLAGS.image_size
        self.num_gpus   = nG        = FLAGS.num_gpus
        self.FLAGS      = FLAGS

        BS = bs * nG

        self.ph_in_image    = tf.placeholder(tf.uint8,   shape=(BS, szh, szw, 3), name='ph_in_image')
        self.ph_gt_image    = tf.placeholder(tf.uint8,   shape=(BS, szh, szw, 1), name='ph_gt_image')
        self.ph_datum_wt    = tf.placeholder(tf.float32, shape=(BS),              name='ph_datum_wt')
        self.ph_lr          = tf.placeholder(tf.float32, shape=None,              name='ph_lr')

        self.optimizer      = tf.train.AdamOptimizer(learning_rate=self.ph_lr, beta1=0.9)

        self.branches       = [None] * nG

        with tf.device('/cpu:0'):
            for gpu_id in range(nG):
                with tf.device('/gpu:%d' % gpu_id), tf.variable_scope('cpu_variables', reuse=gpu_id>0):
                    logger.info('branch: %d' % gpu_id)
                    start_idx = bs * gpu_id
                    phs = [
                        tf.slice(self.ph_in_image, [start_idx, 0, 0, 0], [bs, -1, -1, -1], name='ph_in_image_br%d' % gpu_id),
                        tf.slice(self.ph_gt_image, [start_idx, 0, 0, 0], [bs, -1, -1, -1], name='ph_gt_image_br%d' % gpu_id),
                        tf.slice(self.ph_datum_wt, [start_idx],          [bs],             name='ph_datum_wt_br%d' % gpu_id),
                    ]
                    self.branches[gpu_id] = CENet_Branch(FLAGS, phs, gpu_id, self.optimizer)

        self.out     = self.merge_results(self.branches)
        self.loss    = self.merge_losses(self.branches)
        self.optims  = self.merge_optims(self.branches)
        self.summary = self.merge_summaries(self.loss, self.var.all_vars)

    def merge_results(self, brs):
        return edict({k: tf.concat([b.out[k] for b in brs], 0, name=k) for k in brs[0].out.keys()})

    def merge_losses(self, brs):
        out = edict({})
        for k in brs[0].loss.keys():
            total_sum_loss = tf.reduce_sum([b.loss[k] * b.sum_wt for b in brs])
            total_weights = tf.reduce_sum([b.sum_wt for b in brs])
            v = tf.identity(total_sum_loss / (total_weights + 1e-8), name=k+'_loss')
            out[k] = v
        return out

    def merge_optims(self, brs):
        phase_grad = edict({k : tfx.average_gradients([b.phase[k] for b in brs], [b.sum_wt for b in brs])
                            for k in brs[0].phase.keys()})
        return edict({k : self.optimizer.apply_gradients(v) for k, v in phase_grad.items()})

    def merge_summaries(self, loss, var):
        summary_loss = edict({k + '_summary': tf.summary.scalar(k, v) for k, v in loss.items()})
        summary_weights = edict()
        for v in var:
            name = v.op.name + '_summary'
            summary_weights.update({name: tf.summary.histogram(name, v)})
        return edict({**summary_loss, **summary_weights})

    @property
    def vgg_api(self):
        return self.branches[0].vgg_api

    @property
    def var(self):
        return self.branches[0].var

class CENet_Branch(object):
    def __init__(self, FLAGS, phs, gpu_id, optimizer):
        self.batch_size = FLAGS.batch_size
        self.gpu_id = gpu_id
        self.FLAGS = FLAGS

        self.ph_in_image = utils.to_float_tensor(phs[0])
        self.ph_gt_image = utils.to_float_tensor(phs[1])
        self.ph_datum_wt = phs[2]



        # G-Net
        self.rs_ou_image_train, self.layers = self.G_Net_2D_v1(self.ph_in_image, is_train=True, reuse=False)
        self.rs_ou_image_valid, _           = self.G_Net_2D_v1(self.ph_in_image, is_train=False, reuse=True)

        # Loss
        self.loss   = self.compute_losses()
        self.var    = self.collect_vars()
        self.phase  = self.optimize(optimizer)

        # Out
        self.out = edict({
            'input':              utils.to_uint8_tensor(self.ph_in_image),
            'rs_ou_image_valid':  utils.to_uint8_tensor(self.rs_ou_image_valid),
            'gt':                 utils.to_uint8_tensor(self.ph_gt_image),
            'rs_ou_image_train':  utils.to_uint8_tensor(self.rs_ou_image_train),
        })

    def G_Net_2D(self, im, is_train, reuse):
        with tf.variable_scope('generator_im', reuse=reuse) as scope:
            layers = []
            input = im
            chns = [64] + [128] * 2 + [256] * 9 + [128] * 2 + [64, 32, 3]
            etas = [1] * 6 + [2, 4, 8, 16] + [1] * 7
            kszs = [5] + [3] * 11 + [4, 3, 4, 3, 3]
            stps = [1, 2, 1, 2] + [1] * 8 + [2, 1, 2, 1, 1]
            dcvs = [False] * 12 + [True, False, True, False, False]
            usbn = [True] * 16 + [False]
            usrl = usbn
            for k in range(17):
                if k == 0:
                    temp = input
                elif k < 9:
                    temp = layers[-1]
                elif k < 15:
                    temp = tf.concat([layers[-1], layers[15 - k]], axis=3)
                else:
                    temp = layers[-1]
                layers += [utils.conv_layer_2d(temp, chns[k], etas[k], kszs[k], stps[k],
                                               'l%02d' % k, dcvs[k], usbn[k], is_train, usrl[k])]
            return tf.nn.tanh(layers[-1]), layers

    def G_Net_2D_v1(self, im, is_train, reuse):
        with tf.variable_scope('generator_im', reuse=reuse) as scope:
            layers = []
            input = im
            chns = [64] + [128] * 2 + [256] * 9 + [128] * 2 + [64, 32, 3]
            chns = [x // 8 for x in chns[:-1]] + [chns[-1]]
            etas = [1] * 6 + [2, 4, 8, 16] + [1] * 7
            kszs = [5] + [3] * 11 + [4, 3, 4, 3, 3]
            stps = [1, 2, 1, 2] + [1] * 8 + [2, 1, 2, 1, 1]
            dcvs = [False] * 12 + [True, False, True, False, False]
            usbn = [True] * 16 + [False]
            usrl = usbn
            for k in range(17):
                if k == 0:
                    temp = input
                elif k < 9:
                    temp = layers[-1]
                elif k < 15:
                    temp = tf.concat([layers[-1], layers[15 - k]], axis=3)
                else:
                    temp = layers[-1]
                layers += [utils.conv_layer_2d(temp, chns[k], etas[k], kszs[k], stps[k],
                                               'l%02d' % k, dcvs[k], usbn[k], is_train, usrl[k])]
            return tf.nn.tanh(layers[-1]), layers




    def collect_vars(self):
        t_vars   = [var for var in tf.global_variables() if 'VGG' not in var.name]
        vgg_vars = [var for var in tf.global_variables() if 'VGG' in var.name]

        g_vars_im = [var for var in t_vars if 'generator_im' in var.name]
        d_vars_im = [var for var in t_vars if 'discriminator_im' in var.name]

        assert len(g_vars_im) + len(d_vars_im) == len(t_vars), 'vars number inconsistent! (1)'
        assert len(t_vars) + len(vgg_vars) == len(tf.global_variables()), 'vars number inconsistent! (2)'

        var = edict({
            'all_vars':  t_vars,
            'g_im_vars': g_vars_im,
            'd_im_vars': d_vars_im,
            'vgg_vars':  vgg_vars
        })
        return var

    def compute_losses(self):
        self.sum_wt = tf.reduce_sum(self.ph_datum_wt)

        l1_im, l1_im_datum = utils.compute_lxloss(self.ph_datum_wt,
                                                  self.rs_ou_image_train,
                                                  self.ph_gt_image, name='l1loss_im', mode='l1')
        loss = edict({
            'l1loss_im':            l1_im,
        })
        return loss

    def optimize(self, optimizer):
        phases = edict({
            'l1loss_im': 'g_im_vars',
            'l2loss_im': 'g_im_vars',
            'g_loss_im': 'g_im_vars',
            'd_loss_im': 'd_im_vars',
        })
        phase_opts = edict({})
        logger.info('{:-^72}'.format('Experiment: {:2d} Phases'.format(len(phases))))
        for idx, (phase_name, vname) in enumerate(phases.items()):
            if phase_name in self.loss and vname in self.var:
                phase_opts[phase_name] = optimizer.compute_gradients(self.loss[phase_name], var_list=self.var[vname])
                logger.info('{:d}) {}'.format(idx, phase_name))
            else:
                logger.warning('{:d}) WARNING! LOSS: {:>10s} or VARS: {:>10s} NOT EXIST!'.format(idx, phase_name, vname))
        logger.info('-' * 72)
        return phase_opts