from config import logger
import pytoolkit.utils as pyutils
import tensorflow as tf, os, time, numpy as np
import pytoolkit.tf_funcs as tfx
from easydict import EasyDict as edict
import traceback, tqdm
import utils

class solver_wrapper(object):
    def __init__(self, net, data_layers, sess, FLAGS):
        self.FLAGS = FLAGS
        self.train_data, self.valid_data = data_layers
        self.net = net
        self.sess = sess
        self.saver_all = tf.train.Saver(self.net.var.all_vars, max_to_keep=None)

        self.summary_writer = tf.summary.FileWriter(self.FLAGS.tb_path, self.sess.graph)

        self.fname_temp_model = os.path.join(self.FLAGS.log_path, 'temp_model.ckpt')
        self.fdout_temp_image = os.path.join(self.FLAGS.output_path, 'temp')
        self.fdout_eval_image = os.path.join(self.FLAGS.output_path, 'valid_only')
        self.fdout_epch_image = os.path.join(self.FLAGS.output_path, 'Epoch{:03d}')
        self.fname_ckpt_model = os.path.join(self.FLAGS.log_path, 'phase_{}_model.ckpt')

        self.pgrbar = pyutils.ProgressBar(self.FLAGS.epoches, 10)

    def Train(self):
        self.train()

    def Finetune(self):
        self.train_data.import_status()
        self.restore_from_checkpoint()
        self.train()

    def Evaluate(self, epoch_id=None):
        if epoch_id is None:
            self.restore_from_checkpoint()
        total_loss_vals = self.validate(epoch_id)
        return total_loss_vals

    def train(self):
        saver = self.saver_all
        status = self.train_data.status

        try:
            self.while_loop(saver)
        except Exception as e:
            traceback.print_exc()
            logger.info('Raise Except, Saving Model...')
            saver.save(self.sess, self.fname_temp_model, status.epoch)
        except KeyboardInterrupt:
            cmd = input('Keyboard Interrupt. Save Model? ([Y|N])')
            if cmd.lower() in ['y', 'yes']:
                logger.info('Keyboard Interrupt, Yes Pressed, Saving Model...')
                saver.save(self.sess, self.fname_temp_model, status.epoch)
            else:
                logger.info('Keyboard Interrupt, No Pressed, Model Ignored')
        else:
            logger.info('Completed!!!')

    def while_loop(self, saver):
        train_data = self.train_data

        while train_data.status.epoch < self.FLAGS.epoches:
            status = edict(**train_data.status) # <== Deep Copy
            train_data.export_status()

            crt_phase = self.compute_crt_phase(status)

            tic = time.time()
            feed_dict, valid_len = self.extract_data_and_build_feed_dict(train_data)
            toc = time.time()
            time_cost_data = toc - tic

            ops = {
                **self.net.out,
                **self.net.loss,
                **self.net.summary,
                crt_phase + '_optim':       self.net.optims[crt_phase],
                'ph_lr':                    self.net.ph_lr,
            }

            tic = time.time()
            op_vals = self.sess.run(ops, feed_dict=feed_dict)
            toc = time.time()
            time_cost_optim = toc - tic

            draw_vals = {k: op_vals[k][:valid_len] for k in self.net.out.keys()}
            loss_vals = {k: op_vals[k] for k in self.net.loss.keys()}
            smry_vals = {k: op_vals[k] for k in self.net.summary.keys()}

            self.log_at_every_itera_end_time(status, crt_phase,
                                             train_data.prgbar, loss_vals,
                                             (time_cost_data, time_cost_optim))

            if status.iteration % 100 == 0:
                pass#pyutils.print_params(logger, self.FLAGS)
            if status.iteration % 5 == 0:
                utils.draw_ims(draw_vals, self.fdout_temp_image)
                [self.summary_writer.add_summary(v, status.iteration) for _, v in smry_vals.items()]
            if status.iteration % 5 == 0 and os.path.exists('stop'):
                raise ValueError('Stop file exists!!! Quit!!!')
            if train_data.status.epoch != status.epoch:
                eval_loss_vals = self.Evaluate(status.epoch)
                logger.info('Eval: Ph {:>12s}: {:6.4f}'.format(crt_phase, eval_loss_vals[crt_phase]))
                [logger.info(line) for line in pyutils.dict_to_string(eval_loss_vals, 3)]
                fname_ckpt_model = self.fname_ckpt_model.format(crt_phase)
                saver.save(self.sess, fname_ckpt_model, status.epoch)
                self.pgrbar.Update(1)
                logger.info('ENTIRE-PROGRESS: {}\n'.format(self.pgrbar.GetBar()))

    def validate(self, epoch_id):
        valid_data = self.valid_data
        valid_data.reset()
        eval_path = self.fdout_eval_image if epoch_id is None else self.fdout_epch_image.format(epoch_id)
        total_loss_vals, total_len = None, None
        pbar = tqdm.tqdm(total=valid_data.data_length)
        while valid_data.status.epoch < 1 and (self.FLAGS.eval_nums == -1 or valid_data.status.iteration < self.FLAGS.eval_nums):
            status = edict(**valid_data.status)  # <== Deep Copy
            feed_dict, valid_len = self.extract_data_and_build_feed_dict(valid_data)
            ops = {
                **self.net.out,
                **self.net.loss,
            }
            op_vals = self.sess.run(ops, feed_dict=feed_dict)

            draw_vals = {k: op_vals[k][:valid_len] for k in self.net.out.keys()}
            loss_vals = {k: op_vals[k] for k in self.net.loss.keys()}

            if total_loss_vals is None:
                total_loss_vals = {k: 0 for k, v in loss_vals.items()}
                total_len = 0

            total_loss_vals = {k: total_loss_vals[k] + v * valid_len for k, v in loss_vals.items()}
            total_len += valid_len

            utils.draw_ims(draw_vals, eval_path, status.iteration)
            pbar.update(valid_len)

        total_loss_vals = {k: v / total_len for k, v in total_loss_vals.items()}
        return total_loss_vals

    def compute_crt_phase(self, status):
        return 'celoss_im'

    def extract_data_and_build_feed_dict(self, data_layer):
        batch_tuple = data_layer.next_batch()
        assert type(batch_tuple) == tuple, 'assert type(batch_tuple) == tuple'
        valid_len = batch_tuple[0].shape[0]
        datum_wt = np.ones([valid_len], np.float32)
        if valid_len != data_layer.BS:
            n = data_layer.BS - valid_len
            batch_tuple = [np.concatenate([b, np.zeros([n, *b.shape[1:]], b.dtype)]) for b in batch_tuple]
            datum_wt = np.concatenate([datum_wt, np.zeros([n], datum_wt.dtype)])

        feed_dict = {
            self.net.ph_in_image:           batch_tuple[0],
            self.net.ph_gt_image:           batch_tuple[1],
            self.net.ph_datum_wt:           datum_wt,
            self.net.ph_lr:                 self.FLAGS.lr,
        }
        return feed_dict, valid_len

    def log_at_every_itera_end_time(self, status, crt_phase, prgbar, loss_vals, times):
        time_cost_data, time_cost_optim = times
        line = 'Ep[{:03d}/{:03d}] Iter{: 6d}, {}, (D{:3.2f}s R{:3.2f}s), Ph: {:8s}, Loss: {:6.4f}'.format(
            status.epoch, self.FLAGS.epoches, status.iteration, prgbar.GetBar(),
            time_cost_data, time_cost_optim,
            crt_phase, loss_vals[crt_phase]
        )
        logger.info(line)
        lines = pyutils.dict_to_string(loss_vals, 3)
        for line in lines:
            logger.info(line)

    def restore_from_checkpoint(self):
        ckpt_path, _, unmatched_vars = tfx.restore_from_checkpoint(self.sess, self.FLAGS.log_path, self.net.var.all_vars)
        if ckpt_path is None:
            logger.warning('The entire model file not found! So weights are still initial!')
        else:
            logger.info('The entire model restored from: ' + ckpt_path)
            if len(unmatched_vars) > 0:
                logger.warning('The following vars contain no values in checkpoint file')
                [logger.warning(v.op.name) for v in unmatched_vars]
