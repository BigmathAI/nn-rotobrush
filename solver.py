from config import logger
import pytoolkit.utils as pyutils
import tensorflow as tf, os, time, numpy as np
import traceback
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
        self.fname_ckpt_model = os.path.join(self.FLAGS.log_path, 'phase_{}_model.ckpt')

    def Train(self):
        self.train()

    def Finetune(self):
        self.train()

    def Evaluate(self):
        pass

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
            status = train_data.status
            train_data.export_status()

            crt_phase = self.compute_crt_phase(status)

            tic = time.time()
            feed_dict = self.extract_data_and_build_feed_dict(train_data)
            toc = time.time()
            time_cost_data = toc - tic

            ops = {
                **self.net.out,
                **self.net.loss,
                **self.net.summary,
                crt_phase + '_optim':       self.net.optims[crt_phase],
                'ph_lr':                    self.net.ph_lr,
                'ph_valid_len':             self.net.ph_valid_len,
            }

            tic = time.time()
            op_vals = self.sess.run(ops, feed_dict=feed_dict)
            toc = time.time()
            time_cost_optim = toc - tic

            draw_vals = {k: op_vals[k][:op_vals['ph_valid_len']] for k in self.net.out.keys()}
            loss_vals = {k: op_vals[k] for k in self.net.loss.keys()}
            smry_vals = {k: op_vals[k] for k in self.net.summary.keys()}

            self.log_at_every_itera_end_time(status, crt_phase, loss_vals, (time_cost_data, time_cost_optim))

            if status.iteration % 100 == 0:
                pyutils.print_params(logger, self.FLAGS)
            if status.iteration % 5 == 0:
                utils.draw_ims(draw_vals, self.fdout_temp_image)
                [self.summary_writer.add_summary(v, status.iteration) for _, v in smry_vals.items()]
            if status.iteration % 5 == 0 and os.path.exists('stop'):
                raise ValueError('Stop file exists!!! Quit!!!')
            if train_data.status.epoch != status.epoch:
                self.Evaluate()
                fname_ckpt_model = self.fname_ckpt_model.format(crt_phase)
                saver.save(self.sess, fname_ckpt_model, status.epoch)

    def compute_crt_phase(self, status):
        return 'l1loss_im'

    def extract_data_and_build_feed_dict(self, data_layer):
        batch_tuple = data_layer.next_batch()
        assert type(batch_tuple) == tuple, 'assert type(batch_tuple) == tuple'
        valid_len = batch_tuple[0].shape[0]
        if valid_len != data_layer.BS:
            n = data_layer.BS - valid_len
            batch_tuple = [np.concatenate([b, np.zeros([n] + list(b.shape[1:]), b.dtype)]) for b in batch_tuple]

        feed_dict = {
            self.net.ph_in_image:           batch_tuple[0],
            self.net.ph_gt_image:           batch_tuple[1],
            self.net.ph_valid_len:          valid_len,
            self.net.ph_lr:                 self.FLAGS.lr,
        }
        return feed_dict

    def log_at_every_itera_end_time(self, status, crt_phase, loss_vals, times):
        time_cost_data, time_cost_optim = times
        line = 'Ep[{:03d}/{:03d}] Iter{: 6d}, (D{:3.2f}s R{:3.2f}s), Ph: {:8s}, Loss: {:6.4f}'.format(
            status.epoch, self.FLAGS.epoches, status.iteration, time_cost_data, time_cost_optim,
            crt_phase, loss_vals[crt_phase]
        )
        logger.info(line)


