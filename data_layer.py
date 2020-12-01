import pytoolkit.files as fp, pytoolkit.utils as pyutils
import numpy as np, cv2, os, json, random, math, logging, time, tqdm, psutil
from easydict import EasyDict as edict
from config import FLAGS, logger
from functools import lru_cache

class data_layer_base(object):
    def __init__(self, FLAGS, phase):
        self.FLAGS = FLAGS
        self.phase = phase

        self.data_path = FLAGS.data_path
        self.BS = self.FLAGS.batch_size * self.FLAGS.num_gpus

        self.filelist_path = os.path.join(self.data_path, phase + '-list.npy')
        self.orig_data_indices = np.load(self.filelist_path, 'r')
        self.curr_data_indices = None
        self.data_length = len(self.orig_data_indices)

        self.N, self.data_npy, self.shape = self.preload(os.path.join(self.data_path, 'npy'))
        self.index_map = self.build_index_map(self.data_npy)

        self.un_shuffle()
        if self.phase == 'train':
            self.random_shuffle()

        self.status = self.init_status()
        self.prgbar = pyutils.ProgressBar(self.data_length, 5)
        self.last_batch = None
        logger.info('data length: {:>6s}: {:04d}'.format(self.phase, self.data_length))

    def preload(self, fd_npy):
        fnames_npy = fp.dir(fd_npy, '.npy')
        data_npy = [np.load(fname, 'r') for fname in fnames_npy]
        Process = psutil.Process(os.getpid())
        #tr = tqdm.trange(len(fnames_npy), desc='Memory Cost', leave=True)
        #for d in data_npy:
        #    logger.info(d.shape)
        Ns = [d.shape[0] for d in data_npy]
        N = sum(Ns)
        return N, data_npy, data_npy[0].shape[1:]

    def build_index_map(self, data_npy):
        #import ipdb
        #ipdb.set_trace()
        Ns = [d.shape[0] for d in data_npy]
        N = sum(Ns)
        index_map = np.zeros([N, 2], np.int32)
        s = 0
        for k in range(len(Ns)):
            ns = Ns[k]
            index_map[s:s+ns, 0] = np.ones([ns], np.int32) * k  # <= file id
            index_map[s:s+ns, 1] = np.arange(ns)                # <= id in the file
            s += ns
        return index_map

    def random_shuffle(self):
        random.shuffle(self.curr_data_indices)
        self.shuffled = True

    def un_shuffle(self):
        self.curr_data_indices = self.orig_data_indices.copy()
        self.shuffled = False

    def load_raw_and_processing(self, s, e):
        raw_data_batch = self.load_raw_from_npy(s, e)
        return self.prepare_data_to_feed(raw_data_batch)

    def prepare_data_to_feed(self, data_batch):
        raise NotImplementedError('This is an Abstract method, Must be implemented by the subclass.')

    def load_raw_from_npy(self, s, e):
        tic = time.time()
        def FUNC_COPY_DATA(data_batch, sample_ids, index_map, data_npy, data_range):
            @lru_cache(maxsize=None)
            def lru_cached_data(sample_id):
                fid, datum_id = index_map[sample_id]
                raw_datum = data_npy[fid][datum_id]
                return raw_datum
            s, e = data_range
            for k in range(s, e):
                sample_id = sample_ids[k]
                #fid, datum_id = index_map[sample_id]
                #raw_datum = data_npy[fid][datum_id]
                data_batch[k] = lru_cached_data(sample_id)#raw_datum
        data_batch = np.zeros([e - s] + list(self.shape), np.uint8)
        if False and self.FLAGS.USE_MULTI_THREADS is True:
            sample_ids = self.curr_data_indices[s:e]
            FUNC_ARGS = [data_batch, sample_ids, self.index_map, self.data_npy, None]
            pyutils.do_multi_threads(e - s, self.FLAGS.NUM_THREADS, FUNC_COPY_DATA, FUNC_ARGS)
        else:
            for i in range(s, e):
                sample_id = self.curr_data_indices[i]
                fid, datum_id = self.index_map[sample_id]
                raw_datum = self.data_npy[fid][datum_id]
                data_batch[i - s] = raw_datum
        toc = time.time()
        #print('LOAD: %f' % (toc - tic))
        return data_batch

    def init_status(self):
        status = edict({
            'epoch':                    0,
            'iteration':                0,
            'iter_cur_epoch':           0,
            'start_idx':                0,
            #-----------------------------------------------------------------------------------------------------------
            'perc':                     0,
        })
        return status

    def export_status(self):
        with open(os.path.join(self.FLAGS.log_path, 'data_status.txt'), 'w') as f:
            json.dump(self.status, f)

    def import_status(self):
        status_fname = os.path.join(self.FLAGS.log_path, 'data_status.txt')
        try:
            with open(status_fname, 'r') as f:
                self.status = edict(json.load(f))
                self.prgbar.Restore(self.status.start_idx)
        except:
            logger.warning('No status file found, will create new')

    def reset(self):
        self.status = self.init_status()
        self.prgbar.Reset()
        self.un_shuffle()

    def get_status(self):
        return self.status

    #-------------------------------------------------------------------------------
    def _pick_up_exact(self, s): # <= pick up a batch, without update any status
        assert s < self.data_length, 'assert s < self.data_length'
        e = min(self.data_length, s + self.BS)
        rst = self.load_raw_and_processing(s, e)
        return rst

    def exact_batch(self, s):
        self.last_batch = rst = self._pick_up_exact(s)
        self.status.iteration += 1
        return rst

    def prev_batch(self):
        if self.last_batch is not None:
            rst = self.last_batch
            self.status.iteration += 1
        else:
            rst = self.crt_batch()
        return rst

    def crt_batch(self):
        assert self.status.start_idx < self.data_length, 'self.status.start_idx < self.data_length'
        return self.exact_batch(self.status.start_idx)

    def random_batch(self):
        s = np.random.randint(self.data_length)
        return self.exact_batch(s)

    def next_batch1(self):
        rst = self.crt_batch()
        s = self.status.start_idx
        e = min(self.data_length, s + self.BS)
        if e == self.data_length:
            self.status.start_idx = 0
            self.status.epoch += 1
            self.status.iter_cur_epoch = 0
            self.prgbar.Reset()
            if self.phase == 'train':
                self.random_shuffle()
        else:
            self.status.start_idx = e
            self.status.iter_cur_epoch += 1
            self.prgbar.Update(e - s)
        self.status.perc = self.status.start_idx / self.data_length
        return rst

    def next_batch(self):
        if self.status.start_idx == self.data_length:
            self.status.start_idx = 0
            self.status.iter_cur_epoch = 0
            self.prgbar.Reset()
            if self.phase == 'train':
                self.random_shuffle()
        s = self.status.start_idx
        e = min(self.data_length, s + self.BS)
        rst = self.crt_batch()
        self.status.start_idx = e
        if e == self.data_length:
            self.status.epoch += 1
        self.status.iter_cur_epoch += 1
        self.prgbar.Update(e - s)
        self.status.perc = self.status.start_idx / self.data_length
        return rst

    #-------------------------------------------------------------------------------
    def num_iters_per_epoch(self):
        return int(math.ceil(self.data_length / self.BS))

#-----------------------------------------------------------------------------------
class data_layer_2d(data_layer_base):
    def __init__(self, FLAGS, phase):
        data_layer_base.__init__(self, FLAGS, phase)

    def load_raw_from_npy(self, s, e):
        return data_layer_base.load_raw_from_npy(self, s, e)

    def prepare_data_to_feed(self, raw_data_batch):
        tic = time.time()
        N, H, W, C = raw_data_batch.shape
        oH, oW = self.FLAGS.image_size
        dl_in_image = np.zeros([N, oH, oW, 3], np.uint8)
        dl_gt_label = np.zeros([N, oH, oW], np.uint8)

        rH = (int(0.5 * H), int(0.99 * H) + 1)
        rW = (int(0.5 * W), int(0.99 * W) + 1)

        xH = np.random.randint(rH[0], rH[1])
        xW = np.random.randint(rW[0], rW[1])
        offset_y = np.random.randint(H - xH + 1)
        offset_x = np.random.randint(W - xW + 1)

        def FUNC_EXTRACT_DATA(dl_in_image, dl_gt_label, raw_data_batch, new_size, offsets, data_range):
            s, e = data_range
            offset_y, offset_x, xH, xW = offsets
            for k in range(s, e):
                datum = raw_data_batch[k]
                if self.phase == 'train':
                    datum = datum[offset_y:offset_y + xH, offset_x:offset_x + xW]
                tmp = cv2.resize(datum, new_size, interpolation=cv2.INTER_NEAREST)
                if self.phase == 'train':
                    if np.random.randint(4) % 4 == 0:
                        tmp = tmp[::-1,:]
                    if np.random.randint(2) % 2 == 0:
                        tmp = tmp[:,::-1]
                dl_in_image[k] = tmp[:,:,:3]
                dl_gt_label[k] = tmp[:,:,3]

        if self.FLAGS.USE_MULTI_THREADS is True:
            FUNC_ARGS = [dl_in_image, dl_gt_label, raw_data_batch, (oW, oH), (offset_y, offset_x, xH, xW), None]
            pyutils.do_multi_threads(N, self.FLAGS.NUM_THREADS, FUNC_EXTRACT_DATA, FUNC_ARGS)
        else:
            if self.phase == 'train':
                raw_data_batch = raw_data_batch[:, offset_y:offset_y + xH, offset_x:offset_x + xW, :]
                # if np.random.randint(2) % 2 == 0:
                #    raw_data_batch = raw_data_batch[:,::-1,:,:]
                # if np.random.randint(2) % 2 == 0:
                #    raw_data_batch = raw_data_batch[:,:,::-1,:]
                # if np.random.randint(2) % 2 == 0:
                #    raw_data_batch = raw_data_batch[:,:,:,list(np.random.permutation(3)) + [3]]

            for k, datum in enumerate(raw_data_batch):
                tmp = cv2.resize(datum, (oW, oH), interpolation=cv2.INTER_NEAREST)
                if self.phase == 'train':
                    if np.random.randint(4) % 4 == 0:
                        tmp = tmp[::-1,:]
                    if np.random.randint(2) % 2 == 0:
                        tmp = tmp[:,::-1]
                    #if np.random.randint(2) % 2 == 0:
                    #    tmp = tmp[:,:,list(np.random.permutation(3)) + [3]]
                dl_in_image[k] = tmp[:,:,:3]
                dl_gt_label[k] = tmp[:,:,3]

        if len(dl_gt_label.shape) == 3:
            dl_gt_label = np.expand_dims(dl_gt_label, 3)

        toc = time.time()
        #print('COST: %f' % (toc - tic))

        return (dl_in_image, dl_gt_label)