import threading, math, logging, coloredlogs, os, time

'''
func()'s args: last one must be a tuple representing the range
'''
def do_multi_threads(n, GLOBAL_NUM_THREADS, func, func_args):
    threads = [None] * GLOBAL_NUM_THREADS
    N_BASE_PER_SAMPLE = int(math.floor(n / GLOBAL_NUM_THREADS))
    REST_SAMPLES = n - N_BASE_PER_SAMPLE * GLOBAL_NUM_THREADS
    s, e = 0, 0
    for k in range(GLOBAL_NUM_THREADS):
        s = e
        e = s + N_BASE_PER_SAMPLE
        e = e + 1 if k < REST_SAMPLES else e
        func_args[-1] = (s, e)
        t = threading.Thread(name='threading', target=func, args=(func_args))
        t.start()
        threads[k] = t
    [t.join() for t in threads]

def create_logger(log_filename):
    log_format = '%(asctime)s %(filename)15s[line:%(lineno)5d] %(message)s'
    log_datefmt = '%m/%d/%Y %H:%M:%S'
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format=log_format, datefmt=log_datefmt)
    logger = logging.getLogger(__name__)
    coloredlogs.install(level=0, logger=logger, fmt=log_format)
    return logger

def print_params(logger, FLAGS):
    logger.debug('{:-^72}'.format('CONFIGURATION PARAMERERS:'))
    lines = ['|--' + '{:<32}'.format(k) + '== ' + str(v) for k, v in FLAGS.flag_values_dict().items()]
    with open(os.path.join(FLAGS.result_path, 'params.txt'), 'w') as f:
        [(logger.debug(line), f.write(line + '\n')) for line in lines]
    logger.debug('-' * 72)

def secs_to_time(sec):
    if sec is None:
        return ''
    if int(sec / 3600) >= 1:
        fmt = '%H:%M:%S'
    else:
        fmt = '%M:%S'
    return time.strftime(fmt, time.gmtime(sec))

class ProgressBar(object):
    def __init__(self, total, len=10):
        self.NUM_BLOCKS = len
        self.Total = total
        self.bar = ''
        self.Reset()

    def GetTotal(self):
        return self.Total

    def GetBar(self):
        fmt = ':>{:d}d'.format(len(str(self.Total)))
        K = ('{' + fmt + '}').format(self.K)
        status = '{:>5.1f}%{} {}/{:d} [{}<{}, {:>7.1f}it/s]'.format(self.perc * 100, self.bar, K,
                                                                    self.Total,
                                                                    secs_to_time(self.time_cost),
                                                                    secs_to_time(self.time_left),
                                                                    self.freq)
        return status

    def Update(self):
        assert self.K < self.Total
        now = time.time()
        if self.last_time is not None:
            self.time_cost = now - self.start_time
            dur = now - self.last_time
            self.last_time = now
            self.time_left = dur * (self.Total - 1 - self.K)
            self.freq = 1.0 / (dur + 1e-8)
        else:
            self.last_time = now
            self.start_time = now

        self.perc = self.K / (self.Total - 1 + 1e-8)
        nb = int(round(self.perc * self.NUM_BLOCKS))
        self.bar = '|' + '█' * nb + ' ' * (self.NUM_BLOCKS - nb) + '|'

        self.K += 1

    def Reset(self):
        self.K = 0
        self.last_time = None
        self.perc = 0
        self.time_left = None
        self.freq = -1
        self.start_time = None
        self.time_cost = None