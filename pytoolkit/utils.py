import threading, math, logging, coloredlogs

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
    log_format = '%(asctime)s %(filename)s[line:%(lineno)d] %(message)s'
    log_datefmt = '%m/%d/%Y %H:%M:%S'
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format=log_format, datefmt=log_datefmt)
    logger = logging.getLogger(__name__)
    coloredlogs.install(level=0, logger=logger, fmt=log_format)
    return logger