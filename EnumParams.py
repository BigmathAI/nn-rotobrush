import os

class EnumParams(object):
    def __init__(self, param_pool_dict):
        self.param_pool_dict = param_pool_dict
        self.fname_batch_run = os.path.join('./', 'batch_run.sh')
        self.basic_cmd = 'rlaunch --cpu=1 --gpu=1 --memory=10240 -- python3 main.py ' \
                         '--num_gpus=1 --batch_size=16 --epoches=4000 --eval_nums=-1'

    def dfs(self, option, options, param_pool_dict, k):
        if k == len(param_pool_dict):
            options.append(option)
            return
        keys = list(param_pool_dict.keys())
        key = keys[k]
        for val in param_pool_dict[key]:
            orig_len = len(option)
            option += '--{}={:>3.1e} '.format(key, val)
            self.dfs(option, options, param_pool_dict, k + 1)
            option = option[:orig_len]

    def GenBatchFile(self):
        fname = self.fname_batch_run
        option = ''
        options = []
        self.dfs(option, options, self.param_pool_dict, 0)
        with open(fname, 'w') as f:
            for k, line in enumerate(options):
                session_name = 'ID{:03d}'.format(k)
                f.write('tmux new-session -d -s {} \; send-keys "{} --exp_id={} {}" Enter\n'.format(session_name,
                                                                                                    self.basic_cmd,
                                                                                                    session_name, line))

    def Run(self):
        cmd = 'sh {}'.format(self.fname_batch_run)
        print(cmd)
        os.system(cmd)

def main():
    param_pool_dict = {
        'lr':                   [1e-5, 1e-4, 1e-3],
        'weight_ce':            [0.4, 0.5, 0.6, 0.7],
        'weight_decay':         [1e-3, 1e-2, 1e-4],
    }
    ep = EnumParams(param_pool_dict)
    ep.GenBatchFile()
    ep.Run()

if __name__ == '__main__':
    main()