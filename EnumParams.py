import os

class EnumParams(object):
    def __init__(self, param_pool_dict):
        self.param_pool_dict = param_pool_dict
        self.fname_batch_run = os.path.join('./', 'batch_run.sh')
        self.basic_cmd = 'rlaunch --cpu=1 --gpu=8 --memory=10240 -- python3 main.py ' \
                         '--num_gpus=8 --batch_size=16 --epoches=4000 --eval_nums=2'

        #self.basic_cmd = 'echo hw'

    def dfs(self, option, options, param_pool_dict, k, id):
        if k == len(param_pool_dict):
            option += '--exp_id={}'.format(id)
            options.append(option)
            return
        keys = list(param_pool_dict.keys())
        key = keys[k]
        for t, val in enumerate(param_pool_dict[key]):
            orig_len = len(option)
            if type(val) == str:
                option += '--{}={} '.format(key, val)
            else:
                option += '--{}={:>3.1e} '.format(key, val)
            self.dfs(option, options, param_pool_dict, k + 1, id + str(t))
            option = option[:orig_len]

    def GenBatchFile(self):
        fname = self.fname_batch_run
        option = ''
        options = []
        self.dfs(option, options, self.param_pool_dict, 0, 'ID')
        with open(fname, 'w') as f:
            if False:
                for k, line in enumerate(options):
                    session_name = line.split('--exp_id=')[-1]
                    f.write('tmux new-session -d -s {} \; send-keys "{} {}" Enter\n'.format(session_name,
                                                                                            self.basic_cmd,
                                                                                            line))
            else:
                session_name = 'SESS'
                #f.write('tmux kill-server\n')
                for k, line in enumerate(options):
                    window_name = line.split('--exp_id=')[-1]
                    if k == 0:
                        f.write('tmux new-session -d -s {} -n {} \; send-keys "{} {}" Enter\n'.format(session_name,
                                                                                                window_name,
                                                                                                self.basic_cmd, line))
                    else:
                        f.write('tmux new-window -t {}:{:d} -n {} \; send-keys "{} {}" Enter\n'.format(session_name,
                                                                                                 k, window_name,
                                                                                                 self.basic_cmd, line))

    def Run(self):
        cmd = 'sh {}'.format(self.fname_batch_run)
        print(cmd)
        os.system(cmd)

def main():
    param_pool_dict = {
        'lr':                   [1e-4, ],
        'weight_ce':            [0.5,  ],
        'weight_decay':         [1e-4, ],
        'version':              ['v2', 'v3', 'v4'],
    }
    ep = EnumParams(param_pool_dict)
    ep.GenBatchFile()
    ep.Run()

if __name__ == '__main__':
    main()