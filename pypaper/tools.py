import pytoolkit.files as fp
import os, time, numpy as np

'''
fd = './src/result/'
fdout = './dst/result/
ext = '.jpg'
indices = [1,2,3,6,39,42]
suffix = ''
copy_specified_files(fd, fdout, ext, indices, suffix)
'''
def copy_specified_files(fd, fd_out, ext, indices, suffix):
    fp.mkdir(fd_out)
    sub_fds = fp.subdirs(fd)
    for sub_fd in sub_fds:
        fnames = fp.dir(sub_fd, ext)
        fnames = [fnames[idx] for idx in indices]
        path, name, ext_ = fp.fileparts(sub_fd)
        sub_fd_out = os.path.join(fd_out, name + ext_ + suffix)
        fp.mkdir(sub_fd_out)
        cmd = 'cp '
        for fname in fnames:
            cmd += fname + ' '
        cmd += sub_fd_out
        print(cmd)
        os.system(cmd)