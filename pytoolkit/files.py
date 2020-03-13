from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, time, shutil

def dir(path, ext=None, use_cache=False, case_sensitive=False):
    def _dir(path, ext):
        files = [os.path.join(path, x) for x in os.listdir(path)]
        if type(ext) == list:
            files = [x for x in files if os.path.isfile(x) and (ext == None or os.path.splitext(x)[-1] in ext)]
        else:
            files = [x for x in files if os.path.isfile(x) and (ext == None or os.path.splitext(x)[-1] == ext)]
        if not case_sensitive:
            files = [x.lower() for x in files]
        return sorted(files)
    if ext is '':
        ext = None
    cache = os.path.join(path, 'cache.list')
    if use_cache and os.path.exists(cache):
        with open(cache, 'r') as f:
            lines = [s.rstrip() for s in f.readlines()]
            lines = [os.path.join(path, line) for line in lines]
            if not case_sensitive:
                lines = [line.lower() for line in lines]
            return lines
    sorted_files = _dir(path, ext)
    if use_cache and not os.path.exists(cache):
        with open(cache, 'w') as f:
            lines = [fileparts(s)[1] + fileparts(s)[2] for s in sorted_files]
            f.writelines([s + '\n' for s in lines])
    return sorted_files

def filelist_without_path(filelist):
    temp = [fileparts(fname) for fname in filelist]
    paths, names, exts = zip(*temp)
    return [name + ext for name, ext in zip(names, exts)]

def subdirs(path):
    fds = [os.path.join(path, x) for x in os.listdir(path)]
    fds = [x for x in fds if os.path.isdir(x)]
    return fds

def fileparts(filename):
    path, filename = os.path.split(filename)
    filename, ext = os.path.splitext(filename)
    return path, filename, ext

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def stem(filename):
    _, name, ext = fileparts(filename)
    return name + ext

def build_filename(path, fname):
    return os.path.join(path, stem(fname))

def write_filelist_to_file(filelist, fname_out):
    with open(fname_out, 'w') as f:
        filelist = [x + '\n' for x in filelist]
        f.writelines(filelist)

def gen_filelist(path):
    filelist = dir(path, case_sensitive=True)
    filelist = filelist_without_path(filelist)
    fname_out = os.path.join(path, 'filelist.txt')
    write_filelist_to_file(filelist, fname_out)

def copytree(src, dst, without_files=False):
    def ig_file(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]
    shutil.copytree(src, dst, ignore=ig_file if without_files else None)

def traverse(fd, ext, foo, subnames=None):
    sub_dirs = subdirs(fd)
    if subnames is not None:
        subnames += sub_dirs
    fnames = dir(fd, ext)
    for fname in fnames:
        if callable(foo):
            foo(fname)
        else:
            foo.append(fname)
    if len(sub_dirs) == 0:
        return
    for subdir in sub_dirs:
        traverse(subdir, ext, foo, subnames)
