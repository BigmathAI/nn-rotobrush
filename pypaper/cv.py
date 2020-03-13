import numpy as np, cv2
import imageio

def load_vd(fname):
    vc = cv2.VideoCapture(fname)
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nf = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    vd = np.zeros([nf, h, w, 3], np.uint8)
    for k in range(nf):
        rst, im = vc.read()
        vd[k] = im
    vc.release()
    return vd

def write_vd(fname, vd, frame_rate=16):
    if len(vd.shape) == 3:
        vd = vd[:,:,:,np.newaxis]
    if vd.shape[-1] == 1:
        vd = np.concatenate([vd] * 3, axis=3)
    nf, h, w, c = vd.shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vw = cv2.VideoWriter(fname, fourcc, frame_rate, (w, h))
    [vw.write(fr) for fr in vd]
    vw.release()

def gen_gif(fnames, fname_out, dur=0.5):
    ims = [cv2.imread(f, cv2.IMREAD_COLOR)[:,:,::-1] for f in fnames]
    imageio.mimsave(fname_out, ims, 'GIF', duration=dur)