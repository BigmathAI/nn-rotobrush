import os, cv2, numpy as np
import pytoolkit.files as fp
import tqdm

def SegEval(rst, gt):
    I = TP = np.sum(np.logical_and(rst, gt).astype(np.float32))
    FP = np.sum(np.logical_and(rst, np.logical_not(gt)).astype(np.float32))
    TN = np.sum(np.logical_and(np.logical_not(rst), np.logical_not(gt)).astype(np.float32))
    FN = np.sum(np.logical_and(np.logical_not(rst), gt).astype(np.float32))
    U = np.sum(np.logical_or(rst, gt).astype(np.float32))

    P = TP / (TP + FP + 1e-8)
    R = TP / (TP + FN + 1e-8)
    IoU = I / (U + 1e-8)
    FPR = FP / (FP + TN + 1e-8)
    F1 = 2.0 * P * R / (P + R + 1e-8)
    return P, R, IoU, FPR, F1

def ComputeEvalForMethod(fd_rst, fd_gt, NumOfThresholds):
    fnames_rst = fp.dir(fd_rst, ['.jpg', '.png', '.tiff'])
    fnames_gt = fp.dir(fd_gt, ['.jpg', '.png', '.tiff'])

    assert len(fnames_rst) == len(fnames_gt), 'len(fnames_rst) == len(fnames_gt)'

    NumOfSamples = len(fnames_rst)

    EvalMatrix = np.zeros([NumOfSamples, 5, NumOfThresholds], np.float32)

    thresholds = np.linspace(0, 1, NumOfThresholds)

    for k in tqdm.tqdm(range(NumOfSamples)):
        fname_rst = fnames_rst[k]
        fname_gt = fnames_gt[k]

        rst = cv2.imread(fname_rst, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(fname_gt, cv2.IMREAD_GRAYSCALE)

        rst = rst.astype(np.float32) / 255.0
        gt = gt.astype(np.float32) / 255.0

        for j, thres in enumerate(thresholds):
            Precion, Recall, IoU, FPR, F1 = SegEval(rst >= thres, gt >= 0.01)
            EvalMatrix[k,:,j] = Precion, Recall, IoU, FPR, F1

    Mu = np.mean(EvalMatrix, axis=0)
    Var = np.mean(EvalMatrix, axis=0)

    np.save(os.path.join(fd_rst, 'EvalMatrix.npy'), EvalMatrix)
    np.save(os.path.join(fd_rst, 'Mu.mu'), Mu)
    np.save(os.path.join(fd_rst, 'Var.var'), Var)