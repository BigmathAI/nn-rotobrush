import cv2
import numpy as np
import math
from numpy.linalg import norm

def _crop_around_center(image, width, height):
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))
    if(width > image_size[0]):
        width = image_size[0]
    if(height > image_size[1]):
        height = image_size[1]
    x1 = int(image_center[0] -  width * 0.5)
    x2 = int(image_center[0] +  width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)
    return image[y1:y2, x1:x2]

def _largest_rotated_rect(w, h, angle):
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi
    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)
    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
    delta = math.pi - alpha - gamma
    length = h if (w < h) else w
    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)
    y = a * math.cos(gamma)
    x = y * math.tan(gamma)
    return (bb_w - 2 * x, bb_h - 2 * y)

def rotate(im, angle, crop=False):
    h, w = im.shape[:2]
    center = tuple(np.array([w, h]) / 2)
    rot_mat = np.vstack([cv2.getRotationMatrix2D(center, angle, 1.0), [0, 0, 1]])
    rot_mat_no_trans = np.matrix(rot_mat[0:2, 0:2])
    h2, w2 = h * 0.5, w * 0.5
    rotated_coords = [
        (np.array([-w2,  h2]) * rot_mat_no_trans).A[0],
        (np.array([ w2,  h2]) * rot_mat_no_trans).A[0],
        (np.array([-w2, -h2]) * rot_mat_no_trans).A[0],
        (np.array([ w2, -h2]) * rot_mat_no_trans).A[0]
    ]
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]
    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]
    right, left, top, bottom = max(x_pos), min(x_neg), max(y_pos), min(y_neg)
    new_w = int(abs(right - left))
    new_h = int(abs(bottom - top))
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - w2)],
        [0, 1, int(new_h * 0.5 - h2)],
        [0, 0, 1]
    ])
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    result = cv2.warpAffine(im, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    if crop is True:
        result = _crop_around_center(result, *_largest_rotated_rect(w, h, math.radians(angle)))
    return result

def overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    X1 = max(x1, x2)
    X2 = min(x1 + w1, x2 + w2)
    Y1 = max(y1, y2)
    Y2 = min(y1 + h1, y2 + h2)
    return [X1, Y1, max(0, X2 - X1), max(0, Y2 - Y1)]

def union(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    X1 = min(x1, x2)
    X2 = max(x1 + w1, x2 + w2)
    Y1 = min(y1, y2)
    Y2 = max(y1 + h1, y2 + h2)
    return [X1, Y1, X2 - X1, Y2 - Y1]

def get_roi(im, box, dup=False):
    h, w = im.shape[:2]
    if dup is True:
        ox, oy, ow, oh = union(box, [0, 0, w, h])
        left = max(0, 0 - ox)
        right = max(0, ox + ow - w)
        top = max(0, 0 - oy)
        bottom = max(0, oy + oh - h)
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REPLICATE)
        box = (box[0] - ox, box[1] - oy, box[2], box[3])
    x, y, roi_w, roi_h = box
    h, w = im.shape[:2]
    ox, oy, ow, oh = overlap(box, [0, 0, w, h])
    roi = np.zeros([roi_h, roi_w, im.shape[2]], im.dtype)
    roi[oy - y: oy - y + oh, ox - x: ox - x + ow] = im[oy: oy + oh, ox: ox + ow]
    return roi

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 16 # Number of bins
    bin = np.int32(bin_n*ang/(2*np.pi))

    bin_cells = []
    mag_cells = []

    cellx = celly = 8

    for i in range(0,img.shape[0]//celly):
        for j in range(0,img.shape[1]//cellx):
            bin_cells.append(bin[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])
            mag_cells.append(mag[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps

    return hist

