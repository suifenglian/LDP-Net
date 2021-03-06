import numpy as np
from scipy.ndimage import sobel
from numpy.linalg import norm
# from sewar.full_ref import sam, scc, ergas
from skimage import filters


def sam(ms, ps):
    assert ms.ndim == 3 and ms.shape == ps.shape

    ms = ms.astype(np.float32)
    ps = ps.astype(np.float32)

    dot_sum = np.sum(ms*ps, axis=2)
    norm_true = norm(ms, axis=2)
    norm_pred = norm(ps, axis=2)

    res = np.arccos(dot_sum/norm_pred/norm_true)

    is_nan = np.nonzero(np.isnan(res))

    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0

    sam = np.mean(res)

    return sam * 180 / np.pi


def sCC(ms, ps):
    # ps_sobel = sobel(ps, mode='constant')
    # ms_sobel = sobel(ms, mode='constant')
    ps_sobel = np.zeros(ps.shape)
    ms_sobel = np.zeros(ps.shape)
    for i in range(ms.shape[2]):
        ps_sobel[:, :, i] = filters.sobel(ps[:, :, i])
        ms_sobel[:, :, i] = filters.sobel(ms[:, :, i])

    scc = np.sum(ps_sobel * ms_sobel)/np.sqrt(np.sum(ps_sobel*ps_sobel))/np.sqrt(np.sum(ms_sobel*ms_sobel))

    return scc


def ergas(ms, ps, ratio=8):
    ms = ms.astype(np.float32)
    ps = ps.astype(np.float32)
    err = ms - ps
    ergas_index = 0
    for i in range(err.shape[2]):
        ergas_index += np.mean(np.square(err[:, :, i]))/np.square(np.mean(ms[:, :, i]))

    ergas_index = (100/ratio) * np.sqrt(1/err.shape[2]) * ergas_index

    return ergas_index



