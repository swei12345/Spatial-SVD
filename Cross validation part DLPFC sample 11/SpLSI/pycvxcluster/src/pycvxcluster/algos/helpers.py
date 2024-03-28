import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.sparse import csr_array
from scipy.sparse import lil_array
from scipy.sparse import find
from scipy.sparse import issparse
from scipy.sparse import diags
import numpy as np


def fnorm(X):
    if issparse(X):
        return sla.norm(X, "fro")
    return la.norm(X, "fro")


def proj_l2(input, weight_vec):
    d, n = input.shape
    if n != len(weight_vec):
        raise ValueError("x and weight_vec must have the same length")
    output = lil_array(input.copy())
    norm_input = np.sqrt(np.sum(input * input, axis=0))
    idx = norm_input > weight_vec
    if idx.any():
        output[:, idx] = output[:, idx] * weight_vec[idx] / norm_input[idx]
    output = output.tocsr()
    return output


def prox_l2(y, weight_vec):
    d, n = y.shape
    if n != len(weight_vec):
        raise ValueError("y and weight_vec must have the same length")
    norm_y_col = np.sqrt(np.sum(y * y, axis=0))
    alpha_vec = weight_vec / (norm_y_col + 1e-15)
    idx = alpha_vec < 1
    # idx = find(rr)
    x = lil_array((d, n))
    if idx.any():
        x[:, idx] = y[:, idx] * (1 - alpha_vec[idx])
    x = x.tocsr()
    return x, idx, norm_y_col
