import numpy as np
from typing import Tuple


def sum_non_neg_diag(X: np.ndarray) -> int:
    dg = np.diag(X)
    p_dg = dg[dg >= 0]
    
    if len(p_dg) == 0:
        return -1
    
    return np.sum(p_dg)


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    return np.array_equal(np.sort(np.array(x)), np.sort(np.array(y)))


def max_prod_mod_3(x: np.ndarray) -> int:
    x_copy = np.array(x)
    x_mod = x_copy % 3
    ind = np.logical_or(x_mod[:-1] == 0, x_mod[1:] == 0)
    prd = x_copy[:-1] * x_copy[1:]
    
    if len(prd[ind]) == 0:
        return -1
    
    return np.max(prd[ind])


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(np.array(image) * np.array(weights), axis=2)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    val, cnt = np.array(x).T
    x_copy = np.repeat(val, cnt)
    
    val, cnt = np.array(y).T
    y_copy = np.repeat(val, cnt)
    
    if len(x_copy) != len(y_copy):
        return -1
    
    return np.dot(x_copy, y_copy)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    norm_x = np.linalg.norm(X, axis=1, keepdims=True)
    norm_y = np.linalg.norm(Y, axis=1, keepdims=True)
    dot_product = np.dot(X, Y.T)

    return np.where(norm_x * norm_y.T != 0, dot_product / (norm_x * norm_y.T), 1)