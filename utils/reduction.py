import numpy as np
from numpy import linalg as LA


def standardize(data):
    """
    Standardize the set data.
    """
    n, m  = data.shape
    mean_col = np.mean(data, 0)
    sigma_col = np.sqrt(np.var(data, 0))
    return (data - mean_col)/(sigma_col + 1e-11)


def PCA(data, d=2):
    """
    Perform the PCA on the set data.

    data : the data to analyse
    d : dimension of the PCA
    """
    cov = np.cov(data.T)
    p  = cov.shape[1]
    res = LA.eig(cov)
    sorting = [[res[0][i], i] for i in range(p)]
    sorting = sorted(sorting, key=lambda eigen: np.abs(eigen[0]))
    new_base = np.zeros((p,p))
    for i in range(p):
        new_base[:,i] = res[1][sorting[p - 1 - i][1]]
    proj = np.zeros((p, d))
    for i in range(d):
        proj[i,i] = 1

    # projection on the first d vectors (linked with the d highest eigenvalues)
    # the result is in the eigenvectors base
    data_proj = np.dot(data, proj)
    return data_proj
