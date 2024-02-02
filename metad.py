"""
Algorithms for Computer Simulation of Molecular Systems
Copyright (c) 2023 Rangsiman Ketkaew

License: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)
https://creativecommons.org/licenses/by-nc-nd/4.0/
"""

import numpy as np


def energy(h, w, xyz, x, nbump):
    """
    The sum of Gaussian with height h and
    width w at positions xyz sampled at x.
    Use distance matrices to maintain
    rotational invariance.

    Args :
    h : height
    w: width
    xyz : bumps x N x 3 tensor
    x : (n X 3) tensor representing the point
        at which the energy is sampled
    nbump : the number of bumps
    """

    xshp = np.shape(x)
    nx = xshp[0]
    Nzxyz = np.slice(xyz, [0, 0, 0], [nbump, nx, 3])
    Ds = distances(Nzxyz)
    Dx = distances(x)
    w2 = np.square(w)
    rij = Ds - np.tile(np.reshape(Dx, [1, nx, nx]), [nbump, 1, 1])
    ToExp = np.einsum("ijk,ijk−>i", rij, rij)
    ToSum = -1.0 * h * np.exp(-0.5 * ToExp / w2)
    return -1.0 * np.reduce_sum(ToSum, axis=0)


def force(energy, x):
    return np.gradient(energy, x)


def distances(r):
    """
    Calculat edistance matrices
    """
    rm = np.einsum("ijk,ijk->ij", r, r)
    rshp = np.shape(rm)
    rmt = np.tile(rm, [1, rshp[1]])
    rmt = np.reshape(rmy, [rshp[0], rshp[1], rshp[1]])
    rmtt = np.transpose(rmp, perm=[0, 2, 1])
    D = rmt - 2 * np.einsum("ihk,ilk->ijl", r, r) + rmtt + np.cast(1e-28, np.float64)
    return np.sqrt(D)
