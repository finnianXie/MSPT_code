import torch
from torch import nn
import torch.nn.functional as F
from pytorch_radon.utils import PI, SQRT2, deg2rad, affine_grid, grid_sample
import skimage.data as d
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
from fastatomography.util import sector_mask
import os
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.nn.functional import mse_loss

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(False)

device = th.device('cuda:0')
# device = th.device('cpu')

img = np.load('shepp_phantom.npy')
# %%
img = gaussian(img, 1)
# img *= sector_mask(img.shape, np.array(img.shape) / 2, img.shape[0] / 2, (0, 360))
# %%
fig, ax = plt.subplots()
ax.imshow(img[36])
plt.show()
# %%
target = th.as_tensor(img).unsqueeze(0).unsqueeze(0).to(device)


# %%
def euler_matrix(phi, theta=None, psi=None):
    """Rotation matrix in 2 and 3 dimensions.
    Its rows represent the canonical unit vectors as seen from the
    rotated system while the columns are the rotated unit vectors as
    seen from the canonical system.
    Parameters
    ----------
    phi : `array-like`
        Either 2D counter-clockwise rotation angle (in radians) or first
        Euler angle.
    theta, psi : float or `array-like`, optional
        Second and third Euler angles in radians. If both are ``None``, a
        2D rotation matrix is computed. Otherwise a 3D rotation is computed,
        where the default ``None`` is equivalent to ``0.0``.
        The rotation is performed in "ZXZ" rotation order, see the
        Wikipedia article `Euler angles`_.
    Returns
    -------
    mat : `numpy.ndarray`
        Rotation matrix corresponding to the given angles. The
        returned array has shape ``(ndim, ndim)`` if all angles represent
        single parameters, with ``ndim == 2`` for ``phi`` only and
        ``ndim == 3`` for 2 or 3 Euler angles.
        If any of the angle parameters is an array, the shape of the
        returned array is ``broadcast(phi, theta, psi).shape + (ndim, ndim)``.
    References
    ----------
    .. _Euler angles:
        https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    """
    cph = np.cos(phi)
    sph = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cps = np.cos(psi)
    sps = np.sin(psi)

    mat = np.array([
        [cph * cps - sph * cth * sps,
         -cph * sps - sph * cth * cps,
         sph * sth],
        [sph * cps + cph * cth * sps,
         -sph * sps + cph * cth * cps,
         -cph * sth],
        [sth * sps + 0 * cph,
         sth * cps + 0 * cph,
         cth + 0 * (cph + cps)]])  # Make sure all components broadcast

    # if squeeze_out:
    #     return mat.squeeze()
    # else:
    #     # Move the `(ndim, ndim)` axes to the end
    #     extra_dims = len(np.broadcast(phi, theta, psi).shape)
    #     newaxes = list(range(2, 2 + extra_dims)) + [0, 1]
    #     return np.transpose(mat, newaxes)


n_theta = 4
theta = th.linspace(0, 180, n_theta)
theta = th.deg2rad(theta).to(device)
phi = th.linspace(0, 180, n_theta)
phi = th.deg2rad(phi).to(device)
psi = th.linspace(0, 180, n_theta)
psi = th.deg2rad(psi).to(device)

cph = th.cos(phi)
sph = th.sin(phi)
cth = th.cos(theta)
sth = th.sin(theta)
cps = th.cos(psi)
sps = th.sin(psi)
#%%
def euler_matrix(phi, theta=None, psi=None):
    """Rotation matrix in 2 and 3 dimensions.
    Its rows represent the canonical unit vectors as seen from the
    rotated system while the columns are the rotated unit vectors as
    seen from the canonical system.
    Parameters
    ----------
    phi : float or `array-like`
        Either 2D counter-clockwise rotation angle (in radians) or first
        Euler angle.
    theta, psi : float or `array-like`, optional
        Second and third Euler angles in radians. If both are ``None``, a
        2D rotation matrix is computed. Otherwise a 3D rotation is computed,
        where the default ``None`` is equivalent to ``0.0``.
        The rotation is performed in "ZXZ" rotation order, see the
        Wikipedia article `Euler angles`_.
    Returns
    -------
    mat : `numpy.ndarray`
        Rotation matrix corresponding to the given angles. The
        returned array has shape ``(ndim, ndim)`` if all angles represent
        single parameters, with ``ndim == 2`` for ``phi`` only and
        ``ndim == 3`` for 2 or 3 Euler angles.
        If any of the angle parameters is an array, the shape of the
        returned array is ``broadcast(phi, theta, psi).shape + (ndim, ndim)``.
    References
    ----------
    .. _Euler angles:
        https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    """
    if theta is None and psi is None:
        squeeze_out = (np.shape(phi) == ())
        ndim = 2
        phi = np.array(phi, dtype=float, copy=False, ndmin=1)
        theta = psi = 0.0
    else:
        # `None` broadcasts like a scalar
        squeeze_out = (np.broadcast(phi, theta, psi).shape == ())
        ndim = 3
        phi = np.array(phi, dtype=float, copy=False, ndmin=1)
        if theta is None:
            theta = 0.0
        if psi is None:
            psi = 0.0
        theta = np.array(theta, dtype=float, copy=False, ndmin=1)
        psi = np.array(psi, dtype=float, copy=False, ndmin=1)
        ndim = 3

    cph = np.cos(phi)
    sph = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cps = np.cos(psi)
    sps = np.sin(psi)

    if ndim == 2:
        mat = np.array([[cph, -sph],
                        [sph, cph]])
    else:
        mat = np.array([
            [cph * cps - sph * cth * sps,
             -cph * sps - sph * cth * cps,
             sph * sth],
            [sph * cps + cph * cth * sps,
             -sph * sps + cph * cth * cps,
             -cph * sth],
            [sth * sps + 0 * cph,
             sth * cps + 0 * cph,
             cth + 0 * (cph + cps)]])  # Make sure all components broadcast

    if squeeze_out:
        return mat.squeeze()
    else:
        # Move the `(ndim, ndim)` axes to the end
        extra_dims = len(np.broadcast(phi, theta, psi).shape)
        newaxes = list(range(2, 2 + extra_dims)) + [0, 1]
        return np.transpose(mat, newaxes)
# %%
print('phi', phi)
print('theta', theta)
print('psi', psi)
print()
line1 = th.stack([cph * cps - sph * cth * sps, -cph * sps - sph * cth * cps, sph * sth], 1)
line2 = th.stack([sph * cps + cph * cth * sps, -sph * sps + cph * cth * cps, -cph * sth], 1)
line3 = th.stack([sth * sps + 0 * cph, sth * cps + 0 * cph, cth + 0 * (cph + cps)], 1)
R = th.stack([line1, line2, line3], 1)
R.shape

R1 = euler_matrix(phi[1].cpu().numpy()[()], theta[1].cpu().numpy()[()], psi[1].cpu().numpy()[()])

print(R[1])
print(R1)
print(np.allclose(R[1].cpu().numpy(), R1))