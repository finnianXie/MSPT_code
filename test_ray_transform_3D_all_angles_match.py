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
img[:] = 0
img[28:37, 23:42, 30 - 5:35 - 5] = 1
img = gaussian(img, 1)
# img *= sector_mask(img.shape, np.array(img.shape) / 2, img.shape[0] / 2, (0, 360))
fig, ax = plt.subplots(1, 3)
ax[0].imshow(img.sum(0))
ax[1].imshow(img.sum(1))
ax[2].imshow(img.sum(2))
plt.show()
target = th.as_tensor(img).unsqueeze(0).unsqueeze(0).to(device)

#%%

def affine_matrix_3D(phi, theta, psi, translation):
    """Rotation matrix in 2 and 3 dimensions.


    Its rows represent the canonical unit vectors as seen from the
    rotated system while the columns are the rotated unit vectors as
    seen from the canonical system.
    Parameters
    ----------
    phi : `array-like`
        Either 2D counter-clockwise rotation angle (in radians) or first
        Euler angle.
    theta, psi : `array-like`
        Second and third Euler angles in radians. If both are ``None``, a
        2D rotation matrix is computed. Otherwise a 3D rotation is computed,
        where the default ``None`` is equivalent to ``0.0``.
        The rotation is performed in "ZXZ" rotation order, see the
        Wikipedia article `Euler angles`_.
    translation : `array-like`, shape (2, n_angles) 2D translation vector of the projection
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
    cph = th.cos(phi)
    sph = th.sin(phi)
    cth = th.cos(theta)
    sth = th.sin(theta)
    cps = th.cos(psi)
    sps = th.sin(psi)
    line1 = th.stack([cph * cps - sph * cth * sps, -cph * sps - sph * cth * cps, sph * sth, translation[0]], 1)
    line2 = th.stack([sph * cps + cph * cth * sps, -sph * sps + cph * cth * cps, -cph * sth, translation[1]], 1)
    line3 = th.stack([sth * sps + 0 * cph, sth * cps + 0 * cph, cth + 0 * (cph + cps), th.zeros_like(translation[1])],
                     1)
    R = th.stack([line1, line2, line3], 1)
    return R


def ray_transform(vol, phi_rad, theta_rad, psi_rad, translation):
    R = affine_matrix_3D(phi_rad, theta_rad, psi_rad, translation)
    out_size_max = vol.shape[2]  # int(np.ceil(np.sqrt((img.shape[2] / 2) ** 2 + (img.shape[3] / 2) ** 2)))
    out_size = (n_theta, 1, out_size_max, out_size_max, out_size_max)
    grid = F.affine_grid(R, out_size)
    out = F.grid_sample(vol.expand(n_theta, 1, vol.shape[2], vol.shape[3], vol.shape[4]), grid)
    print(out.shape)
    # out is (N_batch, channels, Z, Y, X)
    sino = th.sum(out, 2)
    return sino


n_theta = 25
phi_deg = th.linspace(0, 180, n_theta)
theta_deg = th.linspace(0, 0, n_theta)
psi_deg = th.linspace(0, 0, n_theta)

phi_rad_target = th.deg2rad(phi_deg).to(device)
theta_rad_target = th.deg2rad(theta_deg).to(device)
psi_rad_target = th.deg2rad(psi_deg).to(device)

translation = th.zeros((2, n_theta), device=device)
# translation = th.as_tensor(np.random.uniform(-100, 100, theta_target.shape[0]) / 100)
# R_target = affine_matrix_3D(phi_rad_target, theta_rad_target, psi_rad_target, translation)
# R_target.shape
phi_rad_model = phi_rad_target + th.as_tensor(np.random.uniform(-100, 100, phi_rad_target.shape[0]) / 100,
                                              device=device)
theta_rad_model = theta_rad_target.clone()
psi_rad_model = psi_rad_target.clone()

sino_target = ray_transform(target, phi_rad_target, theta_rad_target, psi_rad_target, translation)
from fastatomography.util import plotmosaic

plotmosaic(sino_target.squeeze().cpu().numpy(), 'pytorch projections sum axis 3')
from numpy.random import uniform
from fastatomography.util import *
from fastatomography.tomo import ray_transforms


def ray_transform_astra(real_space_extent, projection_shape, angles_in, interp='linear'):
    """
    Generate the ASTRA-based ray-projection and ray-back-projection operators
    :param real_space_extent: array (3,) the number of pixels in the three volume reconstruction dimensions
    :param projection_shape: array or tuple (2,) shape of the projections
    :param num_projections: int, number of projections to calculate
    :param interp: string, 'nearest' or 'linear'
    :return: A and At the Raytransform ind its agjoint
    call it like

    projections = A(volume, out=projections, angles=angles)

    where volume is a torch cuda tensor of shape (projection_shape[0], projection_shape[1], projection_shape[1])
          projections is a torch cuda tensor of shape projection_shape
          angles is a numpy array of shape (3, num_projections)
    """
    assert len(real_space_extent) == 3, "len(real_space_extent) != 3"
    assert len(projection_shape) == 2, "len(projection_shape) != 2"
    import numpy as np
    from fastatomography.tomo import RayTransform, RayBackProjection, Parallel3dEulerGeometry
    import odl

    num_projections = angles_in.shape[0]

    reco_space = odl.uniform_discr(
        min_pt=[-real_space_extent[0] / 2, -real_space_extent[1] / 2, -real_space_extent[2] / 2],
        max_pt=[real_space_extent[0] / 2, real_space_extent[1] / 2, real_space_extent[2] / 2],
        shape=[projection_shape[0], projection_shape[1], projection_shape[1]],
        dtype='float32', interp=interp)

    phi = np.linspace(0, np.deg2rad(90), int(np.ceil(num_projections ** (1 / 3))))
    theta = np.linspace(0, np.deg2rad(0.5), int(np.ceil(num_projections ** (1 / 3))))
    psi = np.linspace(0, np.deg2rad(0.5), int(np.ceil(num_projections ** (1 / 3))))
    angle_partition = odl.nonuniform_partition(phi, theta, psi)
    print('angle_partition', angle_partition)
    print()
    detector_partition = odl.uniform_partition([-real_space_extent[0] / 2, -real_space_extent[1] / 2],
                                               [real_space_extent[0] / 2, real_space_extent[1] / 2],
                                               [projection_shape[0], projection_shape[1]])
    print('detector_partition', detector_partition)
    print()
    geometry = Parallel3dEulerGeometry(angle_partition, detector_partition, check_bounds=False)
    print('geometry', geometry)
    print()
    angle_partition_dummy = odl.uniform_partition(
        min_pt=[angles_in.min(), -real_space_extent[0] / 2, -real_space_extent[1] / 2],
        max_pt=[angles_in.max(), real_space_extent[0] / 2, real_space_extent[1] / 2],
        shape=[num_projections, projection_shape[0], projection_shape[1]])
    print('angle_partition_dummy', angle_partition_dummy)
    print()
    domain = odl.uniform_discr_frompartition(angle_partition_dummy, dtype=np.float32)
    ray_trafo = RayTransform(reco_space, geometry, impl='astra_cuda')

    # proj_fspace = FunctionSpace(geometry.params, out_dtype=np.float32)
    # proj_space = DiscreteLp(
    #     proj_fspace, geometry.partition, proj_tspace,
    #     interp=proj_interp, axis_labels=axis_labels)
    #
    print('reco_space', reco_space)
    print('domain', domain)
    print()
    rayback_trafo = RayBackProjection(reco_space, geometry, impl='astra_cuda', domain=domain)
    return ray_trafo, rayback_trafo


ps = sino_target.squeeze().shape
projection_shape = ps[1:]
vol_shape = (projection_shape[0], projection_shape[1], projection_shape[1])
real_space_extent = np.array([projection_shape[0], projection_shape[1], projection_shape[1]])

angles_in = th.stack([psi_rad_target, psi_rad_target, phi_rad_target], 0).cpu().numpy()
# angles_in = th.stack([psi_rad_target, psi_rad_target, phi_rad_target], 0).cpu().numpy()
projection_shape, vol_shape, real_space_extent
A, At = ray_transform_astra(real_space_extent, projection_shape, angles_in, interp='linear')

target_astra = th.as_tensor(np.transpose(target.squeeze().cpu().numpy(),(2,1,0))).contiguous().float().cuda()
target_astra = target.squeeze()

y_model = th.as_tensor(np.transpose(sino_target.squeeze().cpu().numpy(), (1, 0, 2))).contiguous().float().cuda()
y_model.fill_(0)
proj_astra = A(target_astra, out=y_model, angles=angles_in)

proj_astra2 = np.transpose(proj_astra.cpu().numpy(), (1, 0, 2))
plotmosaic(proj_astra2, title='ASTRA projections [psi, neg. phi, neg theta]')
