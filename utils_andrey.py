import torch
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn.functional as F
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.ndimage import zoom
from skimage.filters import gaussian
# from fastatomography.util import sector_mask
import os
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, CyclicLR
from torch.nn.functional import mse_loss
from torch.utils.data import BatchSampler, SequentialSampler
from utils import ray_transform, optimize_sino, affine_matrix_3D
from fastatomography.util import plotmosaic
from kornia.filters import filter3d
from ccpi.filters.regularisers import ROF_TV, FGP_TV, PD_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, Diff4th

from tqdm import tqdm
from scipy import io
from copy import deepcopy as copy
from typing import Type, Any
import glob


from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation
from numpy.fft import fft2, ifft2
from pathlib import Path

from gaussian_normalization import axis_gaussian_normalization

import gc




def skip_steps(scheduler, steps_to_skip):
    for _ in range(steps_to_skip):  # to start with max LR
        scheduler.step()
    return scheduler

def plot_losses_model(inner_losses, model, outer_losses, suptitle):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(np.arange(len(inner_losses)), np.log10(inner_losses))
    slice = int(model.shape[0]/2)
    ax[1].imshow(model[slice])
    ax[2].scatter(np.arange(len(outer_losses)), np.log10(outer_losses))
    fig.suptitle(suptitle)
    ax[0].set_title(f'Loss final: {inner_losses[-1]}')
    ax[2].set_title(f'Loss final: {outer_losses[-1]}')
    plt.show()


def plot_angles(angles_model_tuple, angles_init_tuple=None, to_deg=True, title=None):
    if to_deg:
        angles_model_tuple = tuple(th.rad2deg(tensor) for tensor in angles_model_tuple)
        if angles_init_tuple:
            angles_init_tuple = tuple(th.rad2deg(tensor) for tensor in angles_init_tuple)

    (phi_rad_model, theta_rad_model, psi_rad_model) = angles_model_tuple
    if angles_init_tuple:
        (phi_rad_init, theta_rad_init, psi_rad_init) = angles_init_tuple

    fig_number = 4 if angles_init_tuple else 3
    number_of_angles = len(phi_rad_model)

    fig, ax = plt.subplots(1, fig_number, figsize=(15, 5))
    ax[0].scatter(np.arange(number_of_angles), phi_rad_model.detach().cpu().numpy().squeeze())
    ax[1].scatter(np.arange(number_of_angles), theta_rad_model.detach().cpu().numpy().squeeze())
    ax[2].scatter(np.arange(number_of_angles), psi_rad_model.detach().cpu().numpy().squeeze())

    ax[0].set_title('phi')
    ax[1].set_title('theta')
    ax[2].set_title('psi')

    if angles_init_tuple:
        ax[0].scatter(np.arange(number_of_angles), phi_rad_init.detach().cpu().numpy().squeeze())
        ax[1].scatter(np.arange(number_of_angles), theta_rad_init.detach().cpu().numpy().squeeze())
        ax[2].scatter(np.arange(number_of_angles), psi_rad_init.detach().cpu().numpy().squeeze())
        ax[3].scatter(np.arange(number_of_angles), (phi_rad_model - phi_rad_init).detach().cpu().numpy().squeeze())
        ax[3].set_title('phi difference')

    fig.suptitle(title)
    plt.show()


def print_memory_usage(device, log_file=None, info=""):
    # torch.cuda.synchronize(device)
    allocated_memory = torch.cuda.memory_allocated(device) / 1024 ** 2
    memory_reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
    if log_file:
        log_file.write('\n' + info + '\n')
        log_file.write(f"Allocated/Reserved/Total GPU Memory: {allocated_memory:.2f} MB/{memory_reserved:.2f} MB/{total_memory:.2f} MB\n")
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.device == device:
                log_file.write(f"{type(obj).__name__}: {obj.size()} - {obj.element_size() * obj.numel() / 1024 / 1024:.2f} MB\n")
    else:
        print(info)
        print(f"Allocated/Reserved/Total GPU Memory: {allocated_memory:.2f} MB/{memory_reserved:.2f} MB/{total_memory:.2f} MB")
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.device == device:
                print(f"{type(obj).__name__}: {obj.size()} - {obj.element_size() * obj.numel() / 1024 / 1024:.2f} MB")


def create_3d_gaussian_kernel(kernel_size, sigma):
    """
    Create a 3D Gaussian kernel in PyTorch. FWHM = 2.355 sigma

    Args:
        kernel_size (int): The size of the kernel (should be odd).
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: A 3D Gaussian kernel.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be odd.")

    # Create a grid of coordinates
    coords = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1)
    x, y, z = torch.meshgrid(coords, coords, coords)

    # Calculate the Gaussian kernel
    kernel = torch.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()  # Normalize the kernel to sum to 1

    return kernel


class DataStack:
    def __init__(self, path_stack=None, path_angles=None, device=None):
        self.images = None
        self.angles = None
        self.device = device

        if path_stack:
            stack_mat = io.loadmat(path_stack)
            self.images = np.array(stack_mat['data']).astype(np.float32)
            print(f'stack\'s shape: {self.images.shape}')
        if path_angles:
            self.angles = np.loadtxt(path_angles, delimiter=',', usecols=0).astype(np.float32)
            print(f'angles\'s shape: {self.angles.shape}')

    @staticmethod
    def normalization(st) -> Any:
        st -= st.min()
        st /= st.max()
        return st

    def preprocessing(self):
        # Axis rearrangement (and little cropping?)
        self.images = np.transpose(self.images, (2, 0, 1))[:, :, 35:-35]
        # Discard missing angle images
        self.images = np.delete(self.images, [18, 37], axis=0)
        print(f'stack\'s shape: {self.images.shape}')

        # Cropping images
        cr1 = 75
        cr2 = 115
        self.images = self.images[:, cr2:-cr2, cr1:-cr1]
        self.images = self.normalization(self.images)
        print(f'stack\'s shape: {self.images.shape}')

    def gauss_correction(self, show_res=False):
        # Gaussian correction perpendicular to rotation
        self.images = axis_gaussian_normalization(self.images, show_res=show_res)
        self.images = self.normalization(self.images)
        print('stack gaussian normalization done')

    def rescale_images(self, scale=0.10) -> Type['DataStack']:
        stack_to_rescale = self.images
        # print(f'stack shape before: {stack_to_rescale.shape}')

        stack_rescale = []
        for s in stack_to_rescale:
            stack_rescale.append(rescale(s, scale, anti_aliasing=True))

        new_stack = copy(self)
        new_stack.images = np.array(stack_rescale)
        # print(f'stack shape after: {new_stack.images.shape}')

        return new_stack

    def clip(self, min=0, max=1) -> Type['DataStack']:
        new_stack = copy(self)
        new_stack.images = self.normalization(new_stack.images)
        new_stack.images = np.clip(new_stack.images, min, max)
        new_stack.images = self.normalization(new_stack.images)

        return new_stack

    def padding(self, frac=0.1, mode='constant'):
        new_stack = copy(self)

        median = np.median(new_stack.images)

        (_, x, y) = new_stack.images.shape
        x_to_pad = int(x*frac)
        y_to_pad = int(y*frac)
        pad_width = ((0, 0), (x_to_pad, x_to_pad), (y_to_pad, y_to_pad))
        new_stack.images = np.pad(new_stack.images, pad_width, mode=mode, constant_values=median)

        return new_stack

    def discard_images(self, array_to_discard) -> Type['DataStack']:
        new_stack = copy(self)
        new_stack.images = np.delete(new_stack.images, array_to_discard, axis=0)
        new_stack.angles = np.delete(new_stack.angles, array_to_discard, axis=0)

        return new_stack

    def to_sino_data(self) -> th.Tensor:
        return th.as_tensor(self.images, device=self.device, dtype=th.float32).unsqueeze_(1)

class Model:
    def __init__(self, device, phi=None, theta=None, psi=None, translation=None, sino_shape=None):
        self.phi = th.as_tensor(phi, dtype=th.float32) if phi is not None else phi
        self.len = len(phi) if phi is not None else phi
        self.theta = self.fill_param(theta)
        self.psi = self.fill_param(psi)

        self.translation = th.zeros((2, self.len)) if translation is None and phi is not None else translation

        self.volume = None

        self.device = device
        self.on_gpu = True
        self.to_gpu()

        self.reinit_volume(sino_shape=sino_shape)

    def print_shapes(self):
        print(f'model  shape: f{self.volume.shape}')
        print(f'phi shape: f{self.phi.shape}')
        print(f'theta shape: f{self.theta.shape}')
        print(f'psi shape: f{self.psi.shape}')
        print(f'translation shape: f{self.translation.shape}')

    def to_gpu(self):
        self.volume = self.volume.to(self.device) if self.volume is not None else None
        self.phi = self.phi.to(self.device) if self.phi is not None else None
        self.theta = self.theta.to(self.device) if self.theta is not None else None
        self.psi = self.psi.to(self.device) if self.psi is not None else None
        self.translation = self.translation.to(self.device) if self.translation is not None else None
        self.on_gpu = True

    def to_cpu(self):
        self.volume = self.volume.cpu().detach() if self.volume is not None else None
        self.phi = self.phi.cpu().detach() if self.phi is not None else None
        self.theta = self.theta.cpu().detach() if self.theta is not None else None
        self.psi = self.psi.cpu().detach() if self.psi is not None else None
        self.translation = self.translation.cpu().detach() if self.translation is not None else None
        self.on_gpu = False

    def fill_param(self, param):
        if self.len is None:
            return None

        result = 0 if param is None else param
        if not hasattr(result, '__len__'):
            result = th.linspace(result, result, self.len)
        else:
            if len(result) == 1:
                result = th.linspace(result[0], result[0], self.len)
            if len(result) == 2:
                result = th.linspace(result[0], result[1], self.len)
        if result is not None:
            result = th.as_tensor(result)
        return result

    def zoom(self, target_shape):
        vol = self.volume.detach().cpu().squeeze().numpy()

        upscale_factor_1 = target_shape[0] / vol.shape[0]
        upscale_factor_2 = target_shape[1] / vol.shape[1]
        mean_factor = (upscale_factor_1 + upscale_factor_2) / 2

        upscale_factors = (upscale_factor_1, upscale_factor_2, upscale_factor_2)

        vol_upscaled = zoom(vol, upscale_factors)
        vol_upscaled[vol_upscaled < 0] = 0
        vol_upscaled = vol_upscaled / mean_factor  # to keep values on sino the same

        self.write_vol(vol_upscaled)

    def angles_to_rad(self):
        self.phi = th.deg2rad(self.phi)
        self.theta = th.deg2rad(self.theta)
        self.psi = th.deg2rad(self.psi)

    def angles_to_deg(self):
        self.phi = th.rad2deg(self.phi)
        self.theta = th.rad2deg(self.theta)
        self.psi = th.rad2deg(self.psi)

    def all_req_grad(self, is_req=True):
        if self.volume is not None:
            self.volume.requires_grad = is_req
        if self.phi is not None:
            self.phi.requires_grad = is_req
        if self.theta is not None:
            self.theta.requires_grad = is_req
        if self.psi is not None:
            self.psi.requires_grad = is_req

    def reinit_volume(self, shape=None, sino_shape=None):
        if self.volume is not None:
            self.volume = th.zeros(self.volume.shape).to(self.device)

        if sino_shape:
            z, x = sino_shape[-2:]
            shape = (1, 1, z, x, x)
        if shape:
            self.volume = th.zeros(shape, dtype=th.float32).to(self.device)

    def drop_projections(self, array_to_drop):
        self.phi = np.delete(self.phi.cpu().numpy(), array_to_drop, axis=0)
        self.theta = np.delete(self.theta.cpu().numpy(), array_to_drop, axis=0)
        self.psi = np.delete(self.psi.cpu().numpy(), array_to_drop, axis=0)
        self.translation = np.delete(self.translation.cpu().numpy(), array_to_drop, axis=1)
        self.phi = th.as_tensor(self.phi)
        self.theta = th.as_tensor(self.theta)
        self.psi = th.as_tensor(self.psi)
        self.translation = th.as_tensor(self.translation)
        if self.on_gpu:
            self.to_gpu()


    def get_sino(self) -> np.array:
        sino_sims = []
        for k in range(self.len):
            sino_sim = ray_transform(self.volume,
                                     self.phi[k:k+1],
                                     self.theta[k:k+1],
                                     self.psi[k:k+1],
                                     self.translation[:, k:k+1])
            sino_sims.append(sino_sim.squeeze().detach().cpu().numpy())
        sino_sims = np.array(sino_sims)
        return sino_sims

    def get_vol(self) -> np.array:
        return self.volume.detach().cpu().numpy().squeeze()

    def write_vol(self, vol: np.array):
        self.volume = th.as_tensor(vol, device=self.device)
        self.volume.unsqueeze_(0).unsqueeze_(0)

    def save_old(self, path, name: Path, only_vol=False) -> np.array:
        if not os.path.exists(path):
            os.makedirs(path)
        vol = self.volume.squeeze().detach().cpu().numpy()
        np.save(path / (name + '_volume.npy'), vol)
        if not only_vol:
            np.save(path / (name + '_phi.npy'), self.phi.detach().cpu().numpy())
            np.save(path / (name + '_theta.npy'), self.theta.detach().cpu().numpy())
            np.save(path / (name + '_psi.npy'), self.psi.detach().cpu().numpy())
            np.save(path / (name + '_translation.npy'), self.translation.detach().cpu().numpy())
        return vol

    def save(self, path: Path, name: Path, only_vol=False) -> np.array:
        if not os.path.exists(path):
            os.makedirs(path)
        data_dict = {'volume': self.volume.squeeze().detach().cpu().numpy(),
                     'phi': self.phi.detach().cpu().numpy(),
                     'theta': self.theta.detach().cpu().numpy(),
                     'psi': self.psi.detach().cpu().numpy(),
                     'translation': self.translation.detach().cpu().numpy()}
        io.savemat(path / name, data_dict)

    def load_old(self, path, alignment_only=False):
        to_load = ['volume', 'phi', 'theta', 'psi', 'translation']
        if alignment_only:
            to_load = ['phi', 'theta', 'psi', 'translation']
        for name in to_load:
            file_list = glob.glob(str(path) + "/*_" + name + ".npy")
            if file_list:
                tensor = th.as_tensor(np.load(file_list[0]), device=self.device)
                if name == 'volume':
                    tensor.unsqueeze_(0).unsqueeze_(0)
                setattr(self, name, tensor)
            else:
                print(f"Didn't find {name}")
        self.len = len(self.phi)

    def load(self, path, alignment_only=False):
        mat_array = io.loadmat(path)
        self.volume = th.as_tensor(mat_array['volume'], device=self.device).unsqueeze(0).unsqueeze(0)
        names = ['phi', 'theta', 'psi', 'translation']
        for name in names:
            setattr(self, name, th.as_tensor(mat_array[name], device=self.device).squeeze(0))
        self.len = len(self.phi)


    def plot_angles(self, title, init_model):
        plot_angles((self.phi, self.theta, self.psi), title=title,
                    angles_init_tuple=(init_model.phi, init_model.theta, init_model.psi))

    def plot_alignment(self, title="Alignment", init_model=None):
        new_model = copy(self)
        new_model.to_cpu()
        new_model.angles_to_deg()

        was_on_gpu = False

        if init_model:
            if init_model.on_gpu:
                was_on_gpu = True
                init_model.to_cpu()
            init_model.angles_to_deg()

        fig_number = 5 if init_model else 4
        number_of_angles = len(new_model.phi)

        fig, ax = plt.subplots(1, fig_number, figsize=(17, 5))
        ax[0].scatter(np.arange(number_of_angles), new_model.phi.numpy().squeeze())
        ax[1].scatter(np.arange(number_of_angles), new_model.theta.numpy().squeeze())
        ax[2].scatter(np.arange(number_of_angles), new_model.psi.numpy().squeeze())
        ax[3].scatter(new_model.translation.numpy().squeeze()[0], new_model.translation.numpy().squeeze()[1])

        ax[0].set_title('phi')
        ax[1].set_title('theta')
        ax[2].set_title('psi')
        ax[3].set_title('translation')

        if init_model:
            ax[0].scatter(np.arange(number_of_angles), init_model.phi.numpy().squeeze())
            ax[1].scatter(np.arange(number_of_angles), init_model.theta.numpy().squeeze())
            ax[2].scatter(np.arange(number_of_angles), init_model.psi.numpy().squeeze())
            ax[4].scatter(np.arange(number_of_angles), (new_model.phi - init_model.phi).numpy().squeeze())
            ax[4].set_title('phi difference')

        fig.suptitle(title)
        plt.show()

        if init_model:
            if was_on_gpu:
                init_model.to_gpu()
            init_model.angles_to_rad()

    def plot_vol(self, suptitle=''):
        model = self.get_vol()
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        slice = int(model.shape[0]/2)
        ax[0].imshow(model[slice, :, :])
        ax[1].imshow(model[:, slice, :])
        ax[2].imshow(model[:, :, slice])

        fig.suptitle(suptitle)
        plt.show()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def affine_matrix_3D_ZYX(phi, theta, psi, translation):
    c1 = th.cos(phi)
    s1 = th.sin(phi)
    c2 = th.cos(theta)
    s2 = th.sin(theta)
    c3 = th.cos(psi)
    s3 = th.sin(psi)
    line1 = th.stack([c1*c2, c1*s2*s3 - c3*s1, s1*s3 + c1*c3*s2, translation[0]], 1)
    line2 = th.stack([c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3, th.zeros_like(translation[1])], 1)
    line3 = th.stack([-s2, c2*s3, c2*c3, translation[1]], 1)
    R = th.stack([line1, line2, line3], 1)
    return R


def ray_transform(vol, phi_rad, theta_rad, psi_rad, translation):
    n_theta = phi_rad.shape[0]
    R = affine_matrix_3D_ZYX(phi_rad, theta_rad, psi_rad, translation)
    out_size = (n_theta, 1, vol.shape[2], vol.shape[3], vol.shape[4])
    grid = F.affine_grid(R, out_size)
    out = F.grid_sample(vol.expand(n_theta, 1, vol.shape[2], vol.shape[3], vol.shape[4]), grid)
    # print(out.shape)
    # out is (N_batch, channels, Z, Y, X)
    sino = th.sum(out, 3)
    return sino

def rotate_volume(vol, phi_rad, theta_rad, psi_rad, translation):
    n_theta = phi_rad.shape[0]
    R = affine_matrix_3D(phi_rad, theta_rad, psi_rad, translation)
    out_size = (n_theta, 1, vol.shape[2], vol.shape[3], vol.shape[4])
    grid = F.affine_grid(R, out_size)
    out = F.grid_sample(vol.expand(n_theta, 1, vol.shape[2], vol.shape[3], vol.shape[4]), grid)
    # print(out.shape)
    # out is (N_batch, channels, Z, Y, X)
    return out

def shrink_nonnegative(x, tau):
    x = th.abs(x) - tau
    x[th.sign(x) < 0] = 0
    return x

class RadonReconstruction:
    def __init__(self, model: Model, sino_target: th.Tensor, reg_pars: dict = None, save_path=None,
                 lr_model=0.02, vol_scheduler: dict = None, align_scheduler: dict = None, lrs_alignment: dict = None):
        self.model = model
        self.sino_target = sino_target
        self.save_path = save_path

        self.reg_pars = reg_pars
        self.lr_model = lr_model

        self.vol_scheduler = vol_scheduler
        self.align_scheduler = align_scheduler

        self.lrs_alignment = lrs_alignment

        # Tensor penalizing the density at the edges
        self.penalize_tensor = None
        self.reg_edges = 0
        self.edge_pow = 5.0
        self.edge_y_squeeze_factor = 1

        # Store initial model alignment
        self.model_init = copy(model)
        del self.model_init.volume
        self.model_init.volume = None

        # Angle regularization factor
        self.angle_reg = 0.10
        self.width_phi = 0.25   # deg
        self.width_theta = 0.5   # deg
        self.width_psi = 0.5   # deg
        self.width_theta_frac = 0.5  # percent of diff phi
        self.width_psi_frac = 0.5  # percent of diff phi

        self._phi_diff_init = None
        self._k_phi = None
        self._k_theta = None
        self._k_psi = None

        # Sub volume reconstruction
        self.sub_volume = None

        # Gaussian kernel regularization params
        self.gauss_kernel = None


    def fit_volume(self, num_iterations, batch_size, progbar=True, print_logs=False, tau=1e-4):
        self.model.all_req_grad(False)
        device = self.sino_target.device
        self.model.volume.requires_grad = True
        losses = []

        # print_memory_usage(th.device('cuda:0'), info=f'before fitting')
        if self.reg_edges > 0:
            _, _, z, y, x = self.model.volume.shape
            self.make_edge_tensor(z, y, x)


        optimizer_model = Adam([self.model.volume], self.lr_model)
        if self.vol_scheduler is not None:
            scheduler = self.apply_scheduler(optimizer_model, self.vol_scheduler)

        sampler = BatchSampler(SequentialSampler(range(self.sino_target.shape[0])),
                               batch_size=batch_size, drop_last=False)
        n_batches = len(sampler)
        epochs = tqdm(range(num_iterations)) if progbar else range(num_iterations)

        for epoch in epochs:
            th.cuda.empty_cache()

            optimizer_model.zero_grad()
            loss = 0

            for batch in sampler:
                sino_sim = ray_transform(self.model.volume,
                                         self.model.phi[batch],
                                         self.model.theta[batch],
                                         self.model.psi[batch],
                                         self.model.translation[:, batch])

                current_loss = mse_loss(sino_sim, self.sino_target[batch])
                current_loss /= n_batches
                current_loss.backward()
                loss += current_loss

            loss_edges = self.reg_edges * self.penalize_edges(self.model.volume)
            if self.penalize_tensor is not None:
                loss_edges.backward()
            if print_logs:
                print(f'loss: {loss.item()}, loss edges: {loss_edges}')
            losses.append(loss.item())
            loss += loss_edges

            optimizer_model.step()

            with th.no_grad():
                self.model.volume[:, :, :, :, :] = shrink_nonnegative(self.model.volume, tau)
                if self.reg_pars is not None:
                    self.FGP_TV_regularization()
                if self.gauss_kernel is not None:
                    self.model.volume[:,:,:,:,:] = self.gaussian_blur_regularization()  # ?????????
                    self.model.volume.requires_grad = True

            if self.vol_scheduler is not None:
                scheduler.step()

        vol = self.model.get_vol()

        return vol, np.array(losses)

    def FGP_TV_regularization(self):
        vol = self.model.volume.detach().cpu().numpy().squeeze()
        (vol, info_vec_gpu) = FGP_TV(vol,
                                     self.reg_pars['regularisation_parameter'],
                                     self.reg_pars['number_of_iterations'],
                                     self.reg_pars['tolerance_constant'],
                                     self.reg_pars['methodTV'],
                                     self.reg_pars['nonneg'], 'gpu')

        self.model.volume[0, 0, :, :, :] = th.as_tensor(vol, device=self.model.device)

    # def mse_loss(self, input, target, weights=None):
    #     squared_errors = th.square(input - target)
    #     if weights is None:
    #         weights = th.ones_like(squared_errors, device=self.model.device)
    #     weighted_squared_errors = squared_errors * weights
    #     weighted_mse_loss = torch.mean(weighted_squared_errors)
    #     return weighted_mse_loss
    #
    # def init_penalize_edges(self):
    #     b, _, h, w = self.sino_target.shape
    #     y, x = th.meshgrid(th.arange(h), th.arange(w))
    #     center_y, center_x = h // 2, w // 2
    #     distance_from_center = th.sqrt(th.clamp((y - center_y) ** 2 + (x - center_x) ** 2, 0.01))
    #     if torch.isnan(distance_from_center).any():
    #         raise ValueError("The distance_from_center contains NaN values.")
    #
    #     edge_tensor = distance_from_center.max() - distance_from_center
    #     edge_tensor = edge_tensor / edge_tensor.max()
    #     edge_tensor = th.pow(edge_tensor, self.edge_pow)
    #     if torch.isnan(edge_tensor).any():
    #         raise ValueError("The edge_tensor contains NaN values.")
    #     return edge_tensor


    def penalize_edges(self, volume):
        if self.penalize_tensor is None:
            return 0
        term = volume * self.penalize_tensor
        num_elements = term.numel()
        reg_term = th.abs(term)
        return th.sum(reg_term) / num_elements

    def make_edge_tensor(self, depth, height, width):
        # Create a meshgrid of coordinates for the tensor
        z, y, x = th.meshgrid(th.arange(depth), th.arange(height), th.arange(width))

        # Calculate the Euclidean distance from the center
        k = self.edge_y_squeeze_factor
        center_z, center_y, center_x = depth // 2, height // 2, width // 2
        distance_from_center = th.sqrt(th.clamp((z - center_z) ** 2 + k*(y - center_y) ** 2 + (x - center_x) ** 2, 0.01))
        if torch.isnan(distance_from_center).any():
            raise ValueError("The distance_from_center contains NaN values.")

        edge_tensor = distance_from_center.max() - distance_from_center
        edge_tensor = edge_tensor / edge_tensor.max()
        edge_tensor = 1 - edge_tensor
        edge_tensor = edge_tensor / edge_tensor[0, center_y, center_x]
        edge_tensor = th.pow(edge_tensor, self.edge_pow)

        #edge_tensor_batched = edge_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1)
        edge_tensor_batched = edge_tensor.unsqueeze(0).unsqueeze(0)
        if torch.isnan(edge_tensor_batched).any():
            raise ValueError("The edge_tensor_batched contains NaN values.")

        self.penalize_tensor = edge_tensor_batched
        self.penalize_tensor = self.penalize_tensor.to(self.model.device)
        self.penalize_tensor.requires_grad = False

    def init_gauss_blur_regularization(self, kernel_size, sigma):
        self.gauss_kernel = create_3d_gaussian_kernel(kernel_size, sigma).unsqueeze(0)
        self.gauss_kernel = self.gauss_kernel.to(self.model.device)
        self.gauss_kernel.requires_grad = False

    def gaussian_blur_regularization(self):
        volume = self.model.volume.detach()
        return filter3d(volume, self.gauss_kernel)

    @staticmethod
    def apply_scheduler(optimizer, scheduler_info: dict) -> torch.optim.lr_scheduler:   # not tested
        sched_func = scheduler_info['function']

        if 'max_lr' not in scheduler_info['sched_params']:
            scheduler_info['sched_params'] = get_lr(optimizer)

        scheduler = sched_func(optimizer, **(scheduler_info['sched_params']))
        if sched_func is CyclicLR:
            if scheduler_info['skip_steps']:
                for _ in range(scheduler_info['sched_params']['step_size_up']):  # to start with max LR
                    scheduler.step()
        return scheduler

    def init_optimizers(self):
        if self.lrs_alignment is None:
            self.lrs_alignment = {'fac': 1, 'lr_phi': 1e-7, 'lr_theta': 1e-7, 'lr_psi': 1e-8, 'lr_translation': 1e-6}

        return (
            Adam([self.model.phi], self.lrs_alignment['lr_phi'] * self.lrs_alignment['fac']),
            Adam([self.model.theta], self.lrs_alignment['lr_theta'] * self.lrs_alignment['fac']),
            Adam([self.model.psi], self.lrs_alignment['lr_psi'] * self.lrs_alignment['fac']),
            Adam([self.model.translation], self.lrs_alignment['lr_translation'] * self.lrs_alignment['fac'])
        )

    def fit_alignment(self, num_iterations, num_inner_iterations=100, batch_size=1, progbar=True, exp_view=True,
                      to_fit={'phi': False, 'theta': False, 'psi': False, 'translation': True}):
        self.model.all_req_grad(False)

        losses = []
        self.angles_regul_init()

        optimizers = self.init_optimizers()
        if self.align_scheduler is not None:
            schedulers = [self.apply_scheduler(optimizer, self.align_scheduler) for optimizer in optimizers]

        epoch_to_show = 5

        sampler = BatchSampler(SequentialSampler(range(self.sino_target.shape[0])),
                               batch_size=batch_size, drop_last=False)

        epochs = tqdm(range(num_iterations)) if progbar else range(num_iterations)
        # logs = open(base_path / 'logs.txt', 'a')
        for epoch in epochs:
            self.model.all_req_grad(False)
            self.model.volume.requires_grad = True

            self.model.reinit_volume()
            _, inner_losses = self.fit_volume(num_iterations=num_inner_iterations,
                                              batch_size=batch_size, progbar=False, print_logs=False)

            self.model.volume = self.model.volume.detach().clone()
            self.model.phi.requires_grad = to_fit['phi']
            self.model.theta.requires_grad = to_fit['theta']
            self.model.psi.requires_grad = to_fit['psi']
            self.model.translation.requires_grad = to_fit['translation']
            self.model.volume.requires_grad = False

            for optimizer in optimizers:
                optimizer.zero_grad()

            losses_it = []

            for batch in sampler:
                sino_sim = ray_transform(self.model.volume,
                                         self.model.phi[batch],
                                         self.model.theta[batch],
                                         self.model.psi[batch],
                                         self.model.translation[:, batch])

                loss = mse_loss(sino_sim, self.sino_target[batch])
                loss_angles = self.angles_regularization_diff()
                loss += loss_angles
                losses_it.append(loss.item())
                loss.backward()

            losses_it = np.array(losses_it).sum()
            losses.append(losses_it)

            # break
            for optimizer in optimizers:
                optimizer.step()

            # Step
            if self.align_scheduler is not None:
                for scheduler in schedulers:
                    scheduler.step()

            if epoch % 5 == 0:
                print(f'{str(epoch).ljust(4)}: loss = {loss.item()} \t loss_angles = {loss_angles.item()}')
                self.model.save(self.save_path, 'running.mat')
                vol = self.model.volume.squeeze().detach().cpu().numpy()

                if epoch % epoch_to_show == 0:
                    if exp_view:
                        epoch_to_show *= 2
                    self.model.plot_alignment(f'Epoch {epoch:03d}', init_model=self.model_init)
                    plot_losses_model(inner_losses, vol, losses, f'Epoch {epoch:03d}')

            # print_memory_usage(self.model.device, log_file=None, info=f"Epoch {epoch:03d}")
        self.model.plot_alignment(f'Epoch {epoch:03d}', init_model=self.model_init)
        plot_losses_model(inner_losses, vol, losses, f'Epoch {epoch:03d}')
        # logs.close()
        return vol, inner_losses, losses

    def angles_regul_init(self):
        # degree _____________
        width_phi = self.width_phi
        width_theta = self.width_theta
        width_psi = self.width_psi
        # _____________________

        width_phi *= (th.pi / 180)
        width_theta *= (th.pi / 180)
        width_psi *= (th.pi / 180)

        self._k_phi = (width_phi ** 2)
        self._k_theta = (width_theta ** 2)
        self._k_psi = (width_psi ** 2)
        # self._k_theta = (self.width_theta_frac ** 2)
        # self._k_psi = (self.width_psi_frac ** 2)

        self._phi_diff_init = th.diff(self.model_init.phi)

    # def angles_regul_init(self):
    #     # degree _____________
    #     width_phi = self.width_phi
    #     width_theta = self.width_theta
    #     width_psi = self.width_psi
    #     # _____________________
    #
    #     # For old regularization
    #     # rad
    #     width_phi *= (th.pi / 180)
    #     width_theta *= (th.pi / 180)
    #     width_psi *= (th.pi / 180)
    #
    #     k_phi = (width_phi ** 2)
    #     k_theta = (width_theta ** 2)
    #     k_psi = (width_psi ** 2)
    #
    #     return k_phi, k_theta, k_psi
    #
    # def angles_regularization(self, k_phi, k_theta, k_psi, print_loses=False):
    #     if self.angle_reg == 0:
    #         return th.tensor(0)
    #     mean_loss = 0.010
    #
    #     reg_loss_phi = mse_loss(self.model.phi, self.model_init.phi) / k_phi
    #     reg_loss_theta = mse_loss(self.model.theta, self.model_init.theta) / k_theta
    #     reg_loss_psi = mse_loss(self.model.psi, self.model_init.psi) / k_psi
    #
    #     if print_loses:
    #         print(f'reg_loss_phi: {reg_loss_phi.item()}')
    #         print(f'reg_loss_theta: {reg_loss_theta.item()}')
    #         print(f'reg_loss_psi: {reg_loss_psi.item()}')
    #
    #     reg_loss = th.exp(reg_loss_phi + reg_loss_theta + reg_loss_psi) - 1
    #     reg_loss *= mean_loss * self.angle_reg
    #
    #     return reg_loss

    def angles_regularization_diff(self, print_loses=False):
        if self.angle_reg == 0:
            return th.tensor(0)
        mean_loss = 0.010

        reg_loss_phi = mse_loss(self.model.phi, self.model_init.phi) / self._k_phi
        reg_loss_theta = th.mean(th.square(th.diff(self.model.theta))) / self._k_theta
        reg_loss_psi = th.mean(th.square(th.diff(self.model.psi))) / self._k_psi

        if print_loses:
            print(f'reg_loss_phi: {reg_loss_phi.item()}')
            print(f'reg_loss_theta: {reg_loss_theta.item()}')
            print(f'reg_loss_psi: {reg_loss_psi.item()}')

        reg_loss = th.exp(reg_loss_phi + reg_loss_theta + reg_loss_psi) - 1
        reg_loss *= mean_loss * self.angle_reg

        return reg_loss


if __name__ == "__main__":
    print(
        mse_loss(th.Tensor([0,1]), th.Tensor([1,3]))
          )
    print(
        th.tensor(2.5).item()
    )

    # Define the kernel size and sigma
    kernel_size = 9  # You can adjust this as needed
    sigma = 1.0  # You can adjust this as needed

    # Create the 3D Gaussian kernel
    gaussian_kernel = create_3d_gaussian_kernel(kernel_size, sigma).unsqueeze(0)

    # Print the kernel (optional)
    print(gaussian_kernel.shape)

    plt.plot(gaussian_kernel[0, kernel_size // 2 + 1, kernel_size // 2 + 1])
    plt.show()
