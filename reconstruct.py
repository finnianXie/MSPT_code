import matplotlib.pyplot as plt
import torch
import numpy as np
import torch as th
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, CyclicLR
from fastatomography.util import plotmosaic
from ccpi.filters.regularisers import ROF_TV, FGP_TV, PD_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, Diff4th
from copy import deepcopy as copy
from pathlib import Path
from utils_andrey import skip_steps, plot_losses_model, plot_angles, DataStack, Model, RadonReconstruction, print_memory_usage
from utils_andrey import rotate_volume
from scipy import io
from skimage import filters

th.cuda.empty_cache()
#%%
# print_memory_usage(th.device('cuda:0'))
# log_file_path = base_path / 'logs.txt'
# log_file = open(log_file_path, 'w')
# print_memory_usage(th.device('cuda:0'), log_file=log_file, info='before optimization')
# # log_file.close()
#%%
# Constans
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(False)

device = th.device('cuda:0')
device2 = th.device('cuda:1')
# device = th.device('cpu')

base_path = Path('/home/andrey/nanocube_save/')
if not os.path.exists(base_path):
    base_path = Path('/home/andrey/nanocube')
recon_path = base_path / 'reconstruct_ptycho'
res_path = base_path / 'reconstruct_results'

fn_stack = 'full_stack_38_projections.mat'
fn_stack_masked = 'masked_stack_36_projections.npy'
fn_angles = 'angles.txt'
theta = np.load(base_path/ 'stack_rotation_correction/thetas.npy')
#%%
np.mean(np.abs(np.diff((theta))))
#%%
# # Loading data
# stack = DataStack(path_stack=base_path / fn_stack, path_angles=base_path / fn_angles, device=device)
#
# # Making rearrangement and cropping
# stack.preprocessing()
# plotmosaic(stack.images, title='Stack cropped', dpi=600)
#
# # Transposing data
# stack.images = np.transpose(stack.images, (0, 2, 1))
# plotmosaic(stack.images, title='Stack transposed', dpi=600)
#
# # Correcting across rotation
# stack.gauss_correction()
# plotmosaic(stack.images, title='Stack gauss corrected', dpi=600)
#%%
####################################################################
# Denoising stack
images_denoised = np.load(base_path / 'stack_36_projections_denoise_bm3d.npy')
# plotmosaic(images_denoised, title='Stack denoised gauss corrected', dpi=600)
#%%
stack_denoised = DataStack(path_stack=base_path / fn_stack, path_angles=base_path / fn_angles, device=device)
stack_denoised.images = images_denoised
# Correcting across rotation
stack_denoised.gauss_correction()
plotmosaic(stack_denoised.images, title='Stack denoised gauss corrected', dpi=600)
#%%
stack = stack_denoised
#%%
test_stack = copy(stack)
test_stack = test_stack.clip(0.4)
test_stack = test_stack.rescale_images(0.10)
test_stack.images = test_stack.normalization(test_stack.images)
# test_stack = test_stack.clip(0.02)
plotmosaic(test_stack.images, dpi=600)
#%%
####################################################
# Translation fitting
stack_rescaled = copy(stack)
# Clipping
stack_rescaled = stack_rescaled.clip(0.375)
plotmosaic(stack_rescaled.images, title='Stack clipped', dpi=600)


# Rescaling and normalization
stack_rescaled = stack_rescaled.rescale_images(scale=0.10)
# stack_rescaled = stack
stack_rescaled.images = stack.normalization(stack_rescaled.images)
plotmosaic(stack_rescaled.images, title='Stack rescaled', dpi=600)
print(stack_rescaled.images.shape)

# # Padding
# stack_rescaled = stack_rescaled.padding(frac=0.10)
# plotmosaic(stack_rescaled.images, title='Stack padding', dpi=600)
stack_rescaled.images.shape

#%%
# Define sino_target
sino_target = stack_rescaled.to_sino_data()
sino_target.shape
#%%
# Initializing the model
model = Model(device=device, phi=stack.angles, theta=0,
              sino_shape=sino_target.shape)
model.angles_to_rad()
model.plot_alignment(title='Before optimization')
model.to_gpu()

model_init = copy(model)
model.print_shapes()
#%%
print(model.volume.shape)
print(sino_target.shape)
#%%
# Store initial model on CPU
model_init = copy(model)
model_init.to_cpu()
#%%
# Load initial model
model = copy(model_init)
model.to_gpu()
model.plot_alignment(title='Before optimization')
#%%
# Volume fitting
model.to_gpu()
model.reinit_volume()


reg_pars = {'algorithm': FGP_TV,
            'regularisation_parameter': 20e-6,
            'number_of_iterations': 50,
            'tolerance_constant': 1e-06,
            'methodTV': 0,
            'nonneg': 1}

# vol_scheduler = {
#     'function': CyclicLR,
#     'skip_steps': True,
#     'sched_params': {
#             'base_lr': 0.000005,
#             'max_lr': 0.00005,  # 0.5
#             'step_size_up': 20,  # 20
#             'mode': "triangular2",
#             'cycle_momentum': False}
# }
#
# align_scheduler = {
#     'function': CyclicLR,
#     'skip_steps': True,
#     'sched_params': {
#             'base_lr': 1e-7,
#             'step_size_up': 100,  # 20
#             'mode': "triangular2",
#             'cycle_momentum': False}
# }

radon = RadonReconstruction(
    model=model,
    sino_target=sino_target,
    reg_pars=reg_pars,
    vol_scheduler=None,
    lr_model=0.001,
    save_path=res_path
)
#%%
edge_tensor = radon.penalize_tensor.cpu().squeeze().numpy()
fig, ax = plt.subplots(2,2)
ax[0, 0].imshow(edge_tensor[32,:,:])
ax[0, 1].plot(edge_tensor[32,32,:])
ax[1, 0].imshow(stack_rescaled.images[27])
ax[1, 1].plot(stack_rescaled.images[27])
plt.show()
#%%
model.reinit_volume()
radon.reg_edges = 5
vol, losses = radon.fit_volume(num_iterations=50, batch_size=36)
plot_losses_model(losses, vol, losses, "vol fit")
plotmosaic(model.get_sino(), title='sino sim', dpi=600)
plotmosaic(stack_rescaled.images, title='Stack exp', dpi=600)
model.save(res_path, 'running_2.mat')
#%%
# Fitting alignment
radon.angle_reg = 0
radon.reg_edges = 5
to_fit = {'phi': False, 'theta': False, 'psi': False, 'translation': True}
radon.lrs_alignment = {'fac': 1, 'lr_phi': 1e-4, 'lr_theta': 1e-4, 'lr_psi': 1e-5, 'lr_translation': 1e-5}
vol, inner_losses, losses = radon.fit_alignment(num_iterations=10000, num_inner_iterations=50, batch_size=36,
                                                progbar=True, exp_view=True, to_fit=to_fit)
model.save(res_path / 'alignment', 'translation_fit_2.mat')
#%%
model.load(res_path / 'running.mat')
#%%
model.save(res_path / 'alignment', 'translation_fit.mat')
model.plot_alignment(title='After trans optimization')
#%%
# Load model in the run
model.load(base_path)
#%%
# Plot resulted alignment
model.plot_alignment(title='After optimization', init_model=model_init)

#%%
# Plotting resulted sino to check
plotmosaic(model_init.get_sino(), title='sino before', dpi=400)
plotmosaic(model.get_sino(), title='sino after', dpi=400)
plotmosaic(stack_rescaled.images, title='Stack padding', dpi=400)
#%%
plotmosaic(stack_rescaled.images - model_init.get_sino(), title='sino brfore', dpi=400)
plotmosaic(stack_rescaled.images - model.get_sino(), title='sino after', dpi=400)
#%%
model.load(res_path / 'alignment/translation_fit.mat')
model_sino = model.get_sino()
plotmosaic(model_sino, title='sino after', dpi=400)
fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.imshow(model_sino[28], cmap='viridis')
ax.axis('off')
plt.show()
#%%




#%%
####################################################
# Rotations fitting
stack_to_fit = copy(stack)
# Clipping
stack_to_fit = stack_to_fit.clip(0.4)
# plotmosaic(stack_to_fit.images, title='Stack clipped', dpi=600)

# Rescaling and normalization
stack_to_fit = stack_to_fit.rescale_images(scale=0.20)
# stack_rescaled = stack
stack_to_fit.images = stack.normalization(stack_to_fit.images)
# plotmosaic(stack_to_fit.images, title='Stack rescaled', dpi=600)

# Padding
stack_to_fit = stack_to_fit.padding(frac=0.10)
plotmosaic(stack_to_fit.images, title='Stack padding', dpi=600)
stack_to_fit.images.shape
#%%
# Define sino_target
sino_target = stack_to_fit.to_sino_data()
sino_target.shape
#%%
# Initializing the model
model = Model(device=device)
load_model_path = res_path / 'alignment/translation_fit.mat'
model.load(load_model_path)
model.theta = th.zeros_like(model.phi, device=device)
model.plot_alignment(title='Before optimization')
model.volume = None
model.reinit_volume(sino_shape=sino_target.shape)
model.to_gpu()

model_init = copy(model)
model.print_shapes()
#%%
print(model.volume.shape)
print(sino_target.shape)
#%%
# Store initial model on CPU
model_init = copy(model)
model_init.to_cpu()
#%%
# Load initial model
model = copy(model_init)
model.to_gpu()
model.plot_alignment(title='Before optimization')
#%%
# Volume fitting
model.to_gpu()
model.reinit_volume()


reg_pars = {'algorithm': FGP_TV,
            'regularisation_parameter': 20e-6,
            'number_of_iterations': 50,
            'tolerance_constant': 1e-06,
            'methodTV': 0,
            'nonneg': 1}

# vol_scheduler = {
#     'function': CyclicLR,
#     'skip_steps': True,
#     'sched_params': {
#             'base_lr': 0.000005,
#             'max_lr': 0.00005,  # 0.5
#             'step_size_up': 20,  # 20
#             'mode': "triangular2",
#             'cycle_momentum': False}
# }
#
# align_scheduler = {
#     'function': CyclicLR,
#     'skip_steps': True,
#     'sched_params': {
#             'base_lr': 1e-7,
#             'step_size_up': 100,  # 20
#             'mode': "triangular2",
#             'cycle_momentum': False}
# }

radon = RadonReconstruction(
    model=model,
    sino_target=sino_target,
    reg_pars=reg_pars,
    vol_scheduler=None,
    lr_model=0.001,
    save_path=res_path
)
#%%
th.cuda.empty_cache()
#%%
radon.model = model_init
radon.model.to_gpu()
radon.model.reinit_volume()
radon.reg_edges = 10
radon.edge_y_squeeze_factor = 4
vol, losses = radon.fit_volume(num_iterations=50, batch_size=36)
plot_losses_model(losses, vol, losses, "vol fit")
sino_sim_before = radon.model.get_sino()
plotmosaic(sino_sim_before, title='before', dpi=600)
radon.model.save(res_path, 'running_2.mat')
radon.model.to_cpu()
#%%
# Fitting alignment
radon.model = model
radon.model.reinit_volume()
radon.angle_reg = 0.01
radon.reg_edges = 10
radon.edge_y_squeeze_factor = 4
to_fit = {'phi': True, 'theta': True, 'psi': True, 'translation': False}
radon.lrs_alignment = {'fac': 10, 'lr_phi': 1e-4, 'lr_theta': 1e-4, 'lr_psi': 1e-5, 'lr_translation': 1e-2}
vol, inner_losses, losses = radon.fit_alignment(num_iterations=5000, num_inner_iterations=50, batch_size=36,
                                                progbar=True, exp_view=True, to_fit=to_fit)
model.save(res_path / 'alignment', 'rotation_fit_2.mat')
#%%
model.load(res_path / 'running.mat')
#%%
model.plot_alignment(title='After trans optimization', init_model=model_init)
#%%

#%%
model.save(res_path / 'alignment', 'rotation_fit_2.mat')
model.plot_alignment(title='After trans optimization', init_model=model_init)
#%%
sino_sim_after = model.get_sino()
plotmosaic(sino_sim_before, title='before', dpi=600)
plotmosaic(sino_sim_after, title='after', dpi=600)
#%%
index = 29

fig, ax = plt.subplots(1,3, figsize=(10, 5), dpi=300)
ax[0].imshow(stack_to_fit.images[index])
ax[1].imshow(sino_sim_before[index])
ax[2].imshow(sino_sim_after[index])
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()
#%%
model.save(res_path / 'alignment', 'rotation_fit_2_plato.mat')
model.plot_alignment(title='After trans optimization', init_model=model_init)
#%%












#%%
############################################################################
# Making Mask volume
stack_rescaled = copy(stack)
# Clipping
stack_rescaled = stack_rescaled.clip(0.4)
plotmosaic(stack_rescaled.images, title='Stack clipped', dpi=600)

# Rescaling and normalization
stack_rescaled = stack_rescaled.rescale_images(scale=0.10)
# stack_rescaled = stack
stack_rescaled.images = stack.normalization(stack_rescaled.images)
plotmosaic(stack_rescaled.images, title='Stack rescaled', dpi=600)

# # Padding
# stack_rescaled = stack_rescaled.padding(frac=0.10)
# plotmosaic(stack_rescaled.images, title='Stack padding', dpi=600)
sino_target = stack_rescaled.to_sino_data()
stack_rescaled.images.shape
# _______________________________________________________
#%%
test_images = stack_rescaled.images
smoothed_images = np.array([filters.gaussian(image, sigma=2) for image in test_images])
plotmosaic(smoothed_images, title='Stack rescaled', dpi=600)
#%%
# Initializing the model, loading predefined alignment
load_model_path = res_path / 'alignment/rotation_fit_2.mat'
model = Model(device=device)
model.load(load_model_path)
model.volume = None
model.reinit_volume(sino_shape=sino_target.shape)
model.plot_alignment(title='Before optimization')
model.to_gpu()

model.print_shapes()
print(stack_rescaled.images.shape)

losses_all = []

reg_pars = {'algorithm': FGP_TV,
            'regularisation_parameter': 1000e-6,
            'number_of_iterations': 50,
            'tolerance_constant': 1e-06,
            'methodTV': 0,
            'nonneg': 1}

radon = RadonReconstruction(
    model=model,
    sino_target=sino_target,
    reg_pars=reg_pars,
    vol_scheduler=None,
    lr_model=1e-3,
    save_path=base_path
)
#%%
radon.model = model
radon.model.reinit_volume()
radon.reg_edges = 10
radon.edge_pow = 10
radon.edge_y_squeeze_factor = 2
radon.reg_pars['regularisation_parameter'] = 1e-3

_, _, z, y, x = radon.model.volume.shape
radon.make_edge_tensor(z, y, x)
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.plot(radon.penalize_tensor.squeeze().cpu().numpy()[z//2, y//2, :])
ax.plot(radon.penalize_tensor.squeeze().cpu().numpy()[z//2, :, x//2])
plt.show()
#%%
vol, losses = radon.fit_volume(num_iterations=100, batch_size=36, progbar=True)
losses_all = np.append(losses_all, losses)
plot_losses_model(losses, vol, losses_all, f'mask volume')
plotmosaic(model.get_sino(), title='model', dpi=600)
_ = model.save(res_path/ 'mask_vol', "mask_test.mat")
#%%
# Load low res mask
model = Model(device)
model.load(res_path/ 'mask_vol/mask_test.mat')
plotmosaic(model.get_sino(), title='model', dpi=600)

# Convert to binary and blur
threshold = 0.002
model.volume[model.volume > threshold] = 1
model.volume[model.volume < threshold] = 0
model.save(res_path/ 'mask_vol', "mask_binary.mat")
model.volume = th.as_tensor(filters.gaussian(model.volume.cpu().squeeze().numpy(), sigma=2), device=device).unsqueeze(0).unsqueeze(0)
plotmosaic(model.get_sino(), title='model', dpi=600)
model.save(res_path/ 'mask_vol', "mask_gauss.mat")
#%%
# Upscale volume
_, x, y = stack.images.shape
model.zoom((x, y))
model.save(res_path/ 'mask_vol', "mask_full_res.mat")
#%%
# Get mask sino
sino_mask = model.get_sino()
sino_mask = sino_mask / sino_mask.max()
print(sino_mask.shape)
plotmosaic(sino_mask, title='sino_mask', dpi=300)
#%%
# Convert to binary and blur
sino_mask_binary = copy(sino_mask)
threshold = 0.1
sino_mask_binary[sino_mask_binary > threshold] = 1
sino_mask_binary[sino_mask_binary < threshold] = 0
plotmosaic(sino_mask_binary, title='sino_mask_binary', dpi=300)
smoothed_mask = np.array([filters.gaussian(image, sigma=50) for image in sino_mask_binary])
plotmosaic(smoothed_mask, title='smoothed_mask', dpi=300)
#%%
# Masking orig sino
images_masked = stack.images * smoothed_mask
plotmosaic(stack.images, title='smoothed_mask', dpi=600)
plotmosaic(images_masked, title='smoothed_mask', dpi=600)
#%%
np.save(base_path / 'stack_masked.npy', images_masked)
#%%









#%%
############################################################################
# Full resolution reconstruction with masked stack
images_masked = np.load(base_path / 'stack_masked.npy')
# plotmosaic(images_denoised, title='', dpi=600)

stack_masked = DataStack(path_stack=base_path / fn_stack, path_angles=base_path / fn_angles, device=device)
stack_masked.images = images_masked

sino_target = stack_masked.to_sino_data()
plotmosaic(images_masked, title='', dpi=600)
sino_target.shape
# _______________________________________________________
#%%
# Initializing the model, loading predefined alignment
# load_model_path = res_path / 'upscaled' # load after upscaling
load_model_path = res_path / 'alignment/rotation_fit_2.mat'  # load after upscaling
model = Model(device=device)
model.load(load_model_path)
model.plot_alignment(title='Alignment')
model.reinit_volume(sino_shape=sino_target.shape)
model.print_shapes()
model.to_gpu()
sino_target.shape
#%%
np.save(base_path / 'angles.npy', model.phi.cpu().numpy())
#%%
# Exclude poor projections
to_drop = [0, 1, 2, 33, 34, 35]
stack_to_fit = stack_masked.discard_images(to_drop)
plotmosaic(stack_to_fit.images, dpi=600)

sino_target = stack_to_fit.to_sino_data()

# Exclude poor projections for model
model.reinit_volume()
model.drop_projections(to_drop)
#%%
th.cuda.empty_cache()
#%%
model.reinit_volume()
losses_all = []
#%%
radon = RadonReconstruction(
    model=model,
    sino_target=sino_target,
    reg_pars=None,
    vol_scheduler=None,
    lr_model=1e-4,
    save_path=base_path
)
radon.reg_edges = 5
kernel_size = 7
radon.init_gauss_blur_regularization(kernel_size, 0.5)
# fig, ax = plt.subplots()
# ax.plot(radon.gauss_kernel[0, kernel_size // 2, kernel_size // 2].cpu().numpy())
# plt.show()

for _ in range(3):
    vol, losses = radon.fit_volume(num_iterations=100, batch_size=1, progbar=True, print_logs=True)
    losses_all = np.append(losses_all, losses)
    plot_losses_model(losses, vol, losses_all, f'Scale 1')
    model.save(res_path / 'upscaled', "full_res_new.mat")
    # break

np.save(res_path / 'losses.npy', losses_all)
#%%
radon.penalize_tensor
#%%
torch.isnan(radon.penalize_tensor).any()
#%%
load_model_path = res_path / 'upscaled/full_res_new.mat' # load after upscaling
model = Model(device=device)
model.load(load_model_path)
#%%
model.volume.shape
#%%
sino_model = model.get_sino()
#%%
plotmosaic(sino_model, dpi=600)
#%%
sino_model.shape
#%%









#%%
####################################
# Save simulated sinogramm w/wo translations
# model_path = res_path / 'upscaled/full_res.mat'
model_path = res_path / 'masked_full_res_model.mat'
model = Model(device=device)
model.load(model_path)

from scipy import io
vol = io.loadmat(res_path / 'vol_thresholded_0.0025.mat')['volume']
model.volume = th.as_tensor(vol, device=device).unsqueeze(0).unsqueeze(0)
# model.plot_alignment()

model_sino = model.get_sino()

plotmosaic(model_sino, title='with translations', dpi=600)

model.translation = th.zeros_like(model.translation)
model_sino_no_trans = model.get_sino()
#%%
plotmosaic(model_sino_no_trans, title='without translations', dpi=600)
#%%
np.save(base_path / 'sino_sim_no_trans.npy', model_sino_no_trans)






#%%
##################################################
# Save simulated sinogramm from atom tracing
volume_path = res_path / 'vol_scatter_fullsize_2.npy'
model_path = res_path / 'masked_full_res_model.mat'
model = Model(device=device)
model.load(model_path)

model.volume = th.as_tensor(np.load(volume_path), device=device).unsqueeze(0).unsqueeze(0)
#%%
model_sino = model.get_sino()

plotmosaic(model_sino, title='with translations', dpi=600)

model.translation = th.zeros_like(model.translation)
model_sino_no_trans = model.get_sino()

plotmosaic(model_sino_no_trans, title='without translations', dpi=600)

#%%
np.save(base_path / 'sino_traced_no_trans.npy', model_sino_no_trans)




#%%
##################################################
# Rotate volume
model_path = res_path / 'mask_vol.mat'
model = Model(device=device)
model.load(model_path)

angle = 90 * th.pi/180
angle2 = 45 * th.pi/180
phi = th.tensor([angle], device=device)
theta = th.tensor([angle2], device=device)
psi = th.tensor([angle], device=device)
trans = th.tensor([[0], [0]], device=device)

rotated_volume = rotate_volume(model.volume, phi, theta, psi, trans)

vol = rotated_volume.squeeze().cpu().numpy()
vol_orig = model.volume.squeeze().cpu().numpy()

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[1, 0].imshow(vol[40, :, :])
ax[1, 1].imshow(vol[:, 40, :])
ax[1, 2].imshow(vol[:, :, 40])
ax[0, 0].imshow(vol_orig[40, :, :])
ax[0, 1].imshow(vol_orig[:, 40, :])
ax[0, 2].imshow(vol_orig[:, :, 40])

plt.show()
np.save(res_path / 'rotation_test.npy', vol)
#%%

#%%
##################################################
# Rotate volume
model_path = res_path / 'masked_full_res_model.mat'
model = Model(device=device)
model.load(model_path)

#%%
# vol = io.loadmat(res_path / 'vol_thresholded_0.0015.mat')['volume']
vol = io.loadmat(res_path / 'mask_vol_full_res.mat')['volume']
model.volume = th.as_tensor(vol, device=device).unsqueeze(0).unsqueeze(0)
#%%
angle = 90 * th.pi/180
angle2 = 45 * th.pi/180
phi = th.tensor([angle], device=device)
theta = th.tensor([angle2], device=device)
psi = th.tensor([angle], device=device)
trans = th.tensor([[0], [0]], device=device)

rotated_volume = rotate_volume(model.volume, phi, theta, psi, trans)

vol = rotated_volume.squeeze().cpu().numpy()
vol_orig = model.volume.squeeze().cpu().numpy()

# np.save(res_path / 'rotated_volume_2.npy', vol)
np.save(res_path / 'rotated_mask.npy', vol)

fig, ax = plt.subplots(2, 3, figsize=(15, 10), dpi=400)
ax[1, 0].imshow(vol[400, :, :])
ax[1, 1].imshow(vol[:, 400, :])
ax[1, 2].imshow(vol[:, :, 400])
ax[0, 0].imshow(vol_orig[400, :, :])
ax[0, 1].imshow(vol_orig[:, 400, :])
ax[0, 2].imshow(vol_orig[:, :, 400])

plt.show()
#%%




#%%
###############################################
# Generate projections for potential volume
model_path = res_path / 'full_res_new_masked.mat'
model = Model(device=device)
model.load(model_path)
model.plot_alignment(title='Alignment')
#%%
vol = np.load(base_path / 'analysis/reverse_rotation/sim_vol_orig.npy')
model.volume = th.as_tensor(vol, device=device).unsqueeze(0).unsqueeze(0)
#%%
model_sino = model.get_sino()

plotmosaic(model_sino, title='potential array', dpi=600)
np.save(base_path / 'analysis/reverse_rotation/sim_projections.npy', model_sino)
#%%
plotmosaic(model_sino, title='potential array', dpi=600)
plotmosaic(stack.images, title='exp', dpi=600)
#%%
model.theta[:] = 0
# model.psi[:] = 0
model.plot_alignment(title='Alignment')
model_sino_without_rot = model.get_sino()
plotmosaic(model_sino_without_rot, title='potential array without theta', dpi=600)
#%%
# model.theta[:] = 0
model.psi[:] = 0
model.plot_alignment(title='Alignment')
model_sino_without_rot = model.get_sino()
plotmosaic(model_sino_without_rot, title='potential array without psi', dpi=600)

#%%
plotmosaic(stack.images, title='exp', dpi=600)
#%%
np.save(base_path / 'sim_projections_orig.npy', model_sino_without_rot)
#%%



#%%
########################################################
# Reconstruction using theta from FFT, psi = 0
model_path = res_path / 'full_res_new_masked.mat'
model = Model(device=device)
model.load(model_path)
model.plot_alignment(title='Alignment')
#%%
theta = np.load(base_path/ 'stack_rotation_correction/thetas.npy')
plt.plot(model.theta.cpu() / th.pi * 180, marker='o', linestyle='-', label='from tomo')
plt.plot(theta, marker='o', linestyle='-', label='from fft fit')
plt.legend()
plt.show()
#%%
images_masked = np.load(base_path / 'stack_masked.npy')
# plotmosaic(images_denoised, title='', dpi=600)

stack_masked = DataStack(path_stack=base_path / fn_stack, path_angles=base_path / fn_angles, device=device)
stack_masked.images = images_masked

sino_target = stack_masked.to_sino_data()
plotmosaic(images_masked, title='', dpi=600)
sino_target.shape
# _______________________________________________________
#%%
# Initializing the model, loading predefined alignment
# load_model_path = res_path / 'upscaled' # load after upscaling
load_model_path = res_path / 'alignment/rotation_fit_2.mat'  # load after upscaling
model = Model(device=device)
model.load(load_model_path)
#_________________________________
# Change theta and psi
theta = np.load(base_path/ 'stack_rotation_correction/thetas.npy')
model.theta = th.as_tensor(theta/180 * th.pi, dtype=th.float32)
model.psi[:] = 0
#_________________________________
model.plot_alignment(title='Alignment')
model.reinit_volume(sino_shape=sino_target.shape)
model.print_shapes()
model.to_gpu()
sino_target.shape
#%%
np.save(base_path / 'angles.npy', model.phi.cpu().numpy())
#%%

#%%
th.cuda.empty_cache()
#%%
model.reinit_volume()
losses_all = []
#%%
radon = RadonReconstruction(
    model=model,
    sino_target=sino_target,
    reg_pars=None,
    vol_scheduler=None,
    lr_model=1e-4,
    save_path=base_path
)
radon.reg_edges = 5
kernel_size = 7
radon.init_gauss_blur_regularization(kernel_size, 1)
# fig, ax = plt.subplots()
# ax.plot(radon.gauss_kernel[0, kernel_size // 2, kernel_size // 2].cpu().numpy())
# plt.show()

for _ in range(3):
    vol, losses = radon.fit_volume(num_iterations=100, batch_size=1, progbar=True, print_logs=True, tau=1e-6)
    losses_all = np.append(losses_all, losses)
    plot_losses_model(losses, vol, losses_all, f'Scale 1')
    model.save(res_path / 'upscaled', "full_res_4.mat")
    # break

np.save(res_path / 'losses.npy', losses_all)
#%%
radon.penalize_tensor
#%%
torch.isnan(radon.penalize_tensor).any()
#%%
load_model_path = res_path / 'upscaled/full_res_new.mat' # load after upscaling
model = Model(device=device)
model.load(load_model_path)
#%%
model.volume.shape
#%%
sino_model = model.get_sino()
#%%
plotmosaic(sino_model, dpi=600)
#%%
sino_model.shape
#%%
