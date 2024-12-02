import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import torch
import numpy as np
import torch as th
import os

from fastatomography.util import plotmosaic, mosaic

from copy import deepcopy as copy
from pathlib import Path
from utils_andrey import skip_steps, plot_losses_model, plot_angles, DataStack, Model, RadonReconstruction, print_memory_usage

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

#%%
# Loading data
stack = DataStack(path_stack=base_path / fn_stack, path_angles=base_path / fn_angles, device=device)

# Making rearrangement and cropping
stack.preprocessing()
plotmosaic(stack.images, title='Stack cropped', dpi=600)

# Transposing data
stack.images = np.transpose(stack.images, (0, 2, 1))
plotmosaic(stack.images, title='Stack transposed', dpi=600)

# Correcting across rotation
stack.gauss_correction()
plotmosaic(stack.images, title='Stack gauss corrected', dpi=600)
#%%
####################################################################
# Denoising stack
images_denoised = np.load(base_path / 'stack_36_projections_denoise_bm3d.npy')
plotmosaic(images_denoised, title='Stack denoised gauss corrected', dpi=600)
#%%
stack_denoised = copy(stack)
stack_denoised.images = images_denoised
# Correcting across rotation
stack_denoised.gauss_correction()
plotmosaic(stack_denoised.images, title='Stack denoised gauss corrected', dpi=600)
#%%
np.save(base_path / 'stack_36_projections_denoise_bm3d_gauss_corr.npy', stack_denoised.images)
#%%
stack = stack_denoised
plotmosaic(stack.images[[5, 10, 16, 22, 28],:,:], title='Stack gauss corrected', dpi=600, cmap='Greys')
#%%
stack_cropped = stack
images = stack_cropped.images[[5, 10, 16, 22, 28],:,:]
images.shape
#%%


def do_plot(ax, Z, transform, x_offset, y_offset, squeeze_r=0, squeeze_l=0):
    im = ax.imshow(Z, interpolation='nearest', cmap='Greys', vmin=0, vmax=1,
                   origin='lower',
                   extent=[x_offset + squeeze_r, x_offset + 1 - squeeze_l, y_offset, y_offset + 1],
                   clip_on=True)

    trans_data = transform + ax.transData
    im.set_transform(trans_data)
    x_right_down_trans, y_right_down_trans = transform.transform_point((x_offset + 1, y_offset))
    # x_right_up_trans, y_right_up_trans = transform.transform_point((x_offset + 1, y_offset + 1))
    # print(f"x_right_down: {x_right_down_trans}, y_right_down: {y_right_down_trans}")
    # print(f"x_right_up: {x_right_up_trans}, y_right_up: {y_right_up_trans}")

    ax.set_xlim(0, 1.0038198375433474)  # Adjust the x-axis limits as needed
    ax.set_ylim(0, 1.55)  # Adjust the y-axis limits as needed
    return x_right_down_trans, y_right_down_trans

# prepare image and figure
fig, axes = plt.subplots(1, 5, figsize=(16, 5), dpi=400)

k_sqz = 0.15
x_of, y_of = do_plot(axes[0], images[0], Affine2D().skew_deg(0, 15), 0, 0, squeeze_r=k_sqz)
x_of, y_of = do_plot(axes[1], images[1], Affine2D().skew_deg(0, 5), 0, y_of)
x_of, y_of = do_plot(axes[2], images[2], Affine2D().rotate_deg(0), 0, y_of)
x_of, y_of = do_plot(axes[3], images[3], Affine2D().skew_deg(0, -5), 0, y_of)
x_of, y_of = do_plot(axes[4], images[4], Affine2D().skew_deg(0, -15), 0, y_of, squeeze_l=k_sqz)

plt.tight_layout()

for ax in axes:
    ax.axis('off')
plt.subplots_adjust(wspace=-0.16)
plt.tight_layout()

# Plot name
plot_name = 'reconstructed_profiles.png'

plt.savefig(base_path / 'figures/' / plot_name, transparent=True)
plt.show()

#%%






















#%%
########################################################################
# Raw data

import numpy as np
from h5py import File
import matplotlib.pyplot as plt

from pathlib import Path

import zarr


data_folder = base_path / 'Co3O4cube/data_nanocube/'


scan = 35
name = f'{scan:03d}.zip'
df = data_folder / name

slice = np.s_[50:-50, 50:-50, ...]

store2 = zarr.open(str(df), mode='r')
data = store2['/data'][:,:,:,:].astype(np.float32)#[slice]
ds = np.array(data.shape)

meta_dict = store2['/meta'][0]
#%%
fig, ax = plt.subplots()
ax.imshow(data[10, 22])
plt.show()
#%%
store2['/meta'][0]
#%%
slice1 = np.s_[330:360:10, 330:360:10]

data_sub = data[slice1]
data_reshaped = data_sub.reshape(-1, 76, 76)
print(data_reshaped.shape)

plotmosaic(data_reshaped)
#%%
# Read raw data
scans = [16, 22, 28, 34, 40]
slice1 = np.s_[330:360:10, 330:360:10]
data_all = []

for scan in scans:
    name = f'{scan:03d}.zip'
    df = data_folder / name

    store = zarr.open(str(df), mode='r')
    data = store['/data'][:, :, :, :].astype(np.float32)  # [slice]
    data_sub = data[slice1]
    print(f'data shape: {data.shape} data_sub shape: {data_sub.shape}')
    data_reshaped = data_sub.reshape(-1, 76, 76)
    data_all.append(data_reshaped)

data_all = np.array(data_all)
#%%
fig, ax = plt.subplots()
ax.imshow(mosaic(data_all[0]))
plt.show()
#%%
images =[]
for i in range(len(data_all)):
    image = mosaic(data_all[i])
    images.append(image)
images = np.array(images)
images.shape
#%%
images = images / images.max()
#%%

def do_plot(ax, Z, transform, x_offset, y_offset, squeeze_r=0, squeeze_l=0):
    im = ax.imshow(Z, interpolation='nearest', vmin=0, vmax=1,
                   origin='lower',
                   extent=[x_offset + squeeze_r, x_offset + 1 - squeeze_l, y_offset, y_offset + 1],
                   clip_on=True)

    trans_data = transform + ax.transData
    im.set_transform(trans_data)
    x_right_down_trans, y_right_down_trans = transform.transform_point((x_offset + 1, y_offset))

    ax.set_xlim(0, 1.0038198375433474)  # Adjust the x-axis limits as needed
    ax.set_ylim(0, 1.55)  # Adjust the y-axis limits as needed
    return x_right_down_trans, y_right_down_trans

# prepare image and figure
fig, axes = plt.subplots(1, 5, figsize=(16, 5), dpi=400)

k_sqz = 0.15
x_of, y_of = do_plot(axes[0], images[0], Affine2D().skew_deg(0, 15), 0, 0, squeeze_r=k_sqz)
x_of, y_of = do_plot(axes[1], images[1], Affine2D().skew_deg(0, 5), 0, y_of)
x_of, y_of = do_plot(axes[2], images[2], Affine2D().rotate_deg(0), 0, y_of)
x_of, y_of = do_plot(axes[3], images[3], Affine2D().skew_deg(0, -5), 0, y_of)
x_of, y_of = do_plot(axes[4], images[4], Affine2D().skew_deg(0, -15), 0, y_of, squeeze_l=k_sqz)

plt.tight_layout()

for ax in axes:
    ax.axis('off')
plt.subplots_adjust(wspace=-0.16)
plt.tight_layout()

# Plot name
plot_name = 'diffraction_profiles.png'

plt.savefig(base_path / 'figures/' / plot_name, transparent=True)
plt.show()
#%%





#%%
########################################################################
# probes scheme
import numpy as np
from PIL import Image, ImageDraw

# Image dimensions
width, height = 1000, 1000

# Create a white background image
image = Image.new('RGB', (width, height), color='white')

# Create a drawing context
draw = ImageDraw.Draw(image)

# Define circles parameters
line_thickness1 = 20
number_of_circles = 3

# Calculate centers
def calc_cntres(width, height, number_of_circles):
    radius = width / (number_of_circles + 1)
    x = np.linspace(0, 1, number_of_circles + 2)[1:-1] * width
    y = np.linspace(0, 1, number_of_circles + 2)[1:-1] * height
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    return radius, list(zip(xv, yv))

radius, centers = calc_cntres(width, height, number_of_circles)

# Draw circles on the image
for x_c, y_c in centers:
    draw.ellipse([(x_c-radius, y_c-radius), (x_c+radius, y_c+radius)], outline='orange', width=line_thickness1)

# Convert the image to a NumPy array
image_array = np.array(image)

fig, ax = plt.subplots()
ax.imshow(image_array)
plt.show()
#%%

def do_plot(ax, Z, transform, x_offset, y_offset, squeeze_r=0, squeeze_l=0):
    im = ax.imshow(Z, interpolation='nearest', vmin=0, vmax=1,
                   origin='lower',
                   extent=[x_offset + squeeze_r, x_offset + 1 - squeeze_l, y_offset, y_offset + 1],
                   clip_on=True)

    trans_data = transform + ax.transData
    im.set_transform(trans_data)
    x_right_down_trans, y_right_down_trans = transform.transform_point((x_offset + 1, y_offset))

    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='tab:blue', linewidth=5, transform=trans_data)

    ax.set_xlim(0, 1.0038198375433474)  # Adjust the x-axis limits as needed
    ax.set_ylim(-0.4, 1)  # Adjust the y-axis limits as needed
    return x_right_down_trans, y_right_down_trans

# prepare image and figure
fig, axes = plt.subplots(1, 5, figsize=(16, 5), dpi=400)

k_sqz = 0.15
x_of, y_of = do_plot(axes[0], image_array, Affine2D().skew_deg(0, -15), 0, 0, squeeze_r=k_sqz)
x_of, y_of = do_plot(axes[1], image_array, Affine2D().skew_deg(0, -5), 0, y_of)
x_of, y_of = do_plot(axes[2], image_array, Affine2D().rotate_deg(0), 0, y_of)
x_of, y_of = do_plot(axes[3], image_array, Affine2D().skew_deg(0, 5), 0, y_of)
x_of, y_of = do_plot(axes[4], image_array, Affine2D().skew_deg(0, 15), 0, y_of, squeeze_l=k_sqz)

plt.tight_layout()

for ax in axes:
    ax.axis('off')
plt.subplots_adjust(wspace=-0.14)
plt.tight_layout()

# Plot name
plot_name = 'probe_scheme.png'

plt.savefig(base_path / 'figures/' / plot_name, transparent=True)
plt.show()