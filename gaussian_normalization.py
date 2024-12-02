import torch
import torch as th
import numpy as np
from torch.nn.functional import mse_loss
from torch.optim import Adam
#from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import matplotlib.pyplot as plt

import os
import sys


def axis_gaussian_normalization(stack: torch.Tensor, show_res=True):

    d1 = th.as_tensor(stack).clone()
    d1[d1<0] = 0
    sh = d1.shape
    refind = sh[0] // 2
    ref = th.sum(d1[refind], 1).float()

    fac = th.ones((d1.shape[0]), requires_grad=True)
    last_loss = 0
    opt = Adam([fac], lr=1e-2)

    ds = th.as_tensor(gaussian_filter1d(th.sum(d1, 2).numpy(),5)).float()

    if show_res:
        fig, ax = plt.subplots(figsize=(16,4))
        for dd in ds:
            ax.plot(dd[:])
        ax.plot(ds[refind, :], linewidth=1)
        plt.show()

    #ds.shape

    mm = 1
    ds = ds[:, mm:-mm]
    ref = ref[mm:-mm]

    for i in range(400):
        opt.zero_grad()

        s = ds * fac.unsqueeze(1).expand_as(ds)

        loss = mse_loss(s, ref)

        # f, a = plt.subplots(figsize=(15, 12))
        # for si in s:
        #     a.plot(np.arange(len(si)), si.detach().numpy())
        # a.plot(np.arange(len(si)), ref.numpy(), linewidth=5)
        # plt.show()

        loss.backward()

        if show_res:
            print(f"i: {i} L = {loss.item():3.6g} dL = {last_loss - loss.item():3.3g}")

        opt.step()
        # print(fac)

        last_loss = loss.item()
        # fac[j] = fac2.detach().item()

    if show_res:
        print(fac)

    if show_res:
        f, a = plt.subplots(figsize=(15, 12))
        s = th.sum(d1, 2).float()
        s *= fac.unsqueeze(1).expand_as(s)
        ref = th.sum(d1[refind], 1).float()
        for si in s:
            a.plot(np.arange(len(si)), si.detach().numpy())
        a.plot(np.arange(ref.shape[0]), ref.numpy(), linewidth=5)
        plt.show()

        f, a = plt.subplots(figsize=(15, 12))
        a.scatter(np.arange(len(fac)), fac.detach().numpy(), linewidth=5)
        plt.show()

    d2_corrected = th.as_tensor(stack) * fac.unsqueeze(1).unsqueeze(1).expand_as(d1)
    d2_corrected = d2_corrected.detach().numpy()
    return d2_corrected