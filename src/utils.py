#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torchmetrics.image import PeakSignalNoiseRatio

import os
import glob
import numpy as np
from math import log10
from datetime import datetime
import OpenEXR
from PIL import Image
import Imath

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.io as sio


def clear_line():
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')


def progress_bar(batch_idx, num_batches, report_interval, train_loss):
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 21 + dec
    progress = (batch_idx % report_interval) / report_interval
    fill = int(progress * bar_size) + 1
    print('\rBatch {:>{dec}d} [{}{}] Train loss: {:>1.5f}'.format(batch_idx + 1, '=' * fill + '>', ' ' * (bar_size - fill), train_loss, dec=str(dec)), end='')


def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms


def show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr):
    """Formats validation error stats."""

    clear_line()
    print('Train time: {} | Valid time: {} | Valid loss: {:>1.5f} | Avg PSNR: {:.2f} dB'.format(epoch_time, valid_time, valid_loss, valid_psnr))


def show_on_report(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    clear_line()
    dec = int(np.ceil(np.log10(num_batches)))
    print('Batch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(batch_idx + 1, num_batches, loss, int(elapsed), dec=dec))


def plot_per_epoch(ckpt_dir, title, measurements, y_label):
    """Plots stats (train/valid loss, avg PSNR, etc.)."""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()


def load_hdr_as_tensor(img_path):
    """Converts OpenEXR image to torch float tensor."""

    # Read OpenEXR file
    if not OpenEXR.isOpenExrFile(img_path):
        raise ValueError(f'Image {img_path} is not a valid OpenEXR file')
    src = OpenEXR.InputFile(img_path)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = src.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    # Read into tensor
    tensor = torch.zeros((3, size[1], size[0]))
    for i, c in enumerate('RGB'):
        rgb32f = np.fromstring(src.channel(c, pixel_type), dtype=np.float32)
        tensor[i, :, :] = torch.from_numpy(rgb32f.reshape(size[1], size[0]))
        
    return tensor


def reinhard_tonemap(tensor):
    """Reinhard et al. (2002) tone mapping."""

    tensor[tensor < 0] = 0
    return torch.pow(tensor / (1 + tensor), 1 / 2.2)


def psnr(input, target, data_range=None):
    """Computes peak signal-to-noise ratio."""
    
    #return 10 * torch.log10(1 / F.mse_loss(input, target))

    if torch.is_tensor(input):
        data_in = input
    else:
        data_in = torch.tensor(input)

    if torch.is_tensor(target):
        data_out = target
    else:
        data_out = torch.tensor(target)

    if data_range == None:
        psnr_torch = PeakSignalNoiseRatio()
    else:
        psnr_torch = PeakSignalNoiseRatio(data_range=data_range)

    return psnr_torch(data_in, data_out)


def create_montage(img_name, noise_type, save_path, source_t, denoised_t, clean_t, show):
    """Creates montage for easy comparison."""

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    #fig.canvas.set_window_title(img_name.capitalize()[:-4])
    fig.canvas.manager.set_window_title(img_name.capitalize()[:-4])

    # Bring tensors to CPU
    source_t = source_t.cpu().narrow(0, 0, 3)
    denoised_t = denoised_t.cpu()
    clean_t = clean_t.cpu()
    
    source = tvF.to_pil_image(source_t)
    denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
    clean = tvF.to_pil_image(clean_t)

    # Build image montage
    psnr_vals = [psnr(source_t, clean_t), psnr(denoised_t, clean_t)]
    titles = ['Input: {:.2f} dB'.format(psnr_vals[0]),
              'Denoised: {:.2f} dB'.format(psnr_vals[1]),
              'Ground truth']
    zipped = zip(titles, [source, denoised, clean])
    for j, (title, img) in enumerate(zipped):
        ax[j].imshow(img)
        ax[j].set_title(title)
        ax[j].axis('off')

    # Open pop up window, if requested
    if show > 0:
        plt.show()

    # Save to files
    fname = os.path.splitext(img_name)[0]
    source.save(os.path.join(save_path, f'{fname}-{noise_type}-noisy.png'))
    denoised.save(os.path.join(save_path, f'{fname}-{noise_type}-denoised.png'))
    fig.savefig(os.path.join(save_path, f'{fname}-{noise_type}-montage.png'), bbox_inches='tight')

def save_hsi(img_name, noise_type, save_path, source_t, denoised_t, clean_t, max_val_train=255.0):
    # Bring tensors to CPU
    source_t = source_t.cpu().numpy()
    denoised_t = denoised_t.cpu().numpy()
    clean_t = clean_t.cpu().numpy()

    mdic_source = {"data": source_t}
    mdic_denoised = {"data": denoised_t}

    psnr_source = psnr(source_t, clean_t, max_val_train).item()
    psnr_source = "{:.2f}".format(psnr_source)

    psnr_denoised = psnr(denoised_t, clean_t, max_val_train).item()
    psnr_denoised = "{:.2f}".format(psnr_denoised)

    fname = os.path.splitext(img_name)[0]
    sio.savemat(os.path.join(save_path, f'{fname}-{noise_type}-psnr-{psnr_source}-noisy.mat'), mdic_source)
    sio.savemat(os.path.join(save_path, f'{fname}-{noise_type}-psnr-{psnr_denoised}-denoised.mat'), mdic_denoised)


class AvgMeter(object):
    """Computes and stores the average and current value.
    Useful for tracking averages such as elapsed times, minibatch losses, etc.
    """

    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_hsi_as_tensor(path, matContentHeader="data"):
    mat = sio.loadmat(path)
    mat = mat[matContentHeader]

    assert isinstance(mat, np.ndarray) and mat.dtype != np.uint8
    return tvF.to_tensor(mat)


def normalize(img, norm_type="max", max_val=None, min_val=None):
    if "max" == norm_type:
        return img / max_val * 255.0
    
    elif "min_max" == norm_type:
        return ((img - min_val) / (max_val - min_val)) * 255
    
    elif "max_single_img" == norm_type:
        return img / torch.max(img) * 255.0
    
    elif "min_max_single_img" == norm_type:
        return ((img - torch.min(img)) / (torch.max(img) - torch.min(img))) * 255
    
    else:
        raise NotImplementedError("This normalization is not implemented")


def inverse_normalize(img, norm_type="max", max_val=None, min_val=None):
    if "max" == norm_type:
        return img * max_val / 255.0
    
    elif "min_max" == norm_type:
        return img * (max_val - min_val) / 255.0 + min_val
    
    elif "max_single_img" == norm_type: #TODO dogru degil
        return img / torch.max(img) * 255.0
    
    elif "min_max_single_img" == norm_type: #TODO dogru degil
        return img * (torch.max(img) - torch.min(img)) / 255.0 + torch.min(img)
    
    else:
        raise NotImplementedError("This normalization is not implemented")


def inverse_normalizessss(img, max_val):
    return img / 255.0 * max_val


def normalize_hsi(hsi, max_val=255):
    curr_max = np.amax(hsi)
    return hsi / curr_max * max_val


def get_max_min(dir, ext="mat"):
    search_pattern = os.path.join(dir, f"*.{ext}")
    files = glob.glob(search_pattern)

    global_max = -float('inf')
    global_min = float('inf')

    for file in files:
        img = load_hsi_as_tensor(file)
        if torch.max(img) > global_max:
            global_max = torch.max(img)
        if torch.min(img) < global_min:
            global_min = torch.min(img)
    
    print("Maximum value among all elements:", global_max.item())
    print("Minimum value among all elements:", global_min.item())

    return global_max, global_min