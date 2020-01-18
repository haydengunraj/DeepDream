import numpy as np
import torch
import torch.nn as nn
import scipy.fftpack
from torchvision import transforms, models
from PIL import Image, ImageDraw

from cnn.models import VGG19Finetune

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def flatten_module(module, modules=[], max_depth=-1):
    children = list(module.children())
    if max_depth and len(children):
        for child in children:
            flatten_module(child, modules, max_depth-1)
    else:
        modules.append(module)
    return modules


def truncate_model(model, layer, max_depth=-1):
    modules = flatten_module(model, max_depth=max_depth)
    return nn.Sequential(*modules[:layer+1])


def load_model(arch, layer=None, weights_path=None):
    max_depth = -1
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        if weights_path is None:
            model = models.vgg19(pretrained=True)
        else:
            model = VGG19Finetune(num_classes=365, weights_path=weights_path)
    elif arch == 'googlenet':
        model = models.googlenet(pretrained=True, transform_input=False)
        max_depth = 1
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    if layer is not None:
        model = truncate_model(model, layer, max_depth)
    return model


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = np.clip(image_np*255, 0, 255).astype(np.uint8)
    return image_np


def clip(image_tensor):
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor


def as_uint8(arr):
    return np.uint8(255*(arr - arr.min())/arr.ptp())


def normalize(arr):
    return (arr - arr.mean()) / arr.std()


def rgb_gaussian_random_field(size, alpha=4.0, eps=1e-10):
    k_idx = np.mgrid[:size, :size] - size//2
    k_idx = scipy.fftpack.fftshift(k_idx)
    amplitude = np.power(k_idx[0]*k_idx[0] + k_idx[1]*k_idx[1] + eps, -alpha/4.0)
    amplitude[0, 0] = 0
    gfield = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(3):
        noise = np.random.normal(size=(size, size)) + 1j * np.random.normal(size=(size, size))
        real = np.fft.ifft2(noise*amplitude).real
        gfield[..., i] = as_uint8(normalize(real))
    return gfield


def random_shapes(size, num_shapes=100, scales=6, factor=0.6, max_vertices=8, min_int=50):
    shape_size = size
    vertices = np.random.choice(list(range(3, max_vertices+1)), num_shapes)
    indices = list(range(0, size))
    centers = np.random.choice(indices, num_shapes*2)
    shapes_per_scale = num_shapes // scales
    intensities = list(range(min_int, 256))

    image = Image.new('RGB', (size, size), 0)
    drawer = ImageDraw.Draw(image)
    scale_count = 1
    for i, verts in enumerate(vertices):
        if not ((i + 1) % shapes_per_scale) and scale_count != scales:
            shape_size *= factor
            indices = indices[:int(shape_size)]
            scale_count += 1
        fill = tuple(np.random.choice(intensities, 3))
        xy = np.random.choice(indices, verts*2)
        center = centers[i*2:(i + 1)*2] - int(shape_size/2)
        xy[::2] += center[0]
        xy[1::2] += center[1]
        drawer.polygon(list(xy), outline=fill, fill=fill)
    return image
