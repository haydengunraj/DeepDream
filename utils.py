import numpy as np
import torch
import scipy.fftpack
from torchvision import transforms, models
from PIL import Image, ImageDraw

IMAGENET_NUM_CLASSES = 1000

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def register_hook(layer, layer_output_list):
    """Register a forward hook for a module"""
    def _hook(module, inputs, output):
        layer_output_list.append(output)
    handle = layer.register_forward_hook(_hook)
    return handle


def load_model(arch, num_classes=1000, weights_path=None):
    """Load a model and generate its layer dict"""
    if weights_path is None:
        num_classes = IMAGENET_NUM_CLASSES
        pretrained = True
    else:
        pretrained = False
    model = getattr(models, arch)(num_classes=num_classes, pretrained=pretrained)
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    model.eval()
    layer_dict, layer_list = get_layer_dict(model)
    return model, layer_dict, layer_list


def get_layer_dict(model, indent='  '):
    """Create a name: module dictionary and generate a list of strings for display"""
    layer_dict = {}
    layer_list = []
    for name, module in model.named_modules():
        if name:
            layer_dict[name] = module
            layer_list.append(indent*name.count('.') + name + ' ' + module.__class__.__name__)
    return layer_dict, layer_list


preprocess = transforms.Compose([
    # Default preprocessing for PyTorch models
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


def deprocess(image):
    """Convert back to uint8 image domain"""
    image = image.squeeze().transpose(1, 2, 0)
    image = image * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def clip(tensor):
    """Clipping for model outputs"""
    for c in range(3):
        m, s = mean[c], std[c]
        tensor[0, c] = torch.clamp(tensor[0, c], -m / s, (1 - m) / s)
    return tensor


def tile(image, size):
    """Divide an image into tiles"""
    nr = int(np.ceil(image.shape[0] / size))
    nc = int(np.ceil(image.shape[1] / size))
    h = int(np.ceil(image.shape[0] / nr))
    w = int(np.ceil(image.shape[1] / nc))
    tiles = []
    for r in range(nr):
        tiles.append([])
        for c in range(nc):
            crop = image[r * h:(r + 1) * h, c * w:(c + 1) * w, :]
            tiles[-1].append(crop)
    return tiles


def untile(images):
    """Reconstruct an image from tiles"""
    return np.concatenate([np.concatenate(row, axis=1) for row in images], axis=0)


def as_uint8(arr):
    """Convert an array to full uint8 range"""
    return np.uint8(255*(arr - arr.min())/arr.ptp())


def normalize(arr):
    """Mean subtraction and std normalization"""
    return (arr - arr.mean()) / arr.std()


def rgb_gaussian_random_field(size, alpha=4.0, eps=1e-10):
    """Generate a random gaussian field image in RGB space"""
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
    """Generate an image of random polygons in RGB space"""
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
