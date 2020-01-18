import torch
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from utils import preprocess, deprocess, clip


def dream(image, model, iterations, lr):
    ''' Updates the image to maximize outputs for n iterations '''
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves, skip_octaves=0):
    ''' Main deep dream method '''
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    # Extract image representations for each octave
    octaves = [image]
    for i in range(num_octaves - 1):
        if i >= skip_octaves:
            octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


def dream_progression(image, model, iterations, lr, octave_scale, num_octaves, skip_octaves=0):
    ''' Show progression at each iteration '''
    width, height = image.size
    images = np.zeros((iterations + 1, height, width, 3), dtype=np.uint8)
    images[0] = image

    for i in range(iterations):
        images[i+1] = deep_dream(images[i], model, 1, lr, octave_scale, num_octaves, skip_octaves)

    return images
