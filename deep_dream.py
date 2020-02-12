import torch
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from utils import preprocess, deprocess, clip


def dream(image, model, output=None, iterations=20, lr=0.01):
    """Updates the image to maximize outputs for n iterations"""
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
    image = Variable(tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        if output is None:
            loss = out.norm()
        else:
            loss = sum(output.pop().norm() for _ in output)
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep_dream(image, model, output=None, iterations=20, lr=0.01, octave_scale=1.4, num_octaves=10, skip_octaves=0):
    """Main deep dream method"""
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    # Generate scaled images for ech octave
    octaves = [image]
    for i in range(num_octaves - 1):
        if i >= skip_octaves:
            octaves.append(nd.zoom(octaves[-1], (1, 1, 1/octave_scale, 1/octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        if octave > 0:
            # Upsample to match octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape)/np.array(detail.shape), order=1)
        # Add detail to base image
        input_image = octave_base + detail
        # Run dream process
        dreamed_image = dream(input_image, model, output, iterations, lr)
        # Remove base image to obtain purely dreamed details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


def dream_progression(image, model, output=None, iterations=20, lr=0.01, octave_scale=1.4, num_octaves=10, skip_octaves=0):
    """Show progression at each iteration"""
    width, height = image.size
    images = np.zeros((iterations + 1, height, width, 3), dtype=np.uint8)
    images[0] = image

    for i in range(iterations):
        images[i+1] = deep_dream(images[i], model, output, 1, lr, octave_scale, num_octaves, skip_octaves)

    return images
