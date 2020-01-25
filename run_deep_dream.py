import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os
from PIL import Image

from utils import load_model
from deep_dream import deep_dream, dream_progression
from utils import register_hook, rgb_gaussian_random_field, random_shapes, IMAGENET_NUM_CLASSES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='vgg19', help='network architecture')
    parser.add_argument('--image', type=str, default=None, help='path to input image')
    parser.add_argument('--size', type=int, default=256, help='square size of random noise image')
    parser.add_argument('--weights', type=str, default=None, help='path to pretrained weights')
    parser.add_argument('--num_classes', type=int, default=None, help='number of classes in pretrained model')
    parser.add_argument('--iterations', default=20, type=int, help='number of gradient ascent steps per octave')
    parser.add_argument('--save', action='store_true', help='flag for saving the image')
    parser.add_argument('--save_all', action='store_true', help='flag for saving all iterations')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--octave_scale', type=float, default=1.4, help='image scale between octaves')
    parser.add_argument('--num_octaves', type=int, default=10, help='number of octaves')
    parser.add_argument('--skip_octaves', type=int, default=0, help='number of initial octaves to skip')
    parser.add_argument('--max_size', type=int, default=2000, help='maximum image dimension')
    parser.add_argument('--noise_type', type=str, default='gauss', help='type of random noise to use')
    parser.add_argument('--no_plot', action='store_true', help='flag for suppressing plot')
    args = parser.parse_args()

    # Check for weights + num_classes
    if args.weights is not None and args.num_classes is None:
        parser.error('--weights requires --num_classes')

    # Set torch home dir
    os.environ['TORCH_HOME'] = 'weights'

    # Load image
    if args.image is None:
        if args.noise_type == 'white':
            image = np.random.randint(0, 255, (args.size, args.size, 3)).astype(np.uint8)
        elif args.noise_type == 'gauss':
            image = rgb_gaussian_random_field(args.size)
        elif args.noise_type == 'shapes':
            image = random_shapes(args.size)
            image = np.asarray(image)
        else:
            raise ValueError('Undefined noise type - {}'.format(args.noise_type))
    else:
        image = Image.open(args.image)
        image = np.array(image)

    # Resize if necessary
    max_dim = max(image.shape[:2])
    if max_dim > args.max_size:
        orig_shape = image.shape[-2::-1]
        scale = args.max_size/max_dim
        new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
        image = np.array(Image.fromarray(image).resize(new_size))
        print('Resizing image from {} to {}'.format(orig_shape, new_size))

    # Define the model
    num_classes = args.num_classes if args.weights is not None else IMAGENET_NUM_CLASSES
    model, layer_dict, layer_list = load_model(args.arch, num_classes=num_classes, weights_path=args.weights)
    if torch.cuda.is_available:
        model = model.cuda()

    # Select the output layer
    n_layers = len(layer_list)
    for i, layer_str in enumerate(layer_list):
        print(str(i).rjust(len(str(n_layers))) + '\t' + layer_str)
    idx = input('\nSelect a layer by index: ')
    if idx:
        idx = int(idx)
        if idx < 0 or idx >= n_layers:
            raise ValueError('Layer index must be an integer in the range [0, {}]'.format(n_layers - 1))
        desired_output, _ = register_hook(layer_dict[layer_list[idx].strip().split()[0]])
    else:
        desired_output = None

    # Extract deep dream image
    output_dir = 'outputs'
    if args.save_all:
        dreamed_images = dream_progression(
            image,
            model,
            output=desired_output,
            iterations=args.iterations,
            lr=args.lr,
            octave_scale=args.octave_scale,
            num_octaves=args.num_octaves,
            skip_octaves=args.skip_octaves
        )
        dreamed_image = dreamed_images[-1]
        if args.image is None:
            output_dir = os.path.join(output_dir, 'output_layer{}_iter{}_'.format(
                args.at_layer, args.iterations) + datetime.now().strftime('%Y-%m-%d_%H.%M.%S'))
        else:
            output_dir = os.path.join(output_dir, 'output_layer{}_iter{}_{}'.format(
                args.at_layer, args.iterations, os.path.splitext(os.path.basename(args.image))[0]))
        os.makedirs(output_dir, exist_ok=True)
        for i in range(args.iterations+1):
            plt.imsave(os.path.join(output_dir, 'step{}.jpg'.format(i)), dreamed_images[i])
    else:
        dreamed_image = deep_dream(
            image,
            model,
            output=desired_output,
            iterations=args.iterations,
            lr=args.lr,
            octave_scale=args.octave_scale,
            num_octaves=args.num_octaves,
            skip_octaves=args.skip_octaves
        )
        if args.save:
            if args.image is None:
                filename = os.path.join(output_dir, 'output_layer{}_iter{}_'.format(
                    args.at_layer, args.iterations) + datetime.now().strftime('%Y-%m-%d_%H.%M.%S.jpg'))
            else:
                filename = os.path.join(output_dir, 'output_layer{}_iter{}_{}'.format(
                    args.at_layer, args.iterations, os.path.basename(args.image)))
            os.makedirs(output_dir, exist_ok=True)
            plt.imsave(filename, dreamed_image)

    # Plot image
    if not args.no_plot:
        plt.figure()
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(dreamed_image)
        plt.show()
