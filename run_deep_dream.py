import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import load_model
from datetime import datetime
import os
from PIL import Image

from deep_dream import deep_dream, dream_progression
from utils import rgb_gaussian_random_field, random_shapes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='vgg19', help='network architecture')
    parser.add_argument('--input_image', type=str, default=None, help='path to input image')
    parser.add_argument('--size', type=int, default=256, help='square size of random noise image')
    parser.add_argument('--max_size', type=int, default=2000, help='maximum image dimension')
    parser.add_argument('--noise_type', type=str, default='gauss', help='type of random noise to use')
    parser.add_argument('--iterations', default=20, type=int, help='number of gradient ascent steps per octave')
    parser.add_argument('--at_layer', default=27, type=int, help='layer at which we modify image to maximize outputs')
    parser.add_argument('--no_plot', action='store_true', help='flag for suppressing plot')
    parser.add_argument('--save', action='store_true', help='flag for saving the image')
    parser.add_argument('--save_all', action='store_true', help='flag for saving all iterations')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--octave_scale', type=float, default=1.4, help='image scale between octaves')
    parser.add_argument('--num_octaves', type=int, default=10, help='number of octaves')
    parser.add_argument('--skip_octaves', type=int, default=0, help='number of initial octaves to skip')
    parser.add_argument('--weights', type=str, default=None, help='path to pretrained weights')
    args = parser.parse_args()

    # Set torch home dir
    os.environ['TORCH_HOME'] = 'weights'

    # Load image
    if args.input_image is None:
        if args.noise_type == 'white':
            image_arr = np.random.randint(0, 255, (args.size, args.size, 3)).astype(np.uint8)
        elif args.noise_type == 'gauss':
            image_arr = rgb_gaussian_random_field(args.size)
        elif args.noise_type == 'shapes':
            image = random_shapes(args.size)
            image_arr = np.asarray(image)
        else:
            raise ValueError('Undefined noise type - {}'.format(args.noise_type))
        image = Image.fromarray(image_arr)
    else:
        image = Image.open(args.input_image)
        image_arr = np.array(image)

    # Resize if necessary
    max_dim = max(image_arr.shape[:2])
    if max_dim > args.max_size:
        scale = args.max_size/max_dim
        new_size = (int(image_arr.shape[1]*scale), int(image_arr.shape[0]*scale))
        image = image.resize(new_size)
        print('Resizing image from {} to {}'.format(image_arr.shape[-2::-1], new_size))

    # Define the model
    model = load_model(args.arch, args.at_layer, args.weights)
    print(model)

    if torch.cuda.is_available:
        model = model.cuda()

    # Extract deep dream image
    output_dir = 'outputs'
    if args.save_all:
        dreamed_images = dream_progression(
            image,
            model,
            iterations=args.iterations,
            lr=args.lr,
            octave_scale=args.octave_scale,
            num_octaves=args.num_octaves,
            skip_octaves=args.skip_octaves
        )
        dreamed_image = dreamed_images[-1]
        if args.input_image is None:
            output_dir = os.path.join(output_dir, 'output_layer{}_iter{}_'.format(
                args.at_layer, args.iterations) + datetime.now().strftime('%Y-%m-%d_%H.%M.%S'))
        else:
            output_dir = os.path.join(output_dir, 'output_layer{}_iter{}_{}'.format(
                args.at_layer, args.iterations, os.path.splitext(args.input_image.split('/')[-1])[0]))
        os.makedirs(output_dir, exist_ok=True)
        for i in range(args.iterations+1):
            plt.imsave(os.path.join(output_dir, 'step{}.jpg'.format(i)), dreamed_images[i])
    else:
        dreamed_image = deep_dream(
            image,
            model,
            iterations=args.iterations,
            lr=args.lr,
            octave_scale=args.octave_scale,
            num_octaves=args.num_octaves,
            skip_octaves=args.skip_octaves
        )
        if args.save:
            if args.input_image is None:
                filename = os.path.join(output_dir, 'output_layer{}_iter{}_'.format(
                    args.at_layer, args.iterations) + datetime.now().strftime('%Y-%m-%d_%H.%M.%S.jpg'))
            else:
                filename = os.path.join(output_dir, 'output_layer{}_iter{}_{}'.format(
                    args.at_layer, args.iterations, args.input_image.split('/')[-1]))
            os.makedirs(output_dir, exist_ok=True)
            plt.imsave(filename, dreamed_image)

    # Plot image
    if not args.no_plot:
        plt.figure()
        plt.subplot(121)
        plt.imshow(image_arr)
        plt.subplot(122)
        plt.imshow(dreamed_image)
        plt.show()
