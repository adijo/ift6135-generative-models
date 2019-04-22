#!/usr/bin/env python
import argparse
#TODO: Implement a GAN
import scipy
import time
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils
import torch.nn.functional as F

import wgan_gp



from torch.utils.data import dataset
from torch import nn, autograd
# from torch.nn.modules import upsampling
# from torch.functional import F
from torch.optim import Adam

def generate_pictures():
    gp_scaling = 10 #Lambda in the paper
    parser = argparse.ArgumentParser(
        description='Run a wgan-gp with parameters')
    parser.add_argument('--generator_pt', type=str, default="generatordefault74.pt",
                        help='Generator parameters file')
    parser.add_argument('--n_pictures', type=int, default=1000, help='Number of pictures to generate')

    parser.add_argument('--out_dir', type=str, default="samples", help='Number of pictures to generate')

    args = parser.parse_args()

    generator = wgan_gp.Generator()
    generator.load_state_dict(torch.load(args.generator_pt))

    #cuda = torch.cuda.is_available()
    cuda = False
    for i in range(0,int(args.n_pictures/64)):
        fixed_z = torch.FloatTensor(args.n_pictures,100,1,1).normal_(0,1) #Used to compare pictures from epoch to epoch
        if cuda:
            generator=generator.cuda()
            fixed_z = fixed_z.cuda()
        fake_pictures = generator(fixed_z)

        for j in range(64):
            torchvision.utils.save_image(fake_pictures[j].detach().cpu(), args.out_dir + '/wgan_gp'+str(i*64 + j)+'.png', normalize=True)


if __name__ == "__main__":
    generate_pictures()
