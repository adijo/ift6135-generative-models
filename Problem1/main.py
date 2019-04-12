import numpy as np
import scipy.stats
import scipy as sp
import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable

from Problem1 import networks

# https://wiseodd.github.io/techblog/2017/01/20/gan-pytorch/
# https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = networks.Discriminator("""input parameters""").to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for it in range(100000):
        # Sample data
        z = Variable(torch.randn(mb_size, Z_dim))
        X, _ = mnist.train.next_batch(mb_size)
        X = Variable(torch.from_numpy(X))

        # Dicriminator forward-loss-backward-update
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
        D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake

        D_loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
