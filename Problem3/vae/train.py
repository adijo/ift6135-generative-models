from torch import optim
from torch.autograd import Variable
import torch

from Problem3.vae.svhn import get_data_loader
from Problem3.vae.vae import VAE, loss_fn


def train(model, optimizer, train_loader, loss_function, epoch, use_cuda=True, log_interval=10):
    # toggle model to train mode
    model.train()
    train_loss = 0
    # in the case of MNIST, len(train_loader.dataset) is 60000
    # each `data` is of BATCH_SIZE samples and has shape [128, 1, 28, 28]
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if use_cuda:
            data = data.cuda()
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, log_var = model(data)
        # calculate scalar loss
        loss = loss_function(recon_batch, data, mu, log_var)

        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


Z_DIMS = 100
NUM_EPOCHS = 1
BATCH_SIZE=64
use_cuda = torch.cuda.is_available()
model = VAE()
if use_cuda:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_loader, valid_loader, test_loader = get_data_loader("svhn", BATCH_SIZE)

for epoch in range(NUM_EPOCHS):
    train(model, optimizer, train_loader, loss_fn, epoch, use_cuda=use_cuda)
