import torch
from torch import autograd
import matplotlib.pyplot as plt


def get_gradient_penalty(model, real_data, fake_data, use_cuda, batch_size, lambda_constant):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(0) if use_cuda else alpha

    zeta = alpha*real_data + (1-alpha)*fake_data

    if use_cuda:
        zeta = zeta.cuda(0)
    zeta = autograd.Variable(zeta, requires_grad=True)

    predictions = model(zeta)

    gradients = autograd.grad(
        outputs=predictions,
        inputs=zeta,
        grad_outputs=torch.ones(predictions.size()).cuda(0) if use_cuda else torch.ones(predictions.size()),
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]

    return lambda_constant * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def new_figure(figure_title):
    fig, ax = plt.subplots()
    ax.set_title(figure_title)
    return ax

