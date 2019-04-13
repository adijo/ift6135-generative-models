import numpy as np
import scipy.stats
import scipy as sp
import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
from ta_code.samplers import distribution3 as gaussian_distribution
from ta_code.samplers import distribution2
from ta_code.samplers import distribution1

from Problem1 import networks

# https://wiseodd.github.io/techblog/2017/01/20/gan-pytorch/
# https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/


def main():
    pass


def problem_1(p_distribution, q_distribution):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 512
    model_input_size = 100
    num_epochs = 100
    D = networks.Discriminator(model_input_size).to(device)
    optimizer = optim.Adam(D.parameters(), lr=1e-3)

    for i in range(num_epochs):
        # Sample data
        X_encodings, Y_encodings = one_hot_encodings(p_distribution(batch_size), q_distribution(batch_size), model_input_size)
        #X_stats = Variable(torch.from_numpy(average_spans_and_probabilities(X_values)))
        #Y_stats = Variable(torch.from_numpy(average_spans_and_probabilities(Y_values)))

        # Dicriminator forward-loss-backward-update
        inputs = X_values, Y_values
        outputs = D(X_values), D(Y_values)
        loss = networks.JSDLoss(inputs, outputs)
        loss.backward()
        optimizer.step()


def one_hot_encodings(p_distribution_values, q_distribution_values, bins):
    p_distribution_values = np.array([value[0] for value in p_distribution_values])
    q_distribution_values = np.array([value[0] for value in q_distribution_values])
    joint_values = np.concatenate((p_distribution_values+q_distribution_values))
    min_value = np.min(joint_values)

    histogram = np.histogram(p_distribution_values+q_distribution_values, bins=bins)
    bin_size = histogram[1][1] - histogram[1][0]

    X_encodings = get_one_hot(np.array([int((value-min_value)/bin_size) for value in p_distribution_values]), bins)
    Y_encodings = get_one_hot(np.array([int((value-min_value)/bin_size) for value in q_distribution_values]), bins)

    return X_encodings, Y_encodings


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


if __name__ == '__main__':
    X_encodings, Y_encodings = one_hot_encodings(gaussian_distribution(5), gaussian_distribution(5), 10)
    k = 4
