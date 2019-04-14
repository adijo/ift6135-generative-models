import torch
import torch.optim as optim
import Problem1.utils as utils
import numpy as np
import matplotlib.pyplot as plt

from Problem1 import networks
from ta_code import samplers


def discriminator(p_distribution, q_distribution, parameters):
    """
    Problem 1.1 solution

    :param p_distribution: The real data distribution
    :param q_distribution: The fake data distribution
    :param parameters: The hyper-parameters to use
    :return: jensen_shannon_distance, D_loss at the end of training
    """
    return 0, 0


def critic(x_distribution, y_distribution, parameters):
    """
    Problem 1.2 solution

    :param p_distribution: The real data distribution
    :param q_distribution: The fake data distribution
    :param parameters: The hyper-parameters to use
    :return: jensen_shannon_distance, D_loss at the end of training
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model = networks.SimpleMLP(parameters['input_dimensions']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'])

    one = torch.FloatTensor([1])
    minus_one = one * -1
    if use_cuda:
        one = one.cuda(0)
        minus_one = minus_one.cuda(0)

    for i in range(parameters['num_iterations']):
        # Sample data
        real_data = torch.from_numpy(x_distribution(0, parameters['batch_size'])).float()
        if use_cuda:
            real_data = real_data.cuda(0)

        fake_data = torch.from_numpy(y_distribution(parameters['phi'], parameters['batch_size'])).float()
        if use_cuda:
            fake_data = fake_data.cuda(0)

        model.zero_grad()

        # Train using real data
        D_real = model(real_data)
        D_real = D_real.mean()
        D_real.backward(minus_one)

        # Train using fake data
        D_fake = model(fake_data)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # Train using gradient penalty
        gradient_penalty = utils.get_gradient_penalty(model, real_data.data, fake_data.data, use_cuda, parameters['batch_size'], parameters['lambda_constant'])
        gradient_penalty.backward()

        D_loss = D_fake - D_real + gradient_penalty
        wasserstein_distance = D_real - D_fake
        optimizer.step()

        if i % 10 == 9:
            print(
                "Iteration {}/{}: Wasserstein Distance: {}, D_loss: {}"
                .format(i+1, parameters['num_iterations'], wasserstein_distance, D_loss)
            )

    return wasserstein_distance, D_loss


def problem_1_3():
    """
    Problem 1.3 solution
    """
    parameters = {
        'batch_size': 512,
        'learning_rate': 1e-3,
        'num_iterations': 100,
        'lambda_constant': 10,
        'input_dimensions': 2,
        'phi': -1
    }

    phi_range = np.arange(-1, 1, 0.1)
    jensen_shannon_distances = np.zeros(len(phi_range))
    wasserstein_distances = np.zeros(len(phi_range))

    for i in range(1):# range(len(phi_range)):
        parameters['phi'] = phi_range[i]
        print("\n==============================================")
        print("TOTAL PROGRESS = {}%".format(100.0*i/len(phi_range)))
        print("==============================================\n")

        print("-> Training discriminator (phi = {})".format(parameters['phi']))
        jensen_shannon_distance, D_loss = discriminator(
            p_distribution=samplers.distribution1,
            q_distribution=samplers.distribution1,
            parameters=parameters
        )

        print("-> Training critic (phi = {})".format(parameters['phi']))
        wasserstein_distance, D_loss = critic(
            x_distribution=samplers.distribution1,
            y_distribution=samplers.distribution1,
            parameters=parameters
        )

        jensen_shannon_distances[i] = jensen_shannon_distance
        wasserstein_distances[i] = wasserstein_distance

    # TODO: Plot the two graphs using (phi_range, jensen_shannon_distances) and (phi_range, wasserstein_distances)


if __name__ == '__main__':
    problem_1_3()
