import torch
import torch.optim as optim
import Problem1.utils as utils
import numpy as np
import matplotlib.pyplot as plt

from Problem1 import networks
from ta_code import samplers


def problem_1_3():
    batch_size = 512
    phi_range = np.arange(-1, 1, 0.1)

    jensen_shannon_distances = np.zeros(len(phi_range))
    wasserstein_distances = np.zeros(len(phi_range))

    for i in range(len(phi_range)):
        phi = phi_range[i]
        print("\n==============================================")
        print("TOTAL PROGRESS = {}%".format(100.0*i/len(phi_range)))
        print("==============================================\n")

        print("-> Training discriminator (phi = {})".format(phi))
        jensen_shannon_distance, D_loss = discriminator(
            p_distribution=samplers.distribution1(0, batch_size),
            q_distribution=samplers.distribution1(phi, batch_size),
            batch_size=batch_size
        )

        print("-> Training critic (phi = {})".format(phi))
        wasserstein_distance, D_loss = critic(
            p_distribution=samplers.distribution1(0, batch_size),
            q_distribution=samplers.distribution1(phi, batch_size),
            batch_size=batch_size
        )

        jensen_shannon_distances[i] = jensen_shannon_distance
        wasserstein_distances[i] = wasserstein_distance

    # TODO: Plot the two graphs using phi_range as x values and jsd and wsd distances as y values


def discriminator(p_distribution, q_distribution, batch_size=512):
    # TODO: Implement
    return 0, 0


def critic(p_distribution, q_distribution, batch_size=512):
    """
    Problem 1 solution
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    num_iterations = 100
    lambda_constant = 10
    input_dimensions = 2
    model = networks.SimpleMLP(input_dimensions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(num_iterations):
        # Sample data
        real_data = torch.from_numpy(p_distribution).float()
        if use_cuda:
            real_data = real_data.cuda(0)

        fake_data = torch.from_numpy(q_distribution).float()
        if use_cuda:
            fake_data = fake_data.cuda(0)

        model.zero_grad()

        # Train using real data
        D_real = model(real_data)
        D_real = -D_real.mean()
        D_real.backward()

        # Train using fake data
        D_fake = model(fake_data)
        D_fake = D_fake.mean()
        D_fake.backward()

        # Train using gradient penalty
        gradient_penalty = utils.get_gradient_penalty(model, real_data.data, fake_data.data, use_cuda, batch_size, lambda_constant)
        gradient_penalty.backward()

        D_loss = D_fake - D_real + gradient_penalty
        wasserstein_distance = D_real - D_fake
        optimizer.step()

        if i % 10 == 9:
            print(
                "Iteration {}/{}: Wasserstein Distance: {}, D_loss: {}"
                .format(i+1, num_iterations, wasserstein_distance, D_loss)
            )

    return wasserstein_distance, D_loss


if __name__ == '__main__':
    problem_1_3()
