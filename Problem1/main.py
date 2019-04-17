import torch
import torch.optim as optim
import Problem1.utils as utils
import numpy as np
import matplotlib.pyplot as plt

from Problem1 import networks
from ta_code import samplers


def problem_1_1_discriminator(p_distribution, q_distribution, parameters):
    """
    Problem 1.1 solution

    :param p_distribution: The real data distribution
    :param q_distribution: The fake data distribution
    :param parameters: The hyper-parameters to use
    :return: jensen_shannon_distance at the end of training
    """
    # =====
    # Setup
    # =====
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model = networks.DiscriminatorMLP(parameters['input_dimensions'], parameters['hidden_layers_size']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=parameters['learning_rate'])

    minus_one = torch.FloatTensor([1]) * -1
    if use_cuda:
        minus_one = minus_one.cuda(0)

    # ========
    # Training
    # ========
    ten_percent = parameters['num_iterations']/10
    for i in range(parameters['num_iterations']):
        # ===========
        # Sample data
        # ===========
        real_data = torch.from_numpy(p_distribution(0, parameters['batch_size'])).float()
        if use_cuda:
            real_data = real_data.cuda(0)

        fake_data = torch.from_numpy(q_distribution(parameters['phi'], parameters['batch_size'])).float()
        if use_cuda:
            fake_data = fake_data.cuda(0)

        model.zero_grad()

        # ===================================================================
        # Feed forward and back propagation
        # ===================================================================
        # We want:
        #   D_real           to approach 1 coming from 0
        #   D_fake           to approach 0 coming from 1
        #
        # The following feed forward and back propagation operations
        # achieve that goal, therefore maximizing the JSD objective function:
        # ===================================================================
        D_real = model(real_data)
        D_real = torch.mean(torch.log(D_real))

        D_fake = model(fake_data)
        D_fake = torch.mean(torch.log(1 - D_fake))

        # Calculate current distance
        current_jensen_shannon_distance = torch.log(torch.cuda.FloatTensor([2])) + 0.5*(D_real + D_fake)
        current_jensen_shannon_distance.backward(minus_one)

        optimizer.step()

        if i % ten_percent == 0 or i+1 == parameters['num_iterations']:
            print(
                "Iteration {}/{}: Current Jensen-Shannon distance: {}"
                .format(i + 1, parameters['num_iterations'], current_jensen_shannon_distance.item())
            )

    return current_jensen_shannon_distance.item()


def problem_1_2_critic(p_distribution, q_distribution, parameters):
    """
    Problem 1.2 solution

    :param p_distribution: The real data distribution
    :param q_distribution: The fake data distribution
    :param parameters: The hyper-parameters to use
    :return: wasserstein_distance at the end of training
    """
    # =====
    # Setup
    # =====
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model = networks.CriticMLP(parameters['input_dimensions'], parameters['hidden_layers_size']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=parameters['learning_rate'])

    # ========
    # Training
    # ========
    ten_percent = parameters['num_iterations']/10
    for i in range(parameters['num_iterations']):
        # ===========
        # Sample data
        # ===========
        real_data = torch.from_numpy(p_distribution(0, parameters['batch_size'])).float()
        if use_cuda:
            real_data = real_data.cuda(0)

        fake_data = torch.from_numpy(q_distribution(parameters['phi'], parameters['batch_size'])).float()
        if use_cuda:
            fake_data = fake_data.cuda(0)

        model.zero_grad()

        # ====================================================================
        # Feed forward and back propagation
        # ====================================================================
        # We want:
        #   D_real           to approach 1 coming from 0
        #   D_fake           to approach 0 coming from 1
        #   gradient_penalty to approach 0 coming from +inf
        #
        # The following feed forward and back propagation operations
        # achieve that goal, therefore maximizing the WGAN objective function:
        # ====================================================================
        D_real = model(real_data)
        D_real = D_real.mean()

        D_fake = model(fake_data)
        D_fake = D_fake.mean()

        gradient_penalty = utils.get_gradient_penalty(model, real_data.data, fake_data.data, use_cuda,
                                                      parameters['batch_size'], parameters['lambda_constant'])

        # Calculate current loss and distance
        current_wasserstein_distance = D_real - D_fake
        current_D_loss = -current_wasserstein_distance + gradient_penalty
        current_D_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        if i % ten_percent == 0 or i+1 == parameters['num_iterations']:
            print(
                "Iteration {}/{}: Current Wasserstein distance: {}, Critic loss: {}"
                .format(i+1, parameters['num_iterations'], current_wasserstein_distance.item(), current_D_loss.item())
            )

    return current_wasserstein_distance.item()


def problem_1_3():
    """
    Problem 1.3 solution
    """
    print("\n===================================================================")
    print("Running the training for problem 1.3 and plotting graphs afterwards")
    print("===================================================================\n")

    parameters = {
        'batch_size': 512,
        'learning_rate': 1e-3,
        'num_iterations': 40000,
        'lambda_constant': 10,
        'input_dimensions': 2,
        'hidden_layers_size': 32,
        'phi': -1
    }

    phi_range = np.arange(-1, 1.1, 0.1)
    jensen_shannon_distances = np.zeros(len(phi_range))
    wasserstein_distances = np.zeros(len(phi_range))

    # ==================================================================
    # Gathering the distances of p and q for the different values of phi
    # ==================================================================
    for i in range(len(phi_range)):
        parameters['phi'] = phi_range[i]

        print("-> Training problem_1_1_discriminator (phi = {})".format(parameters['phi']))
        jensen_shannon_distance = problem_1_1_discriminator(
            p_distribution=samplers.distribution1,
            q_distribution=samplers.distribution1,
            parameters=parameters
        )

        print("-> Training problem_1_2_critic (phi = {})".format(parameters['phi']))
        wasserstein_distance = problem_1_2_critic(
            p_distribution=samplers.distribution1,
            q_distribution=samplers.distribution1,
            parameters=parameters
            )

        jensen_shannon_distances[i] = jensen_shannon_distance
        wasserstein_distances[i] = wasserstein_distance

        print("\n=======================")
        print("TOTAL PROGRESS = {}%".format(int(100.0*((i+1)/len(phi_range)))))
        print("=======================\n")

    # ====================
    # Plotting the results
    # ====================
    ax = utils.new_figure("Jensen-Shannon distance")
    ax.set_xlabel("Phi")
    ax.set_ylabel("Distance")
    ax.plot(phi_range, jensen_shannon_distances, "o")
    plt.show()

    ax = utils.new_figure("Wasserstein distance")
    ax.set_xlabel("Phi")
    ax.set_ylabel("Distance")
    ax.plot(phi_range, wasserstein_distances, "o")
    plt.show()


def problem_1_4_discriminator(f_1_distribution, f_0_distribution, parameters):
    """
    Problem 1.4 solution

    :param f_1_distribution: The unknown distribution
    :param f_0_distribution: The known distribution
    :param parameters: The hyper-parameters to use
    :return: The trained discriminator
    """
    # =====
    # Setup
    # =====
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model = networks.DiscriminatorMLP(parameters['input_dimensions'], parameters['hidden_layers_size']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=parameters['learning_rate'])

    minus_one = torch.FloatTensor([1]) * -1
    if use_cuda:
        minus_one = minus_one.cuda(0)

    # ========
    # Training
    # ========
    ten_percent = parameters['num_iterations']/10
    for i in range(parameters['num_iterations']):
        # ===========
        # Sample data
        # ===========
        f_1_samples = torch.from_numpy(f_1_distribution(parameters['batch_size'])).float()
        if use_cuda:
            f_1_samples = f_1_samples.cuda(0)

        f_0_samples = torch.from_numpy(f_0_distribution(parameters['batch_size'])).float()
        if use_cuda:
            f_0_samples = f_0_samples.cuda(0)

        model.zero_grad()

        # ==========================================================
        # Feed forward and back propagation
        # ==========================================================
        # We want:
        #   D_real           to approach 1 coming from 0
        #   D_fake           to approach 0 coming from 1
        #
        # The following feed forward and back propagation operations
        # achieve that goal, therefore maximizing the objective:
        # ==========================================================
        D_known = model(f_0_samples)
        D_known = torch.mean(torch.log(D_known))

        D_unknown = model(f_1_samples)
        D_unknown = torch.mean(torch.log(1 - D_unknown))

        objective = D_known + D_unknown
        objective.backward(minus_one)

        optimizer.step()

        if i % ten_percent == 0 or i+1 == parameters['num_iterations']:
            print(
                "Iteration {}/{}, Value of the objective to maximize: {}"
                .format(i + 1, parameters['num_iterations'], objective.item())
            )

    return model


if __name__ == '__main__':
    problem_1_3()
