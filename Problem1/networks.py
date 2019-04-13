import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        y = x.view(-1, 10)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y

# class JSDLoss(outputs, distribution_type):
#     def forward(self, distribution_p_values, distribution_q_values, discriminator):
#         real_samples_loss = expected_log_likelihood(discriminator, distribution_p_values, "real")
#         fake_samples_loss = expected_log_likelihood(discriminator, distribution_q_values, "fake")
#
#         # We are maximizing the objective function, so we prefix it with a minus
#         loss = -(torch.log(2.0) + 0.5 * (torch.mean(D(X)) + torch.mean(1 - D(Y))))
#         return loss   # a single number (averaged loss over batch samples)
#
#     def backward(self, grad_output):
#         ... # implementation
#        return grad_input, None

def expected_log_likelihood(discriminator, distribution_values, distribution_type):
    stats = average_spans_and_probabilities(distribution_values)

    if distribution_type.lower() == 'real':
        return sum(stats[0][i]*np.log(stats[1][i]) for i in range(len(stats[0])))
    if distribution_type.lower() == 'fake':
        return 1

    raise ValueError("distribution_type has to be either real or fake")

def average_spans_and_probabilities(X_values, num_bins=10):
    """
    Get a vector of the span of values in X_values and the probability of the values in X_values to be in each bucket of the span

    :param X_values: A list of values generated by a probability function
    :param num_bins: (int) How many bins our histogram should have
    :return:
    """
    histogram = np.histogram(X_values, num_bins)
    min_value = histogram[1][0]
    span_size = (histogram[1][1]-histogram[1][0])
    average_spans = np.array([(min_value+span_size/2.0)+i*span_size for i in range(len(histogram[1])-1)])
    probability_per_average_span = histogram[0]/len(X_values)
    return average_spans, probability_per_average_span
