#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch 
import matplotlib.pyplot as plt

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))
plt.show()


############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
 
from ta_code import samplers
from Problem1 import main

print("\n==================================================================")
print("Running the training for problem 1.4 and plotting graph afterwards")
print("==================================================================\n")
D = main.problem_1_4_discriminator(
    f_1_distribution=samplers.distribution4,
    f_0_distribution=samplers.distribution3,
    parameters={
        'batch_size': 512,
        'learning_rate': 1e-3,
        'num_iterations': 20000,
        'input_dimensions': 1,
        'hidden_layers_size': 128
    }
)

############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density

# evaluate xx using your discriminator
xx_tensor = torch.from_numpy(xx).cuda().float() if torch.cuda.is_available() else torch.from_numpy(xx).float()
xx_tensor = xx_tensor.view(xx_tensor.shape[0], 1)
r = D(xx_tensor).data.cpu().numpy()
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')

# estimate the density of distribution4 (on xx) using the discriminator;
estimate = samplers.distribution3(xx_tensor.shape[0])*r/(1-r)
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')
plt.show()
