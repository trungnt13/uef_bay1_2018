from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import seaborn

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

tf.enable_eager_execution()

from tut1_normal_gamma import NormalGamma

# This is set to be the same configuration with
# the NormalGamma distribution in the book
# p.102 ``Pattern Recognition and Machine Learning'',
# M. Bishop
dist = NormalGamma(loc=0, scale=np.sqrt(2),
                   concentration=5, rate=6)
samples = dist.sample(25000).numpy()
mean_mean, precision_mean = dist.mean().numpy()


plt.figure(figsize=(8, 8))

# the distribution
seaborn.kdeplot(samples[:, 0], samples[:, 1],
                n_levels=30)

# location of the mean is colored by red-point
plt.scatter(mean_mean, precision_mean,
            s=120, alpha=0.6, color='red')

# some adjustments
plt.grid(True)
plt.xlabel(r"$\mu$")
plt.ylabel(r"$\lambda$")
plt.gca().set_aspect('equal')
plt.savefig('/tmp/tmp.pdf')
