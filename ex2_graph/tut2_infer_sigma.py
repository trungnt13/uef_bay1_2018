# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np

import tensorflow as tf
tf.enable_eager_execution()

import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
tfd = tfp.distributions

# ===========================================================================
# Constant
# ===========================================================================
a = 8
b = 0.5
mu = 0
n_samples = 100000

# ===========================================================================
# Following the generative procedure
# ===========================================================================
# Step 1: generate the precision Beta
beta_dist = tfd.Gamma(concentration=a, rate=b)
beta = beta_dist.sample(n_samples)
# the prior probability
p_beta_given_a_and_b = beta_dist.prob(beta)

# Step 2: generate the data point
# scale is standard deviation
x_dist = tfd.Normal(loc=mu, scale=tf.sqrt(1 / beta))
x = x_dist.sample()
# the likelihood
p_x_given_mu_and_beta = x_dist.prob(x)

# ====== plotting the prior ====== #
plt.figure()
sns.distplot(beta.numpy(), bins=120, kde=True)
plt.title(r"Prior distribution: $p(\beta|a=%g, b=%g)$" % (a, b))

# ====== plotting the likelihood ====== #
plt.figure()
sns.distplot(x.numpy(), bins=120, kde=True)
plt.title(r"Likelihood distribution: $p(X|\mu=%g, \sigma=\sqrt{\beta^{-1}})$" % mu)

# ====== plotting the posterior ====== #
# the posterior probability, this is only
# proportionally, not exactly because we omit
# the evidence p(X)
# If we want to calculate p(X), we need to marginalize out
# beta using sum rule:
# p(X) = p(X, beta_1) +  p(X, beta_2) + ... +  p(X, beta_∞)
# This is not easy
p_beta_given_x = p_x_given_mu_and_beta * p_beta_given_a_and_b
p_beta_given_x = p_beta_given_x / tf.reduce_sum(p_beta_given_x)
posterior_dist = tfd.Categorical(probs=p_beta_given_x)

beta = beta.numpy()
posterior = []
for i in range(n_samples // 2000):
  idx = posterior_dist.sample(2000).numpy()
  posterior.append(beta[idx])
posterior = np.concatenate(posterior)

plt.figure()
sns.distplot(posterior, bins=120, kde=True)
plt.title(r"Sampled posterior distribution: $p(\beta|X)$")

# ====== plotting the close form solution ====== #
a0 = a + n_samples / 2
b0 = b + n_samples / 2 * np.var(x.numpy())
posterior_dist = tfd.Gamma(concentration=a0, rate=b0)
posterior = posterior_dist.sample(n_samples)

plt.figure()
sns.distplot(posterior, bins=120, kde=True)
plt.title(
    r"Closed form solution: $p(\beta|X) \sim Gamma(a=%g, b=%g)$"
    % (a0, b0))


from odin import visual as V
V.plot_save('/tmp/tmp.pdf', dpi=200)
