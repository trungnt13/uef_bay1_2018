# Prepare the libraries for the exercise
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import daft
import seaborn

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

import tensorflow_probability as tfp
tfd = tfp.distributions
np.random.seed(5218)

# Distribution parameters
n_samples = 1000

# ===========================================================================
# First task
# Draw 1000 samples from this Gaussian varying parameters
# Make scatter plots.
# ===========================================================================
n_mean = 3
n_variance = 3
idx = 1
plt.figure(figsize=(8, (n_mean * n_variance) * 3))
for mean in np.linspace(0, 10, num=n_mean):
  for variance in np.linspace(1, 20, num=n_variance):
    # sampling
    dist = tfd.Normal(loc=mean,
                      scale=np.sqrt(variance))
    samples = dist.sample(n_samples).numpy()
    # plotting the scatter plot
    ax = plt.subplot(n_mean * n_variance, 2, idx)
    ax.scatter(np.arange(len(samples)), samples, s=4)
    ax.grid(True)
    plt.title("Mean:%.2f  Var:%.2f" % (mean, variance))
    # plotting the density plot
    ax = plt.subplot(n_mean * n_variance, 2, idx + 1)
    seaborn.kdeplot(samples, shade=True)
    ax.grid(True)
    # move to next row
    idx += 2
_ = plt.tight_layout()

# ===========================================================================
# Second task
# For this task we need to specify the prior distribution of
# mean and precision
# ===========================================================================
# ====== define function for posteriors distribution ====== #
def mean_posterior_prior(samples, mean, variance,
                         mu_prior=0, tau_prior=2):
  """ This based on equation (55) and (56)

  Parameters
  ----------
  samples : array [n_samples,]
    the samples from data distribution (i.e. X ~ Normal(mu, sigma^2))
  mean : a scalar
    the mean of the data distribution (i.e. mu)
  variance : a scalar
    the variance of the data distribution (i.e. sigma^2 or scale^2)
  mu_prior : a scalar
    or mu_0, the mean of the prior distribution (i.e.
    mu ~ Normal(mu_0, tau^2))
  tau_prior : a scalar
    the standard deviation of the prior distribution
    (i.e. mu ~ Normal(mu_0, tau^2))
  """
  N = len(samples)
  # create the prior distribution
  prior = tfd.Normal(loc=mu_prior,
                     scale=tau_prior,
                     name="mean_prior")
  # create the posterior distribution
  var_N = (N / variance +
           1 / np.power(tau_prior, 2)) ** (-1)
  mu_N = var_N * (N / variance * np.mean(samples) +
                  1 / np.power(tau_prior, 2) * mu_prior)
  posterior = tfd.Normal(loc=mu_N,
                         scale=np.sqrt(var_N),
                         name="mean_posterior")
  return posterior, prior

def precision_posterior_prior(samples, mean, variance,
                              a_prior=5, b_prior=6):
  """ This using the solution of Question 3 from Exercise 3

  Parameters
  ----------
  samples : array [n_samples,]
    the samples from data distribution (i.e. X ~ Normal(mu, sigma^2))
  mean : a scalar
    the mean of the data distribution (i.e. mu)
  variance : a scalar
    the variance of the data distribution (i.e. sigma^2 or scale^2)
  a_prior : a scalar
    alpha for prior Gamma (i.e. precision ~ Gamma(a, b))
  b_prior : a scalar
    beta for prior Gamma (i.e. precision ~ Gamma(a, b))
  """
  # create the prior distribution
  prior = tfd.Gamma(concentration=a_prior,
                    rate=b_prior,
                    name="precision_prior")
  # create the posterior distribution
  N = len(samples)
  a_N = a_prior + N / 2
  b_N = b_prior + 1 / 2 * np.sum((samples - mean) ** 2)
  posterior = tfd.Gamma(concentration=a_N,
                        rate=b_N,
                        name="precision_posterior")
  return posterior, prior

# ******************** perform sampling ******************** #
n_mean = 2
n_variance = 2
n_columns = 7
idx = 1
indices = np.arange(n_samples)
matplotlib.rc('font', **{'size': 6})

plt.figure(figsize=(18, (n_mean * n_variance) * 3))
for mean in np.linspace(0, 10, num=n_mean):
  for variance in np.linspace(1, 200, num=n_variance):
    # sampling
    dist = tfd.Normal(loc=mean,
                      scale=np.sqrt(variance))
    samples = dist.sample(n_samples).numpy()

    # sampling from the mean and precision posterior distribution
    # Note: a list is returned,
    # first distribution is posterior
    # second distribution is prior
    mean_dist = mean_posterior_prior(
        samples, mean, variance)
    precision_dist = precision_posterior_prior(
        samples, mean, variance)

    # ====== plotting the X distribution ====== #
    ax = plt.subplot(n_mean * n_variance, n_columns, idx)
    ax.scatter(indices, samples, s=4)
    ax.grid(True)
    plt.title(r"$\mu=%.2f;\sigma^2=%.2f$" % (mean, variance))

    ax = plt.subplot(n_mean * n_variance, n_columns, idx + 1)
    seaborn.distplot(samples, norm_hist=True)

    # ====== plotting the mean posterior distribution ====== #
    ax = plt.subplot(n_mean * n_variance, n_columns, idx + 2)
    mean_posterior_samples = mean_dist[0].sample(n_samples).numpy()
    ax.scatter(indices, mean_posterior_samples, s=4)
    ax.grid(True)
    plt.title(r"Mean($\mu_N=%.4f;\sigma_N^2=%.4f$)" %
              (mean_dist[0].loc.numpy(),
               mean_dist[0].scale.numpy()**2))

    # plot the density of the posterior
    ax = plt.subplot(n_mean * n_variance, n_columns, idx + 3)
    seaborn.distplot(mean_posterior_samples, norm_hist=True)

    # plotting the prior
    ax = ax.twinx()
    xmin, xmax = ax.get_xlim()
    series = np.linspace(xmin, xmax, num=n_samples)
    ax.plot(series, mean_dist[1].prob(series).numpy(),
            color='red', linestyle='--')
    plt.title(r"Prior $\mathcal{N}$($\mu_N=%.4f;\sigma_N^2=%.4f$)" %
      (mean_dist[1].loc.numpy(),
       mean_dist[1].scale.numpy()**2))

    # ====== plotting the precision posterior distribution ====== #
    ax = plt.subplot(n_mean * n_variance, n_columns, idx + 4)
    precision_posterior_samples = precision_dist[0].sample(n_samples).numpy()
    ax.scatter(indices, precision_posterior_samples, s=4)
    ax.grid(True)
    plt.title(r"Precision($a_N=%.2f;b_N=%.2f$)" %
              (precision_dist[0].concentration.numpy(),
               precision_dist[0].rate.numpy()))

    # plot the density of the posterior
    ax = plt.subplot(n_mean * n_variance, n_columns, idx + 5)
    seaborn.distplot(precision_posterior_samples, norm_hist=True)

    # plotting the prior
    ax = ax.twinx()
    xmin, xmax = ax.get_xlim()
    series = np.linspace(xmin, xmax, num=n_samples)
    ax.plot(series, precision_dist[1].prob(series).numpy(),
            color='red', linestyle='--')
    plt.title(r"Prior $Gamma$($a_N=%.2f;b_N=%.2f$)" %
      (precision_dist[1].concentration.numpy(),
       precision_dist[1].rate.numpy()))

    # ====== the bi-variate mean-precision posterior distribution ====== #
    ax = plt.subplot(n_mean * n_variance, n_columns, idx + 6)
    seaborn.kdeplot(mean_posterior_samples,
                    precision_posterior_samples,
                    n_levels=30, shade=True, shade_lowest=False,
                    ax=ax)
    ax.grid(True)
    plt.title(r"Mean-Precision NormalGamma")

    # iterating to next row
    idx += n_columns
_ = plt.tight_layout()

# ===========================================================================
# Bonus task
# ===========================================================================
plt.savefig("/tmp/tmp.pdf")
