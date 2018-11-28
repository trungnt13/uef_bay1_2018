# Prepare the libraries for the exercise
import matplotlib
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

import tensorflow_probability as tfp
tfd = tfp.distributions
np.random.seed(5218)
# ===========================================================================
# Some configurations
# ===========================================================================
np.random.seed(5218)

# ===========================================================================
# Ex1
# ===========================================================================
# n_samples = 100
# # Two independent dice
# X = np.random.randint(1, 7, n_samples)
# Y = np.random.randint(1, 7, n_samples)

# Z = 8 * np.log(X) - np.random.rand(n_samples)

# # plotting the distrubtion
# plt.figure(figsize=(16, 4))
# ids = np.arange(n_samples)
# plt.subplot(1, 3, 1)
# plt.scatter(ids, X, s=4); plt.title("X")
# plt.subplot(1, 3, 2)
# plt.scatter(ids, Y, s=4); plt.title("Y")
# plt.subplot(1, 3, 3)
# plt.scatter(ids, Z, s=4); plt.title("Z")

# print("Cov(X, Y):\n", np.cov(X, Y))
# print("Cov(X, Z):\n", np.cov(X, Z))
# print("Cov(Y, Z):\n", np.cov(Y, Z))

# ===========================================================================
# Ex4 Correct answer
# ===========================================================================
def gmm(batch_size, n_clusters, n_dim, sigma0,
        using_multivariate_normal=True):
  """ This is the solution for the process in 1.3 """
  # parameters for Dirichlet distribution
  alpha = np.ones(shape=(n_clusters,))
  print("Alpha:", alpha)
  print("Sigma0:", sigma0)

  # step: 1
  dirichlet = tfd.Dirichlet(concentration=alpha)
  theta = dirichlet.sample()

  # step: 2
  # ====== using multi-variate normal for generating components' mean ====== #
  if using_multivariate_normal:
    normal_1 = tfd.MultivariateNormalDiag(
        loc=np.full(shape=(n_dim,), fill_value=0, dtype='float32'),
        scale_diag=np.full(shape=(n_dim,), fill_value=sigma0, dtype='float32'))
  # ====== using multiple Normal distribution instead ====== #
  else:
    normal_1 = tfd.Normal(
        loc=np.full(shape=(n_dim,), fill_value=0, dtype='float32'),
        scale=np.full(shape=(n_dim,), fill_value=sigma0, dtype='float32'))
    normal_1 = tfd.Independent(normal_1, reinterpreted_batch_ndims=1)
  # sampling `n_clusters` time, hence, mean for each cluster
  mu_k = normal_1.sample(n_clusters) # (batch_size, n_clusters, n_dim)

  # ====== Now for the assignment, need 1 indicator for each
  # examples, hence we need `batch_size` amount of indicator
  # (step: 3(a))====== #
  categorical = tfd.OneHotCategorical(probs=theta)
  z = categorical.sample(batch_size) # (batch_size, n_clusters)
  z = tf.cast(z, tf.bool)

  # ====== sampling the data points (step: 3(b)) ====== #
  normal_2 = tfd.Normal(loc=mu_k, scale=1)
  # this make each draw sample will generate sample for
  # all 4 components
  normal_2 = tfd.Independent(normal_2, reinterpreted_batch_ndims=2)
  x_all_components = normal_2.sample(batch_size) # (batch_size, n_clusters, n_dim)

  # ====== selecting the right component for each sample (step: 3(b)) ====== #
  # (batch_size, n_clusters, n_dim) * (batch_size, n_clusters)
  # = (batch_size, n_dim)
  x = tf.boolean_mask(x_all_components, z)

  return x.numpy(), np.argmax(z.numpy(), axis=-1)

X1, Z1 = gmm(1000, n_clusters=4, n_dim=2, sigma0=2,
             using_multivariate_normal=True)
X2, Z2 = gmm(1000, n_clusters=4, n_dim=2, sigma0=10,
             using_multivariate_normal=True)
X3, Z3 = gmm(1000, n_clusters=4, n_dim=2, sigma0=2,
             using_multivariate_normal=False)
X4, Z4 = gmm(1000, n_clusters=4, n_dim=2, sigma0=10,
             using_multivariate_normal=False)

def plot_clusters(x, z, title):
  # visualizing the mixture
  import seaborn as sns
  cluster_colors = sns.color_palette("hls", len(np.unique(Z1)))
  plt.scatter(x[:, 0], x[:, 1], c=[cluster_colors[i] for i in z], s=6)
  plt.title(str(title))

plt.subplot(1, 4, 1)
plot_clusters(X1, Z1, "Sigma:2  Multi-variate")
plt.subplot(1, 4, 2)
plot_clusters(X2, Z2, "Sigma:10 Multi-variate")
plt.subplot(1, 4, 3)
plot_clusters(X3, Z3, "Sigma:2  Normal")
plt.subplot(1, 4, 4)
plot_clusters(X4, Z4, "Sigma:10 Normal")

# ===========================================================================
# Ex4 Bonus
# ===========================================================================
# number of samples
n_samples = 1000
# number of cluster
n_clusters = 2
# number of feature dimension
n_dim = 2

# parameters for Dirichlet distribution
alpha = np.ones(shape=(n_clusters,))
# sigma for the generation of the mean
# NOTE: we have to repeat the sigma for each cluster
sigma = np.repeat(np.expand_dims(np.array([0.5, 8], dtype='float32'),
                                 axis=0),
                  repeats=n_clusters, axis=0)
print(sigma.shape)

dirichlet = tfd.Dirichlet(concentration=alpha)
print(dirichlet)
theta = dirichlet.sample()
print("Theta:", theta)

normal_1 = tfd.Normal(loc=0, scale=sigma)
print(normal_1)
normal_1 = tfd.Independent(normal_1, reinterpreted_batch_ndims=2)
print(normal_1)
mu_k = normal_1.sample()
print("Mu_k:", mu_k)

categorical = tfd.Categorical(probs=theta)
print(categorical)
z = categorical.sample()
print("Z_i:", z)

mu_x = mu_k[z]
normal_2 = tfd.Normal(loc=mu_x, scale=1)
print(normal_2)
normal_2 = tfd.Independent(normal_2, reinterpreted_batch_ndims=1)
print(normal_2)
x = normal_2.sample()
print("X_i:", x)

def gmm(batch_size, n_clusters, n_dim):
  """ This is the solution for the process in 1.3 """
  # parameters for Dirichlet distribution
  alpha = np.ones(shape=(n_clusters,))
  sigma = np.repeat(np.random.randint(1, 25, (1, n_dim)),
                    repeats=n_clusters, axis=0).astype('float32')
  print("Alpha:", alpha)
  print("Sigma0:", sigma[0])

  dirichlet = tfd.Dirichlet(concentration=alpha)
  theta = dirichlet.sample(batch_size)

  normal_1 = tfd.Normal(loc=0, scale=sigma)
  normal_1 = tfd.Independent(normal_1, reinterpreted_batch_ndims=2)
  mu_k = normal_1.sample(batch_size) # (batch_size, n_clusters, n_dim)

  categorical = tfd.OneHotCategorical(probs=theta)
  z = categorical.sample() # (batch_size, n_clusters)
  z = tf.cast(z, tf.bool)

  # another solution is using tf.where (but it is only
  # feasible for 2 components)
  # tf.where(z == 0, component_1, component_2)

  # (batch_size, n_dim)
  mu_x = tf.boolean_mask(mu_k, z)
  normal_2 = tfd.Normal(loc=mu_x, scale=1)
  normal_2 = tfd.Independent(normal_2, reinterpreted_batch_ndims=1)
  x = normal_2.sample()
  return x.numpy(), np.argmax(z.numpy(), axis=-1)
X1, Z1 = gmm(1000, n_clusters=2, n_dim=2)

def gmm_simple(batch_size):
  """ This is the solution for the process in 1.3 """
  dirichlet = tfd.Dirichlet(concentration=alpha)
  theta = dirichlet.sample(batch_size)

  sigma0 = sigma[0][None, :]
  components = [tfd.Independent(tfd.Normal(loc=0,
                                           scale=np.repeat(sigma0, batch_size, 0)),
                                reinterpreted_batch_ndims=1)
                for i in range(n_clusters)]
  mixture = tfd.Mixture(
      cat=tfd.Categorical(probs=theta),
      components=components)

  normal_2 = tfd.Normal(loc=mixture.sample(), scale=1)
  normal_2 = tfd.Independent(normal_2, reinterpreted_batch_ndims=1)
  x = normal_2.sample()
  return x
X2 = gmm_simple(1000)

def gmm_fun(batch_size):
  """ This is the solution for the process in 1.3 """
  # parameters for Dirichlet distribution
  # ====== configuration ====== #
  n_clusters = 2
  n_dim = 2
  alpha = np.ones(shape=(n_clusters,))
  mu = [0, 1000]
  sigma = [1, 1]

  # ====== validate all argument ====== #
  assert len(mu) == n_clusters
  assert len(alpha) == n_clusters
  assert len(sigma) == n_dim
  # ====== preprocessing ====== #
  mu = np.repeat(np.array(mu)[:, None],
                 repeats=n_dim, axis=1).astype('float32')
  sigma = np.repeat(np.array(sigma)[None, :],
                    repeats=n_clusters, axis=0).astype('float32')
  print("Alpha:", alpha)
  print("Mu   :\n", mu)
  print("Sigma:\n", sigma)

  dirichlet = tfd.Dirichlet(concentration=alpha)
  theta = dirichlet.sample(batch_size)

  normal_1 = tfd.Normal(loc=mu, scale=sigma)
  normal_1 = tfd.Independent(normal_1, reinterpreted_batch_ndims=2)
  mu_k = normal_1.sample(batch_size) # (batch_size, n_clusters, n_dim)

  categorical = tfd.OneHotCategorical(probs=theta)
  z = categorical.sample() # (batch_size, n_dim)
  z = tf.cast(z, tf.bool)

  # (batch_size, n_dim)
  mu_x = tf.boolean_mask(mu_k, z)
  normal_2 = tfd.Normal(loc=mu_x, scale=1)
  normal_2 = tfd.Independent(normal_2, reinterpreted_batch_ndims=1)
  x = normal_2.sample()
  return x.numpy(), np.argmax(z.numpy(), axis=-1)

X1, Z1 = gmm_fun(1000)
