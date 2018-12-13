from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np

import seaborn
import pandas as pd

# path to the downloaded fish data
path = '/tmp/fish_data.txt'

# "pituus": fish length
# "Allas": pool
ds = pd.read_csv(path,
                 sep=";", decimal=',', encoding="Latin-1")
# remove all column that contain NaN values
ids = ds.apply(lambda x: np.all(x.notna()), axis=0)
ds = ds.iloc[:, ids.tolist()]
print(ds.describe())

# ====== plotting the histogram of each features ====== #
ds.hist(figsize=(8, 8), bins=25)
seaborn.pairplot(ds)

# ====== Getting the tank and fish length data ====== #
# Pituus: length
# Paino : weight
data = ds[['Allas', 'Pituus']]
# ====== grouping the data by the Tank ====== #
group = data.groupby(by='Allas')
pool_data = {}
n_pools = len(group.groups)
plt.figure(figsize=(12, 12))
for i, (name, dat) in enumerate(group.groups.items()):
  name = "Pool ID:%s" % int(name)
  dat = np.array(dat, dtype='float32')
  # store the data for later analysis
  pool_data[name] = dat
  # plotting the histogram again
  ax = plt.subplot(n_pools, 2, (i * 2) + 1)
  seaborn.distplot(dat, bins=120, ax=ax)
  plt.title(name, fontsize=10)

  ax = plt.subplot(n_pools, 2, (i * 2) + 2)
  seaborn.distplot(np.log1p(dat), bins=120, ax=ax)
  plt.title("log-scale", fontsize=10)

# re-organize the layout
plt.tight_layout()

# ====== create the GMM for fitting the data ====== #
def fitting_gmm_em(x, n_components):
  """ This method EM (Expectation maximization) to do
  inference for Gaussian mixture model

  More information can be found here:
  https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
  """
  from sklearn.mixture import GaussianMixture
  gmm = GaussianMixture(n_components=int(n_components),
                        covariance_type='diag', n_init=8,
                        random_state=5218)
  # the input data must be at least 2D, so we
  # need to do some preprocessing
  x = np.atleast_2d(x).T
  gmm.fit(x)
  return gmm

# ====== we iterate over each pool to fit GMM ====== #
pool_models = {}
for name, dat in pool_data.items():
  gmm = fitting_gmm_em(dat, n_components=2)
  pool_models[name] = gmm

# ====== visualize the fitted GMM ====== #
# we need this to draw Gaussian distribution
import matplotlib.mlab as mlab

plt.figure(figsize=(12, 18))
for i, name in enumerate(sorted(pool_models.keys())):
  dat = pool_data[name]
  model = pool_models[name]
  # plotting the histogram
  ax = plt.subplot(n_pools, 1, i + 1)
  ax.grid(True)
  ax.hist(dat, bins=120, facecolor='salmon', alpha=0.2)
  # show the GMM density and mean
  ax = ax.twinx()
  mean = gmm.means_.ravel()
  precision = gmm.precisions_.ravel()
  X = np.linspace(start=np.min(dat), stop=np.max(dat),
                  num=1000)
  proba = gmm.predict_proba(np.atleast_2d(X).T)

  n_components = len(mean)
  for n in range(n_components):
    Y = mlab.normpdf(X, mean[n], np.sqrt(1 / precision[n]))
    _, = ax.plot(X, (Y - np.min(Y)) / (np.max(Y) - np.min(Y)),
            label='Component:%d' % n,
            linewidth=2, linestyle='--')
    ax.plot(X, proba[:, n], color=_.get_color(),
            linewidth=1.5, linestyle=':')
  # show some information on the title
  plt.legend()
  plt.title(name)

plt.tight_layout()

# ====== save the figures ====== #
# plt.savefig('/tmp/tmp.pdf')
