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
# we only take sample with "MITTAUSAIKA" = "LOPETUS"
selected_row = ds.MITTAUSAIKA == "LOPETUS"
ds = ds[selected_row]
print(ds.describe())

# ====== plotting the histogram of each features ====== #
ds.hist(figsize=(8, 8), bins=25)
seaborn.pairplot(ds)

# ====== Getting the tank and fish length data ====== #
# Pituus: length
# Paino : weight
data = ds[['Allas', 'Pituus']]
# ====== grouping the data by the Tank ====== #
from sklearn.mixture import GaussianMixture
n_components = 2
# selecting random color for each component
colors = seaborn.color_palette(
    palette='Set2', n_colors=n_components)
# we need this to draw Gaussian distribution
import matplotlib.mlab as mlab

for pool_id in data.Allas.unique():
  # select all data from given tank
  pool_data = data.Pituus[data.Allas == pool_id]

  # Fitting Gaussian on Pool data
  gmm = GaussianMixture(n_components=int(n_components),
                        covariance_type='diag', n_init=8,
                        random_state=5218)
  # the input data must be at least 2D, so we
  # need to do some preprocessing
  pool_data = np.atleast_2d(pool_data.values).T
  gmm.fit(pool_data)

  # Plotting the histogram
  plt.figure(figsize=(8, 2))
  seaborn.distplot(pool_data, bins=18)

  # Visualizing the GMM
  mean = gmm.means_.ravel()
  precision = gmm.precisions_.ravel()
  xmin, xmax = plt.gca().get_xlim()
  X = np.linspace(start=xmin, stop=xmax,
                  num=1000)
  ax = plt.gca().twinx()
  for n in range(n_components):
    Y = mlab.normpdf(X, mean[n], np.sqrt(1 / precision[n]))
    _, = ax.plot(X, Y,
                 label='Component:%d' % n,
                 color=colors[n], linewidth=3, linestyle='--')

  # show extra info
  ax.set_xlim((np.min(pool_data) - 20,
               np.max(pool_data) + 20))
  plt.legend()
  plt.title("Pool #%s" % str(pool_id))
