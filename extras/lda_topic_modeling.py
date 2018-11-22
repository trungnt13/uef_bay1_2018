# Require library:
# numpy
# matplotlib
# seaborn
# tensorflow
# tensorflow-probability
# scikit-learn
from __future__ import print_function, division, absolute_import
import os
import re
import matplotlib
# comment this line if you have screen for your computer
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
# ===========================================================================
# Helper
# ===========================================================================
def doc_preprocessing(d):
  d = re.sub('\n{1,}', ' ', d)
  d = re.sub('\s{2,}', ' ', d)
  return d
np.random.seed(5218)

n_features = 1000
n_topics = 3
# cut-off word appear in greater than 95% of the documents
max_df = 0.95
# cut-off word appear in less than 2 documents
min_df = 2
# ===========================================================================
# Loading the data
# ===========================================================================
categories = ['rec.sport.hockey', 'talk.politics.misc']
train_set = fetch_20newsgroups(subset='train',
                               categories=categories,
                               remove=('headers', 'footers', 'quotes'))
test_set = fetch_20newsgroups(subset='test',
                              categories=categories,
                              remove=('headers', 'footers', 'quotes'))
print("Number of training samples:", len(train_set.data))
print("Number of testing samples:", len(test_set.data))

# ====== show some samples ====== #
for i in np.random.choice(np.arange(len(train_set.data)),
                          size=8, replace=False):
  print(doc_preprocessing(train_set.data[i]), '\n')

# ===========================================================================
# Preprocessing the data
# ===========================================================================
dictionary = CountVectorizer(max_df=max_df, min_df=min_df,
                             max_features=n_features,
                             stop_words='english')
