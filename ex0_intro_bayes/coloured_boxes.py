# !pip install tf-nightly tfp-nightly seaborn
import matplotlib
from matplotlib import pyplot as plt

import seaborn

import numpy as np
from scipy.stats import itemfreq

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.enable_eager_execution()

# ===========================================================================
# How do we specify the distribution in tensorflow probability
# ===========================================================================
boxes = tfd.Categorical(probs=[0.2, # p(r)
                               0.2, # p(b)
                               0.6], # p(g)
                        name="DistributionOfBoxes")
fruits = tfd.Mixture(
    cat=boxes,
    components=[
        # in order, the probabilities of: apple, orange, lime
        tfd.Categorical(probs=[0.3, 0.4, 0.3], name="RedBox"),
        tfd.Categorical(probs=[0.5, 0.5, 0.0], name="BlueBox"),
        tfd.Categorical(probs=[0.3, 0.3, 0.4], name="GreenBox"),
    ],
    name="DistributionOfFruits")

print("p(apple)  =", tf.exp(fruits.log_prob(0)))
print("p(orange) =", tf.exp(fruits.log_prob(1)))
print("p(lime)   =", tf.exp(fruits.log_prob(2)))

# ===========================================================================
# Draw sample and plot histogram
# ===========================================================================
n_samples = 10000
samples = fruits.sample(n_samples)
distribution = itemfreq(samples)
plt.bar(distribution[:, 0], distribution[:, 1])
plt.title("Distribution of fruits after picking %d times" % n_samples)
plt.xticks(np.arange(0, 3), fontsize=12)

# ===========================================================================
# Calculate the log-likelihood
# ===========================================================================
samples1 = fruits.sample(10)
samples2 = [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]

llk1 = tf.reduce_sum(fruits.log_prob(samples1))
print("log[p(sample1|model)] =", llk1)

llk2 = tf.reduce_sum(fruits.log_prob(samples2))
print("log[p(sample2|model)] =", llk2)

if llk1 > llk2:
  print('Samples-1 are more likely to be generated from our model')
elif llk1 == llk2:
  print('Both samples are equal likely to be generated from our model')
else:
  print('Samples-2 are more likely to be generated from our model')

# ===========================================================================
# Could we replicate the same results in numpy using sampling
# ===========================================================================
# ====== configuration ====== #
n_samples = 100
np.random.seed(5218)

# ====== Sampling process ====== #
boxes = ['red', 'blue', 'green']
# this might be slow but it is intuitive
chose_boxes = [np.random.choice(boxes, size=None, p=[0.2, 0.2, 0.6])
               for i in range(n_samples)]
