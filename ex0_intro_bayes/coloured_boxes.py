# !pip install tf-nightly tfp-nightly seaborn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import seaborn

import numpy as np
from scipy.stats import itemfreq

# # ===========================================================================
# # How do we specify the distribution in tensorflow probability
# # ===========================================================================
# import tensorflow as tf
# import tensorflow_probability as tfp
# tfd = tfp.distributions

# tf.enable_eager_execution()

# boxes = tfd.Categorical(probs=[0.2, # p(r)
#                                0.2, # p(b)
#                                0.6], # p(g)
#                         name="DistributionOfBoxes")
# fruits = tfd.Mixture(
#     cat=boxes,
#     components=[
#         # in order, the probabilities of: apple, orange, lime
#         tfd.Categorical(probs=[0.3, 0.4, 0.3], name="RedBox"),
#         tfd.Categorical(probs=[0.5, 0.5, 0.0], name="BlueBox"),
#         tfd.Categorical(probs=[0.3, 0.3, 0.4], name="GreenBox"),
#     ],
#     name="DistributionOfFruits")

# print("p(apple)  =", tf.exp(fruits.log_prob(0)))
# print("p(orange) =", tf.exp(fruits.log_prob(1)))
# print("p(lime)   =", tf.exp(fruits.log_prob(2)))

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
seaborn.distplot(chose_boxes)
plt.show()
