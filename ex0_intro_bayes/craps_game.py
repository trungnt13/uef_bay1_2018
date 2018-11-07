# This script simulate the play of "craps" and "roulette"
# game as mentioned in the example of chapter `2.1.8`
# !pip install seaborn
import matplotlib
# un-comment this if your computer doesn't have screen
# matplotlib.use('Agg')

from matplotlib import pyplot as plt
import seaborn

import numpy as np
from scipy.stats import itemfreq

# fixed random seed for reproducibility
np.random.seed(5218)

# ===========================================================================
# Configuration for the experiments
# ===========================================================================
n_samples = 100
p_crap = 0.5 # let call this p(c)
p_roulette = 1 - p_crap # let call this p(r)
selected_word = 11

# ===========================================================================
# The "craps" game
# ===========================================================================
dice1 = np.random.randint(1, 7, size=n_samples, dtype='int32')
dice2 = np.random.randint(1, 7, size=n_samples, dtype='int32')
craps = dice1 + dice2

plt.figure()

ax = plt.subplot(3, 1, 1)
seaborn.distplot(dice1, bins=np.arange(1, 7))
ax.set_title("First dice")

ax = plt.subplot(3, 1, 2)
seaborn.distplot(dice2, bins=np.arange(1, 7))
ax.set_title("Second dice")

ax = plt.subplot(3, 1, 3)
seaborn.distplot(craps, bins=np.arange(1, 13))
ax.set_title("The results of craps game")

# ====== table of joint probability ====== #
count = np.zeros(shape=(6, 6), dtype='int32')

for d1, d2 in zip(dice1, dice2):
  idx1 = d1 - 1
  idx2 = d2 - 1
  count[idx1, idx2] += 1
  count[idx2, idx1] += 1
# print the numerical count matrix
print(count)

# ====== plot it for fun ====== #
plt.figure()
plt.imshow(count, cmap=plt.cm.Blues)
cbar = plt.colorbar()
# adjust the colorbar a bit
cbar.set_ticks(np.linspace(0, np.max(count), num=5).astype("int32"))
plt.xticks(np.arange(6), np.arange(6) + 1)
plt.xlabel("Dice1")
plt.yticks(np.arange(6), np.arange(6) + 1)
plt.ylabel("Dice2")
plt.title("$p(dice1,dice2)$")

# ===========================================================================
# The "roulette" game
# p(w | r)
# ===========================================================================
roulette = np.random.randint(1, 39, size=n_samples, dtype='int32')

plt.figure()
seaborn.distplot(roulette, bins=np.arange(1, 39))
ax = plt.gca()
ax.set_title("The results of roulette game")
ax.set_xticks(np.arange(1, 39, 3, dtype='int32'))

# ===========================================================================
# The distribution of which word was shouted
# for `n_samples` times
# ===========================================================================
# 1 mean "craps" played
# 0 mean it wasn't played, or "roulette" was played
craps_played = np.random.binomial(n=1, p=p_crap, size=n_samples)
words = np.where(craps_played, craps, roulette)
# this return an array of mapping "shouted_word" -> "its_frequency"
words_distribution = itemfreq(words)

# ====== visualize the distribution of words ====== #
plt.figure()
plt.bar(words_distribution[:, 0], words_distribution[:, 1])
plt.title("Distribution of the shouted words, after you hear it %d times" % n_samples)
plt.xticks(np.arange(1, 39), fontsize=6)
# ===========================================================================
# Could we select all the samples of p("crap"|selected_word)
# ===========================================================================
# ====== visualize who played and what results ====== #
# This is very heavy task so we only do it for 100 samples
plt.figure()
# all the "craps"
all_craps_words = [(idx, word)
                   for idx, (is_craps_played, word) in enumerate(zip(craps_played, words))
                   if is_craps_played]
all_craps_words = np.array(all_craps_words)

all_roulette_words = [(idx, word)
                      for idx, (is_craps_played, word) in enumerate(zip(craps_played, words))
                      if not is_craps_played]
all_roulette_words = np.array(all_roulette_words)

plt.bar(all_craps_words[:, 0], all_craps_words[:, 1], color='red', label="Craps")
plt.bar(all_roulette_words[:, 0], all_roulette_words[:, 1], color='blue', label="Roulette")
plt.hlines(y=selected_word, xmin=0, xmax=n_samples, linestyles='-.',
           color='green', linewidth=1.2,
           label="Selected word: %d" % selected_word)
plt.legend() # showing the legend
plt.title("Which game played, and what is the results, after %d times" % n_samples)

who = []
for is_craps_played, w in zip(craps_played, words):
  if w == selected_word: # i.e. given word = `selected_word`
    who.append(is_craps_played)

print("p(crap|%d)=" % selected_word, np.sum(who) / len(who))
# ===========================================================================
# Saving the figure
# ===========================================================================
plt.show()
