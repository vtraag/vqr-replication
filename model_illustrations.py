#%% Import libraries

from scipy import stats
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from common import lognormal

#%% Create figure output directory

figure_dir = Path('.') / '..' / 'figures'
figure_dir.mkdir(parents=True, exist_ok=True)

#%% Draw lognormal distribution for prediction of reviewer scores

sigma = 1.3

dist = lognormal(-sigma**2/2, sigma)

K = 28
cutoffs = [dist.ppf((i+1)/K) for i in range(K - 2)]

x = np.linspace(0, 3.5, 1000)

sns.set_style('white')
sns.set_palette('Set1')

plt.plot(x, dist.pdf(x))

for i, c in enumerate(cutoffs):
    plt.axvline(c, color='lightgray', linewidth=0.1)
    if i > 22:
        plt.text(x=c+0.1, y=1, s=f'{i+5}', color='lightgray')

plt.xlim(0, 3.5)
plt.ylim(0, 1.7)

plt.xlabel('Evaluation score')
plt.ylabel('Probability density')
plt.savefig(figure_dir / 'reviewer_illustration.pdf', bbox_inches='tight')
#%% Illustration of citation and review scores

sigma_paper_value = 0.4
institutional_value = 0.8

# Institutional distribution of paper values
paper_value_dist = lognormal(institutional_value, sigma_paper_value)

# Actual paper value we use in the illustration
paper_value = 1.5

sigma_review = 0.6
sigma_citation = 0.8

continuous_review_dist = lognormal(np.log(paper_value) - sigma_review**2/2, sigma_review)

review_dist = [0] + [continuous_review_dist.cdf(c) for c in cutoffs] + [1]
review_dist = np.diff(review_dist)

citation_dist = lognormal(np.log(paper_value) - sigma_citation**2/2, sigma_citation)

x = np.linspace(0, 4, 1000)

sns.set_style('white')
sns.set_palette('Set1')

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5,6), constrained_layout=True)

# Plot institutional distribution of paper values
ax = axs[0]

ax.fill_between(x, 0, continuous_review_dist.pdf(x), label='Paper value', alpha=0.5)

ax.set_xlabel('Paper value')
ax.xaxis.set_label_position('top') 
ax.xaxis.tick_top()
ax.set_ylabel('Probability density')
ax.set_yticks([])

ax.set_xlim(0, 3)
ax.set_ylim(0, ax.get_ylim()[1])

ax.axvline(paper_value, color='black')

# Plot individual paper distributions
ytop = 0.8

ax = axs[1]
ax.fill_between(x, 0, continuous_review_dist.pdf(x), label='Continuous review', alpha=0.5)
ax.bar([0] + cutoffs, 
       height=-review_dist, 
       bottom=ytop,
       width=0.7*np.diff([0] + cutoffs + [4]),
       alpha=0.5,
       label='Discrete review')
ax.fill_between(x, 0, citation_dist.pdf(x), label='Citation', alpha=0.5)

ax.axvline(paper_value, color='black')
ax.annotate('Paper value', 
             xy=(paper_value, 1.0), xycoords='data',
             xytext=(0.6, 0.7), textcoords='axes fraction',
             arrowprops=dict(arrowstyle='-',connectionstyle='arc3',color='black'),
             horizontalalignment='left', verticalalignment='center')
con = ConnectionPatch(xyA=(paper_value, 0), coordsA=axs[0].transData,
                      xyB=(paper_value, ytop), coordsB=axs[1].transData,
                      color='black', zorder=0)
fig.add_artist(con)

secax = ax.secondary_xaxis('top')
labels = [f'{i + 5}' if i > 18 else '' for i in range(len(cutoffs))]
secax.set_xticks(cutoffs, labels)
secax.set_xlabel('Review score', 
                 bbox=dict(boxstyle='square,pad=0', fc='white', ec='none'))

ax.set_xlim(0, 3)
ax.set_xlabel('Score')

ax.set_ylabel('Probability density')
ax.set_yticks([])

plt.ylim(0, ytop)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,2,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

plt.savefig(figure_dir / 'model_illustration.pdf', bbox_inches='tight')
