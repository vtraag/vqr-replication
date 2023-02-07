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
cutoffs = [dist.ppf((i+1)/K) for i in range(K - 1)]

x = np.linspace(0, 3.5, 1000)

sns.set_style('white')
sns.set_palette('Set1')

plt.plot(x, dist.pdf(x))

for i, c in enumerate(cutoffs):
    plt.axvline(c, color='lightgray', linewidth=0.1)
    if i > 22:
        plt.text(x=c+0.1, y=1, s=f'{i+3}', color='lightgray')

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

review_prob = [0] + [continuous_review_dist.cdf(c) for c in cutoffs] + [1]
review_prob = np.diff(review_prob)
review_dist = stats.rv_discrete(name='review', values=(np.arange(3, 31), review_prob))

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
width = 1*np.diff([0] + cutoffs + [5])
ax.bar([0] + cutoffs, 
       height=-review_prob, 
       bottom=ytop,
       width=width,
       align='edge',
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
labels = [f'{i + 3}' if i > 19 else '' for i in range(len(review_prob))]
secax.set_xticks(([0] + cutoffs) + 0.5*width, labels)
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
# plt.close()

#%% Illustration of theoretical results using multiple reviewers

sns.set_style('white')
sns.set_palette('Set1')

convolved_review_probs = []
convolved_review_probs.append(review_prob)
for i in range(10):
    convolved_review_probs.append(np.convolve(convolved_review_probs[-1], review_prob))

convolved_scores = []
convolved_review_dists = []
for i, probs in enumerate(convolved_review_probs):
    scores = np.arange(probs.shape[0])/(i+1) + 3
    convolved_scores.append(scores)
    review_dist = stats.rv_discrete(name='review', values=(scores, probs))
    convolved_review_dists.append(review_dist)

scores = np.arange(3, 31)
for i in [0, 2, 4]:
    cdf_scores = convolved_review_dists[i].cdf(scores)
    pmf_scores = np.diff(np.insert(cdf_scores, 0, 0))
    plt.bar(scores, pmf_scores, width=1, alpha=0.5, zorder=4-i, 
            label=f'{i+1} reviewer{"s" if i > 0 else ""}')

plt.legend(loc='best')

plt.xlabel('Reviewer score')
plt.ylabel('Probability')

plt.savefig(figure_dir / 'multiple_reviewers.pdf', bbox_inches='tight')
# plt.close()

#%% Show MAD for multiple reviewers

convolved_MAD = []
mean_review_score = review_dist.mean()
for dist in convolved_review_dists:
    mean_MAD = dist.expect(lambda x: np.abs(x - mean_review_score))
    convolved_MAD.append(mean_MAD)

plt.plot(np.arange(len(convolved_review_dists)) + 1, convolved_MAD, 
         marker='o')
plt.xlabel('Number of reviewers')
plt.ylabel('Expected MAD')

plt.savefig(figure_dir / 'MAD_multiple_reviewers.pdf', bbox_inches='tight')
# %%
