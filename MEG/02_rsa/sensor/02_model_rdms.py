import mne
import numpy as np
import rsatoolbox
import pandas as pd
import copy

import statsmodels.formula.api as smf

import scipy.stats

import pickle

import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from scipy.ndimage import uniform_filter1d

from rsatoolbox.rdm.combine import from_partials
from rsatoolbox.rdm import compare

import matplotlib
font = {'family' : 'calibri', 'size'   : 9}
plt.rcParams['svg.fonttype'] = 'none'
matplotlib.rc('font', **font)


def pretty_plot(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.tick_params(colors='dimgray')
    ax.xaxis.label.set_color('dimgray')
    ax.yaxis.label.set_color('dimgray')
    try:
        ax.zaxis.label.set_color('dimgray')
    except AttributeError:
        pass
    try:
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    except ValueError:
        pass
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax


color = ["#f1a340", "#998ec3"]
subs = [f"sub-{(i+1):02}" for i in range(20)]

ordered = ["rectangle", "square", "isoTrapezoid", "parallelogram", "losange", "kite", "rightKite", "rustedHinge", "hinge", "trapezoid", "random"]
index_by_alpha = [6, 9, 1, 4, 3, 2, 7, 8, 0, 10, 5]
pattern_descriptors={'shape': ordered, 'index': index_by_alpha}

cm = "crossnobis"
for smooth in ["smooth", "unsmooth"]:
    rdms_movie = pickle.load(open(f"./all_rdms/rdms_{cm}_{smooth}.pkl", "rb"))
    times = rdms_movie.subset('subj', ['sub-01']).rdm_descriptors['time']

    def parse_file(fname):
        behave_diss = pd.read_csv(fname, index_col=0).values
        tri_behave_diss = behave_diss[np.triu(np.ones_like(behave_diss, dtype=bool), k=1)]
        zscored = scipy.stats.zscore(tri_behave_diss)
        rdm = rsatoolbox.rdm.RDMs(zscored, pattern_descriptors=pattern_descriptors)
        return rdm

    symbolic = parse_file("../csv_competing_models/symbolic_sym_diss_mat.csv")
    IT = parse_file("../csv_competing_models/IT_sym_diss_mat.csv")

    model_rdms = copy.deepcopy(symbolic)
    model_rdms.append(copy.deepcopy(IT))
    model_rdms.rdm_descriptors['model_names'] = model_names = ['symbolic', 'IT']
    model = rsatoolbox.model.ModelWeighted('full', model_rdms)

    method = "corr_cov"

    fitted = np.zeros((20, 2, 301))
    for idxs, sub in enumerate(subs):
        ds = rdms_movie.subset('subj', sub)
        for idxt, t in enumerate(times):
            topred = ds.subset('time', [t])
            topred.dissimilarities[0] = scipy.stats.zscore(topred.dissimilarities[0])
            for im, m in enumerate([symbolic, IT]):
                fitted[idxs, im, idxt] = rsatoolbox.rdm.compare(m, topred, method=method)

    plt.close('all')
    plt.figure(figsize=(5, 2))
    plt.axhline(y=0, color='dimgray', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='dimgray', linestyle='-', linewidth=0.5)
    plt.axvline(x=1, color='dimgray', linestyle='-', linewidth=0.5)
    plt.axvline(x=0.8, linewidth=0.5, linestyle=(0, (5, 10)), color='k')
    for i in range(1,-1,-1):
        r_ = fitted[:, i, :]
        mu = np.mean(r_, axis=0)
        se = np.std(r_, axis=0)/np.sqrt(20)
        tmin, tmax = 0, .8
        imin, imax = int((tmin + .1)/.004), int((tmax + .1)/.004)
        offset = .1/.004
        r_for_clust = r_[:,imin:imax+1]
        _, clusters, p_values, _ = mne.stats.permutation_cluster_1samp_test(
                r_for_clust,
                n_permutations=2**13,
                verbose=False,
                tail=1)
        print([i, p_values[p_values < .1]])
        p05 = np.nan * np.zeros((301))
        for x in np.where(p_values < .05)[0]:
            filt = [int(x) for x in clusters[x][0]+offset]
            p05[filt] = mu[filt]
            print((np.array(filt)-offset) * .004)
        p1 = np.nan * np.zeros((301))
        plt.plot(times, mu, color=color[i], linewidth=0.4)
        plt.plot(times, p05, label=model_names[i], color=color[i])
        plt.fill_between(times, p05, np.zeros((301)), alpha=.5, facecolor=color[i])

    plt.xlabel('time')
    plt.ylabel(f'model-data crossnobis similarity')
    plt.legend()
    pretty_plot()
    plt.savefig(f"./figs/lm_{cm}_{method}_{smooth}.png", dpi=600)
    plt.savefig(f"./figs/lm_{cm}_{method}_{smooth}.svg")
