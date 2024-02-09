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

from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.signal import savgol_filter

from rsatoolbox.rdm.combine import from_partials
from rsatoolbox.rdm import compare

import fire

color = ["#f1a340", "#998ec3"]
subs = [f"sub-{(i+1):02}" for i in range(20)]

ordered = ["rectangle", "square", "isoTrapezoid", "parallelogram", "losange", "kite", "rightKite", "rustedHinge", "hinge", "trapezoid", "random"]
index_by_alpha = [6, 9, 1, 4, 3, 2, 7, 8, 0, 10, 5]
pattern_descriptors={'shape': ordered, 'index': index_by_alpha}

for smooth in [False, True]:

    def compute_one(s):
        e = mne.read_epochs(f"../../bids_data/derivatives/msm/{s}/meg/{s}_task-POGS_proc-clean+meta_epo.fif")["reference"].as_type("mag").pick_types(meg="mag")
        m = e.metadata
        data = e.get_data()
        if smooth:
            print("smoothed")
            data = savgol_filter(data, 25, 2, axis=2)
        else:
            print("not smoothed")
        data = rsatoolbox.data.TemporalDataset(
                data,
                descriptors = {"subj": s},
                obs_descriptors = {"shape": m["base_shape"], "run": m["run"]},
                channel_descriptors = {"channels": e.ch_names},
                time_descriptors = {"time": e.times})
        del e
        return data

    all_data = Parallel(n_jobs=10)(delayed(compute_one)(s) for s in subs)

    cm = "crossnobis"

    def calc_movie(d):
        return rsatoolbox.rdm.calc_rdm_movie(
                d,
                method=cm,
                descriptor="shape",
                cv_descriptor="run"
            )

    rdms_movie = Parallel(n_jobs=10)(delayed(calc_movie)(d) for d in all_data)
    rdms_movie = from_partials(rdms_movie, descriptor="shape")
    rdms_movie.sort_by(shape=ordered)

    fname = f"./all_rdms/rdms_{cm}_"
    if smooth:
        fname += "smooth"
    else:
        fname += "unsmooth"
    fname += ".pkl"
    print(f"Saving to {fname}")
    pickle.dump(rdms_movie, open(fname, "wb"))
