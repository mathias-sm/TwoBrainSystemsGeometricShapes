import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import pickle

from mne import read_source_spaces, spatial_src_adjacency, set_log_file
from mne.stats import spatio_temporal_cluster_1samp_test, permutation_cluster_1samp_test

from scipy.stats.distributions import t
import fire

where = {"IT": [(.0,.332)], "symbolic": [(.136,.452)]}
result = {"IT": [], "symbolic": []}

based = "../../bids_data/derivatives"
basemsm = f"{based}/msm/sub-average/meg/sub-average_task-POGS_"

src = f"{based}/freesurfer/subjects/fsaverage/bem/fsaverage-ico-5-src.fif"
src = read_source_spaces(src)
adjacency = spatial_src_adjacency(src)

morphed = pickle.load(open(f"{basemsm}all+rsa.pkl", "rb"))

for name in where.keys():
    print(name)
    for (tmin, tmax) in where[name]:
        imin, imax = int((tmin + 0.1) / 0.004), int((tmax + 0.1) / 0.004)
        X = np.array([x.data[:, imin:imax] for x in morphed[name]])
        X = np.mean(X, axis=2)
        clu = permutation_cluster_1samp_test(
            X,
            adjacency=adjacency,
            check_disjoint=True,
            tail=1,
            n_permutations=2**10,
            n_jobs=-1,
            verbose=True,
        )
        result[name].append(clu)

pickle.dump((where, result), open(f"clusters.pkl", "wb"))
