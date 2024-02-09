import numpy as np
import pandas as pd
from scipy.stats import zscore

import pickle
import fire

from joblib import Parallel, delayed

from rsatoolbox.rdm import RDMs
from rsatoolbox.model import ModelWeighted
from rsatoolbox.model.fitter import fit_regress
import rsatoolbox
from mne import read_source_estimate

from rich.progress import track

color = ["#f1a340", "#998ec3"]
subs = [f"sub-{(i+1):02}" for i in range(20)]

ordered = [
    "rectangle",
    "square",
    "isoTrapezoid",
    "parallelogram",
    "losange",
    "kite",
    "rightKite",
    "rustedHinge",
    "hinge",
    "trapezoid",
    "random",
]
index_by_alpha = [6, 9, 1, 4, 3, 2, 7, 8, 0, 10, 5]
pattern_descriptors = {"shape": ordered, "index": index_by_alpha}

def parse_file(fname):
    behave_diss = pd.read_csv(fname, index_col=0).values
    tri_behave_diss = behave_diss[np.triu(np.ones_like(behave_diss, dtype=bool), k=1)]
    zscored = zscore(tri_behave_diss)
    rdm = RDMs(zscored, pattern_descriptors=pattern_descriptors)
    return rdm

symbolic = parse_file("../csv_competing_models/symbolic_sym_diss_mat.csv")
IT = parse_file("../csv_competing_models/IT_sym_diss_mat.csv")

model_rdms = symbolic
model_rdms.append(IT)
model = ModelWeighted("full", model_rdms)


def main(s):
    based = "../../bids_data/derivatives"
    basemne = f"{based}/mne-bids-pipeline/{s}/meg/{s}_task-POGS_"
    basemsm = f"{based}/msm/{s}/meg/{s}_task-POGS_"

    all_rdms = pickle.load(open(f"{basemsm}proc-rsa+crossnobis_rdm.pkl", "rb"))
    times = np.arange(-0.1, 1.1, 0.004)

    def model_one_rdm(rdm):
        fitted = np.zeros((2, len(rdm)))
        for t, lrdm in enumerate(rdm):
            lrdm.dissimilarities[0] = zscore(lrdm.dissimilarities[0])
            for im, m in enumerate([symbolic, IT]):
                fitted[im, t] = rsatoolbox.rdm.compare(m, lrdm, method="corr_cov")[0][0]
        return fitted

    fitted = Parallel(n_jobs=16)(delayed(model_one_rdm)(rdm) for rdm in track(all_rdms))
    fitted = np.array(fitted)

    refstc = f"{basemne}shape+eLORETA+hemi"
    refstc = read_source_estimate(refstc)

    for idx_mdl, name in enumerate(["symbolic", "IT"]):
        this_rtc = refstc.copy()
        this_rtc.data = fitted[:, idx_mdl, :]
        this_rtc.save(f"{basemsm}rsa+eLORETA+{name}", overwrite=True)


if __name__ == "__main__":
    fire.Fire(main)
