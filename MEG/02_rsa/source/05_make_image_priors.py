import numpy as np
import pickle
import fire

import matplotlib.pyplot as plt

based = "../../bids_data/derivatives"

where, result = pickle.load(open("clusters.pkl", "rb"))

morphed = pickle.load(open(f"{based}/msm/sub-average/meg/sub-average_task-POGS_all+rsa.pkl", "rb"))

def make_picture(stc, fname):
    pos_max = 0.35
    brain = stc.plot(
            hemi="split",
            views=["lateral", "medial"],
            subjects_dir=f"{based}/freesurfer/subjects/",
            size=(1500,1000),
            background='white',
            clim={'kind': 'value', "lims": [0.0, pos_max/2, pos_max]},
            surface="white",
            brain_kwargs=dict(offscreen=True, show=False))
    img = brain.screenshot()
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    fig.savefig(fname+".png", dpi=600)


for name, whens in where.items():
    avg = np.array(morphed[name]).mean(axis=0)
    for idx, (tmin, tmax) in enumerate(whens):
        l_avg = avg.copy()
        l_avg.crop(tmin, tmax)
        l_avg = l_avg.mean()
        imin, imax = int((tmin + 0.1) / 0.004), int((tmax + 0.1) / 0.004)
        _, cl, cl_pv, _ = result[name][idx]
        t = .05
        print(cl_pv)
        ll_avg = l_avg.copy()
        ll_avg.data = 0 * ll_avg.data
        for idxp, p in enumerate(cl_pv):
            if p < t:
                filt = cl[idxp][0]
                ll_avg.data[filt] = l_avg.data[filt]
        print(np.max(ll_avg.data))
        make_picture(ll_avg, f"figs/{name}_{tmin}_{tmax}_thresh-{t}")
