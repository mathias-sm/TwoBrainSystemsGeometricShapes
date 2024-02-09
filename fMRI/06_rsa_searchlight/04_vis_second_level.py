import glob
import os
import scipy

from nilearn.image import load_img, new_img_like, smooth_img

from nilearn.plotting import view_img, view_img_on_surf
from nilearn.glm.thresholding import threshold_stats_img

import numpy as np
import pandas as pd

import fire

from nilearn.glm.second_level import SecondLevelModel
base = "../bids_dataset/derivatives/"

def main(task, theory, pop, method="crossnobis", eval="corr_cov"):

    sub_folders = None
    if pop == "adults":
        sub_folders = glob.glob(f"{base}rsa/sub-2*/")
    else:
        sub_folders = glob.glob(f"{base}rsa/sub-3*/")

    sub_labels = [os.path.basename(s[:-1]).split("-")[1] for s in sub_folders]
    sub_labels = sorted([int(x) for x in sub_labels if x != "308"])

    fc = []
    for sub_i in sub_labels:
        # sub-204_task-geometryHard_pred-IT_method-correlation_eval-corr_corr.nii.gz
        sub = f"sub-{sub_i}"
        img = load_img(f"{base}rsa/{sub}/func/{sub}_task-{task}_pred-{theory}_method-{method}_eval-{eval}_corr.nii.gz")
        fc.append(img)

    second_level_model = SecondLevelModel(smoothing_fwhm=8.0, n_jobs=-1)

    design_matrix = pd.DataFrame(np.ones((len(fc))), columns=["intercept"])
    second_level_model = second_level_model.fit(fc, design_matrix=design_matrix)

    ref_tmap = second_level_model.compute_contrast(second_level_contrast="intercept")

    _, ref_t = threshold_stats_img(ref_tmap, alpha=.001, two_sided=False)

    title = f"correlation w/ {theory} model on task={task}, pop={pop}"
    page = view_img(ref_tmap, threshold=ref_t, title=title)
    page.save_as_html(f"reports/rsa/rsa-avg_task-{task}_theory-{theory}_pop-{pop}_method-{method}_eval-{eval}.html")

    # page = view_img_on_surf(ref_tmap, threshold=ref_t, title=title)
    # page.save_as_html(f"reports/rsa/rsa-avg_task-{task}_theory-{theory}_pop-{pop}_method-{method}_eval-{eval}_surf.html")


if __name__ == "__main__":
    for theory in ["IT", "symbolic"]:
        for pop in ["adults", "kids"]:
            main("geometry", theory, pop)
        main("geometryHard", theory, "adults")

