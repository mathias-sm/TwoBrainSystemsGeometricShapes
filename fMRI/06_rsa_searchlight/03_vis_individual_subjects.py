import numpy as np
from nilearn.image import load_img
import fire
from nilearn.plotting import view_img
from rich.progress import track
import matplotlib.pyplot as plt

base = "../bids_dataset/derivatives/"

all_conds = [("geometry", "sub-204"), ("geometry", "sub-205"), ("geometry", "sub-206"), ("geometry", "sub-207"), ("geometry", "sub-208"), ("geometry", "sub-209"), ("geometry", "sub-210"), ("geometry", "sub-211"), ("geometry", "sub-212"), ("geometry", "sub-213"), ("geometry", "sub-214"), ("geometry", "sub-215"), ("geometry", "sub-216"), ("geometry", "sub-217"), ("geometry", "sub-218"), ("geometry", "sub-219"), ("geometry", "sub-220"), ("geometry", "sub-221"), ("geometry", "sub-222"), ("geometry", "sub-223"), ("geometryHard", "sub-204"), ("geometryHard", "sub-205"), ("geometryHard", "sub-206"), ("geometryHard", "sub-207"), ("geometryHard", "sub-208"), ("geometryHard", "sub-209"), ("geometryHard", "sub-210"), ("geometryHard", "sub-211"), ("geometryHard", "sub-212"), ("geometryHard", "sub-213"), ("geometryHard", "sub-214"), ("geometryHard", "sub-215"), ("geometryHard", "sub-216"), ("geometryHard", "sub-217"), ("geometryHard", "sub-218"), ("geometryHard", "sub-219"), ("geometryHard", "sub-220"), ("geometryHard", "sub-221"), ("geometryHard", "sub-222"), ("geometryHard", "sub-223"), ("geometry", "sub-301"), ("geometry", "sub-302"), ("geometry", "sub-303"), ("geometry", "sub-304"), ("geometry", "sub-305"), ("geometry", "sub-306"), ("geometry", "sub-307"), ("geometry", "sub-309"), ("geometry", "sub-310"), ("geometry", "sub-312"), ("geometry", "sub-314"), ("geometry", "sub-315"), ("geometry", "sub-317"), ("geometry", "sub-318"), ("geometry", "sub-320"), ("geometry", "sub-321"), ("geometry", "sub-322"), ("geometry", "sub-323"), ("geometry", "sub-325")]

def plot_one(task, sub, estimate_method="crossnobis", eval_method="corr_cov"):
    anat = load_img(f"{base}fmriprep/{sub}/anat/{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz", "rb")
    for theory in ["IT", "symbolic"]:
        plot_img = load_img(f"{base}rsa/{sub}/func/{sub}_task-{task}_pred-{theory}_method-{estimate_method}_eval-{eval_method}_corr.nii.gz")
        threshold = np.nanpercentile(plot_img.get_fdata(), 99)
        page = view_img(plot_img, threshold=threshold, title=f"correlation w/ {theory} model, sub={sub}", bg_img=anat, vmax=1)
        page.save_as_html(f"reports/rsa/surf_{sub}_task-{task}_theory-{theory}.html")
        plt.hist(plot_img.get_fdata().reshape(-1), bins=200)
        plt.axvline(0, color="k")
        plt.savefig(f"reports/rsa/surf_{sub}_task-{task}_theory-{theory}_histogram.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    # fire.Fire(plot_one)
    for task, sub in track(all_conds):
        plot_one(task, sub)
