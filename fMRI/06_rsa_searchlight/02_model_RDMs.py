import numpy as np
import pickle
import pandas as pd
import fire
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
from nilearn.image import new_img_like, load_img
from rsatoolbox.util.searchlight import evaluate_models_searchlight

theories = ["symbolic", "IT"]
shapes_easy = ["square", "rectangle", "isoTrapezoid", "losange", "hinge", "random"]
shapes_hard = ["square", "rectangle", "isoTrapezoid", "parallelogram", "losange", "kite", "hinge", "random"]
base = "../bids_dataset/derivatives/"

def parse_file(mdl_name, shapes):
    # Read file, then subset rows and columns to match relevant conditions,
    # then take the upper triangular matrix and create a rsatoolbox model with
    # that matrix. Careful about the ordering of the rows/columns!!
    mdl_rdm = pd.read_csv(f"./csv_competing_models/{mdl_name}_sym_diss_mat.csv", index_col=0)
    mdl_rdm = mdl_rdm[[x in shapes for x in mdl_rdm.index]][shapes].values
    triu_mdl_rdm = mdl_rdm[np.triu(np.ones_like(mdl_rdm, dtype=bool), k=1)]
    return(ModelFixed(mdl_name, triu_mdl_rdm))

def main(task, sub, estimate_method="crossnobis", eval_method="corr_cov"):
    shapes = shapes_easy if task == "geometry" else shapes_hard

    # Load brain rdm and model them
    SL_RDM = pickle.load(open(f"{base}rsa/{sub}/func/{sub}_task-{task}_method-{estimate_method}_RDM.pkl", "rb"))
    SL_RDM.sort_by(shape=shapes)  # This is required to align model and data!

    # Note: our models have r=.15 (easy) and r=.03 (hard), so I am not pitting
    # them against one another and instead I'm measuring the goodness of fit of
    # each model per voxel separately.
    # Otherwise one could use a ModelWeighted, however with fit_regress and
    # corr_cov seems to leed to very strongly split-out betas with a
    # distribution of betas that is either -1 or 1 and little inbetween, which
    # then makes a ttest weird
    rdm_models = [parse_file(theory, shapes) for theory in theories]
    eval_scores = evaluate_models_searchlight(SL_RDM, rdm_models, eval_fixed, method=eval_method, n_jobs=-1)
    eval_scores = np.array([e.evaluations[0,:,0] for e in eval_scores])  # Why two dimensions for nothing?!

    # Get brain dimension from glm mask img, used afterward
    ref_img = pickle.load(open(f"{base}nilearn/{sub}/func/{sub}_task-{task}_model-spm_full-false.pkl", "rb")).mask_img
    x, y, z = ref_img.shape

    # Create brain maps of the coefficients for each theory
    for idx_th, theory in enumerate(theories):
        # Grab the correct eval score and fill a brain-shaped array with it
        RDM_brain = np.nan * np.zeros([x * y * z])
        RDM_brain[list(SL_RDM.rdm_descriptors["voxel_index"])] = eval_scores[:,idx_th]
        RDM_brain = RDM_brain.reshape([x, y, z])

        # Make it an image, and save it in a file
        new_img_like(ref_img, RDM_brain).to_filename(f"{base}rsa/{sub}/func/{sub}_task-{task}_pred-{theory}_method-{estimate_method}_eval-{eval_method}_corr.nii.gz")


if __name__ == "__main__":
    fire.Fire(main)
