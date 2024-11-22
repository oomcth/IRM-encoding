import os
import numpy as np
from nilearn import plotting
from nilearn.image import load_img
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import glob
import joblib
from sklearn.manifold import TSNE
import time
from sklearn.linear_model import Ridge
from nilearn.glm.first_level import compute_regressor
import argparse
from sklearn.metrics import pairwise_distances
import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir, standardize
from tqdm import tqdm
from nilearn import plotting
from nilearn.image import smooth_img, swap_img_hemispheres
from nilearn.input_data import NiftiMasker
import nibabel as nib
from scipy.stats import skew
from scipy.stats import norm, kurtosis
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import gudhi
from ripser import ripser
from persim import plot_diagrams
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import gudhi as gd
from sklearn.manifold import LocallyLinearEmbedding
from skdim.id import DANCo


def estimate_intrinsic_dimension_danco(X):
    danco = DANCo()
    dimension_intrinseque = danco.fit_transform(X)
    print(dimension_intrinseque)
    return dimension_intrinseque


# Fast Maximum Likelihood Estimation (MLE)
def estimate_intrinsic_dimension_danco(X, k=10):
    print(X.shape)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]
    # print(np.array(distances).shape)
    log_dist_ratios = np.log(distances[:, -1] / distances[:, 0])
    id_mle = (1 / (k - 1)) * np.sum(log_dist_ratios)
    print(np.mean(id_mle))
    return None


def custom_hrf(tr, oversampling=16, time_length=32, onset=0.):
    """ Generate a simple custom HRF using a Gaussian model """
    dt = tr / oversampling
    time_stamps = np.arange(0, time_length, dt)
    hrf = np.exp(-0.5 * ((time_stamps - onset - 5) ** 2) / 1.0)  # Peak at 5s
    hrf -= 0.2 * np.exp(-0.5 * ((time_stamps - onset - 15) ** 2) / 1.0)  # Undershoot at 15s
    hrf /= hrf.sum()  # Normalize to have a sum of 1
    return hrf


def plot_distribution(data):
    print(data.shape)

    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    sns.kdeplot(data[:, 0], fill=True)
    # sns.kdeplot(data[:, 25306], fill=True)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2',
                    help='gpt2 variants or opt variants')
args = parser.parse_args()

model_name = "activationsgoogle/gemma-2-9b-it"
# model_name = "onehot_emb"
# model_name = "EleutherAI/pythia-410m-step:143000emotions"

activation_folder = lbl.llms_activations
output_folder = lbl.llms_brain_correlations
make_dir(output_folder)

print('\n fitting model {}'.format(model_name))

n_runs = lbl.n_runs
t_r = lbl.t_r

hrf_model = 'glover + derivative + dispersion'
hrf_model = 'glover'

loc = lbl.fmri_data_avg_subject
# loc = "lpp_fr_average_subject"

# fMRI
fmri_runs = []
for run in range(n_runs):
    filename = os.path.join(loc, 'average_subject_run-{}.gz'.format(run))
    with open(filename, 'rb') as f:
        fmri_runs.append(joblib.load(f))

# number of scans per runs
n_scans_runs = [fmri_run.shape[0] for fmri_run in fmri_runs]

n_voxels = fmri_runs[0].shape[1]

# trim first 20 first seconds, ie 10 first elements with a tr of 2s
# same for last 20 seconds
for k in range(n_runs):
    fmri_runs[k] = fmri_runs[k][10:-10]

for run in range(n_runs):
    fmri_runs[run] = standardize(fmri_runs[run])

# LLM
filename = os.path.join(activation_folder, '{}.gz'.format(model_name))
with open(filename, 'rb') as f:
    activations_runs_layers_words_neurons = joblib.load(f)

# corresponding onsets/offsets
filename = os.path.join(activation_folder, 'onsets_offsets.gz')
with open(filename, 'rb') as f:
    runs_onsets_offsets = joblib.load(f)

runs_onsets = []
runs_offsets = []

for run in range(n_runs):
    runs_onsets.append(runs_onsets_offsets[run][0])
    runs_offsets.append(runs_onsets_offsets[run][1])

n_layers = len(activations_runs_layers_words_neurons[0])


def find_overlaps(onsets, offsets):
    # Combiner les onsets et offsets en une liste de tuples
    intervals = list(zip(onsets, offsets))

    # Trier les intervalles en fonction du "onset"
    intervals.sort(key=lambda x: x[0])

    overlaps = []
    for i in range(len(intervals) - 1):
        current_onset, current_offset = intervals[i]
        next_onset, next_offset = intervals[i + 1]
        # Vérifier si l'intervalle courant chevauche l'intervalle suivant
        if current_offset > next_onset:
            overlaps.append((intervals[i], intervals[i + 1]))
    return overlaps


def compute_regressor_from_activations(activations, onsets, offsets, frame_times):
    # activations: n_timesteps x n_neurons
    # print(find_overlaps(onsets, offsets))
    # dur = 1 + 0 * np.array(1 + np.array(durations))
    # dur = np.random.randn(*dur.shape)
    # activations = np.tile(dur, (300, 1)).T
    offsets[:-1] = onsets[1:]
    durations = offsets - onsets
    # activations = np.random.normal(0, 1, activations.shape)
    # norms = np.linalg.norm(activations)
    # mean_norm = np.linalg.norm(np.mean(activations[0]))
    # median_norm = np.median(norms)
    # print("Moyenne des normes :", mean_norm)
    # print("Médiane des normes :", median_norm)
    # input()


    # print(len(durations))
    # print(np.array(activations).shape)
    # input()
    nn_signals = []
    for amplitudes in tqdm(activations.T):
        exp_condition = np.array((onsets, durations, amplitudes))
        signal, name = compute_regressor(
                    exp_condition, hrf_model, frame_times)
        nn_signals.append(signal[:, 0])
    nn_signals = np.array(nn_signals).T
    return nn_signals


def compute_homology(points, max_dimension=1):
    rips_complex = gudhi.RipsComplex(points=points)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    return simplex_tree.compute_persistence()


def compute_cohomology(points, max_dimension=1):
    rips_complex = gudhi.RipsComplex(points=points)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    return simplex_tree.compute_persistent_cohomology()


def compute_persistent_homology_with_tsne(points, max_dimension=1, perplexity=30, n_components=2):
    # Réduction de dimension avec t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000)
    reduced_points = tsne.fit_transform(points)
    plt.scatter(reduced_points[:, 0], reduced_points[:, 1], alpha=0.5)
    plt.show()
    # Calcul de l'homologie persistante
    rips_complex = gudhi.RipsComplex(points=reduced_points, max_edge_length=10)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    persistence = simplex_tree.compute_persistence()

    return persistence, reduced_points


def plot_persistence_diagram(persistence, max_dimension=1):
    gudhi.plot_persistence_diagram(persistence, legend=True)
    plt.title("Diagramme de persistance (Cohomologie)")
    plt.xlabel("Naissance")
    plt.ylabel("Mort")
    plt.show()


def plot_persistence_barcode(persistence, max_dimension=1):
    gudhi.plot_persistence_barcode(persistence, legend=True)
    plt.title("Code-barres de persistance (Cohomologie)")
    plt.xlabel("Paramètre de filtration")
    plt.ylabel("Indice des caractéristiques")
    plt.show()


def reduce_dimension(points, n_components=10):
    # X = points
    # n = X.shape[0]
    
    # # Calcul des distances par paires
    # distances = pairwise_distances(X)
    
    # # Tri des distances par ordre croissant pour chaque point
    # sorted_distances = np.sort(distances, axis=1)
    
    # # Récupération des distances des k voisins les plus proches
    # knn_distances = sorted_distances[:, 1:k+1]  # On ignore la distance à soi-même (0)
    
    # # Calcule les ratios généralisés
    # ratios = np.zeros((n, k-1))
    # for i in range(1, k):
    #     ratios[:, i-1] = knn_distances[:, i] / knn_distances[:, 0]  # Ratio entre le i-ème voisin et le 1er
    
    # # Moyenne des log-ratios généralisés
    # log_ratios = np.mean(np.log(ratios), axis=0)
    
    # # Estimation de la dimension intrinsèque selon la méthode du papier
    # dimension_intrinseque = (k - 2) / (2 * np.mean(log_ratios))

    # return dimension_intrinseque
    # pca = PCA()
    # pca.fit_transform(points)
    # explained_variance = pca.explained_variance_ratio_
    # cumulative_variance = explained_variance.cumsum()
    # dimension_intrinseque = np.argmax(cumulative_variance >= 0.95) + 1
    # cutoff = 0.9
    # cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    # k = np.argmax(cumulative_variance >= cutoff) + 1
    # lambdas = pca.explained_variance_
    # lambdas_k = lambdas[:k]
    # PR = (np.sum(lambdas_k) ** 2) / np.sum(lambdas_k ** 2)
    # print('PR: ', PR)
    # # print(cumulative_variance[:10])
    # print("Dimension intrinsèque estimée PCA:", dimension_intrinseque)
    # plt.plot(cumulative_variance)
    # plt.show()

    # nbrs = NearestNeighbors(n_neighbors=10).fit(points)
    # distances, indices = nbrs.kneighbors(points)
    # log_distances = np.log(distances[:, 1:])
    # mean_log_distance = np.mean(log_distances, axis=1)
    # dimension_intrinseque_estimee = np.mean(mean_log_distance)
    # print(f"Dimension intrinsèque estimée (basée sur les voisins proches): {dimension_intrinseque_estimee}")

    # pca = PCA(n_components=n_components)
    # return pca.fit_transform(points)
    return None


alphas = np.logspace(2, 7, 16)
np.random.seed = 299
for idx_layer in tqdm(range(n_layers)):
    # print('='*62)
    # print('layer {}'.format(idx_layer))

    # nifti_masker = NiftiMasker(mask_img='mask_lpp_en.nii.gz')
    # nifti_masker.fit()

    # roi_names = ['TP', 'aSTS', 'pSTS', 'AG_TPJ', 'BA44', 'BA45', 'BA47']
    # n_rois = len(roi_names)
    # folder_mask = 'roi_masks'

    # roi_list = [os.path.join(folder_mask, '{}.nii.gz'.format(roi_name)) for roi_name in roi_names]
    # roi_list_t = [swap_img_hemispheres(roi_mask) for roi_mask in roi_list]
    # rois_t = nifti_masker.transform(roi_list_t + roi_list)
    # idx_rois = [np.flatnonzero(roi == 1.0) for roi in rois_t]
    # rois_left = nifti_masker.transform(roi_list)
    # idx_rois_left = [np.flatnonzero(roi == 1.0) for roi in rois_left]

    # rois_right = nifti_masker.transform(roi_list_t)
    # idx_rois_right = [np.flatnonzero(roi == 1.0) for roi in rois_right]
    # y_train = np.vstack([fmri_runs[run_train] for run_train in range(len(fmri_runs))])
    # points = np.array(y_train)
    points = activations_runs_layers_words_neurons[0][idx_layer]
    points = np.array(points)
    # activations_words_neurons = np.array(activations_runs_layers_words_neurons[0][idx_layer])  # words x n_neurons
    # onsets = runs_onsets[0]
    # offsets = runs_offsets[0]
    # frame_times = np.arange(n_scans_runs[0]) * t_r + .5 * t_r
    # points = compute_regressor_from_activations(activations_words_neurons, onsets, offsets, frame_times)

    # points = reduce_dimension(points, 2)
    estimate_intrinsic_dimension_danco(points)
    continue
    # plt.scatter(points[:, 0], points[:, 1])
    # plt.show()
    # for a in idx_rois_right:
    #     points = np.array(y_train[:, a])
    #     print(points.shape)
    #     diagrams = ripser(points, maxdim=2)['dgms']
    #     plot_diagrams(diagrams, show=True)
    # for a in idx_rois_left:
    #     points = np.array(y_train[:, a])
    #     print(points.shape)
    #     points = reduce_dimension(points, 4)
    #     print(points.shape)
    #     diagrams = ripser(points, maxdim=1)['dgms']
    #     plot_diagrams(diagrams, show=True)
    # for a in idx_rois_left:
        # persistence = compute_cohomology(points)
        # print('cohomologie: ', persistence)
        # plot_persistence_diagram(persistence)
        # plot_persistence_barcode(persistence)
        # persistence = compute_persistent_homology_with_tsne(points)
        # print('homologie: ', persistence)
        # plot_persistence_diagram(persistence)
        # plot_persistence_barcode(persistence)

    # print(len(activations_runs_layers_words_neurons))
    # print(len(activations_runs_layers_words_neurons[0]))
    # print(len(activations_runs_layers_words_neurons[0][0]))
    # y_train = np.vstack([activations_runs_layers_words_neurons[run_train] for run_train in range(len(fmri_runs))])

    regressors_runs = []
    for run in range([n_runs[0]]):
        H0 = []
        H1 = []
        print(len(activations_runs_layers_words_neurons[0]))
        # for layer in range(len(activations_runs_layers_words_neurons[0])):
        activations_words_neurons = np.array(activations_runs_layers_words_neurons[run][idx_layer])  # words x n_neurons
        onsets = runs_onsets[run]
        offsets = runs_offsets[run]
        frame_times = np.arange(n_scans_runs[run]) * t_r + .5 * t_r
        regresssor_run = compute_regressor_from_activations(activations_words_neurons, onsets, offsets, frame_times)
        regresssor_run = regresssor_run[10:-10]  # trim
        regresssor_run = standardize(regresssor_run)

            # points = np.array(regresssor_run)
            # print(points.shape)
            # # points = reduce_dimension(points, 4)
            # print(points.shape)
            # diagrams = ripser(points, maxdim=1)['dgms']
            # plot_diagrams(diagrams, show=False)

            # H0_diagram = diagrams[0]
            # H1_diagram = diagrams[1]

            # # Calcul des durées de vie
            # H0_persistence = (H1_diagram[:, 1] - H1_diagram[:, 0]) / H1_diagram[:, 0]
            # H1_persistence = H1_diagram[:, 1] - H1_diagram[:, 0]

            # # Durée de vie maximale pour H_0 et H_1
            # max_H0_persistence = np.max(H0_persistence)
            # H0.append(max_H0_persistence)
            # max_H1_persistence = np.max(H1_persistence)
            # H1.append(max_H1_persistence)

            # print(f"Durée de vie maximale pour H_0 : {max_H0_persistence}")
            # print(f"Durée de vie maximale pour H_1 : {max_H1_persistence}")

            # plt.savefig("imageshomologie/PCA4run1-layer" + str(layer) + "nH1:" + str(max_H0_persistence)[:4] + "h1:" + str(max_H1_persistence)[:4] + ".png")
            # plt.close()

        regressors_runs.append(regresssor_run)
        # print(H0)
        # print(H1)

    corr_runs = []
    coef_runs = []

    x_train = np.vstack([regressors_runs[run_train] for run_train in range(len(regressors_runs))])

    points = np.array(x_train)
    print(points.shape)
    points = reduce_dimension(points, 2)
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()

    print(points.shape)
    diagrams = ripser(points, maxdim=1)['dgms']
    plot_diagrams(diagrams, show=False)

    H0_diagram = diagrams[0]
    H1_diagram = diagrams[1]

    # Calcul des durées de vie
    H0_persistence = (H1_diagram[:, 1] - H1_diagram[:, 0]) / H1_diagram[:, 0]
    H1_persistence = H1_diagram[:, 1] - H1_diagram[:, 0]

    # Durée de vie maximale pour H_0 et H_1
    max_H0_persistence = np.max(H0_persistence)
    H0.append(max_H0_persistence)
    max_H1_persistence = np.max(H1_persistence)
    H1.append(max_H1_persistence)

    print(f"Durée de vie maximale pour H_0 : {max_H0_persistence}")
    print(f"Durée de vie maximale pour H_1 : {max_H1_persistence}")

    plt.savefig("imageshomologie/PCA4runall-layer" + str(idx_layer) + "nH1:" + str(max_H0_persistence)[:4] + "h1:" + str(max_H1_persistence)[:4] + ".png")
    plt.close()
    continue
print(H0)
print(H1)
for _ in range(0):
    for run_test in range(n_runs * 0):
        tic = time.time()

        runs_train = np.setdiff1d(np.arange(n_runs), run_test)
        x_train = np.vstack([regressors_runs[run_train] for run_train in runs_train])

        points = np.array(x_train)
        print(points.shape)
        points = reduce_dimension(points, 4)
        print(points.shape)
        diagrams = ripser(points, maxdim=1)['dgms']
        plot_diagrams(diagrams, show=False)

        H0_diagram = diagrams[0]
        H1_diagram = diagrams[1]

        # Calcul des durées de vie
        H0_persistence = (H1_diagram[:, 1] - H1_diagram[:, 0]) / H1_diagram[:, 0]
        H1_persistence = H1_diagram[:, 1] - H1_diagram[:, 0]

        # Durée de vie maximale pour H_0 et H_1
        max_H0_persistence = np.max(H0_persistence)
        H0.append(max_H0_persistence)
        max_H1_persistence = np.max(H1_persistence)
        H1.append(max_H1_persistence)

        print(f"Durée de vie maximale pour H_0 : {max_H0_persistence}")
        print(f"Durée de vie maximale pour H_1 : {max_H1_persistence}")

        plt.savefig("imageshomologie/PCA4runall-layer" + str(layer) + "nH1:" + str(max_H0_persistence)[:4] + "h1:" + str(max_H1_persistence)[:4] + ".png")
        plt.close()

        x_test = regressors_runs[run_test]
        y_train = np.vstack([fmri_runs[run_train] for run_train in runs_train])

        y_test = fmri_runs[run_test]

        ############ start nested CV
        # leave another run apart as a validation test
        run_val = runs_train[0]
        runs_train_val = np.setdiff1d(runs_train, run_val)
        x_train_val = np.vstack([regressors_runs[run_train_val] for run_train_val in runs_train_val])
        x_val = regressors_runs[run_val]
        y_train_val = np.vstack([fmri_runs[run_train] for run_train in runs_train_val])
        y_val = fmri_runs[run_val]

        corr_val = []

        alpha = 46415.888336127726
        ############ end nested CV

        model = Ridge(alpha=alpha, fit_intercept=True)
        # plot_distribution(x_train)
        # plot_distribution(y_train)
        # x_train = np.random.lognormal(mean=1, sigma=1, size=x_train.shape)
        # y_train = np.random.lognormal(mean=1, sigma=1, size=y_train.shape)

        model.fit(x_train, y_train)
        # x_test = np.random.lognormal(mean=1, sigma=1, size=x_test.shape)
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        y_pred = model.predict(x_test)
        # y_pred  = np.random.lognormal(mean=1, sigma=1, size=y_test.shape)
        # plt.scatter(y_pred[:, 25306], y_test[:, 25306])
        print(y_pred.shape)

        # correlation_matrix = np.corrcoef(y_test)
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 6))
        # plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
        # plt.colorbar()
        # plt.title("Correlation Matrix")
        # plt.show()
        # corr = np.corrcoef(y_test, y_pred)
        # plt.figure(figsize=(8, 6))
        # plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
        # plt.colorbar()
        # plt.title("Correlation Matrix")
        # plt.show()

        # corr_tmp = np.array([np.var([np.dot(y_test[j, i].T, y_pred[j, i]) for j in range(y_pred.shape[0])]) / np.corrcoef(y_pred[:, i], y_test[:, i])[0, 1] if np.corrcoef(y_pred[:, i], y_test[:, i])[0, 1] > 0.05 else 0 for i in range(n_voxels)])
        corr_tmp = np.array([spearmanr(y_pred[:, i], y_test[:, i])[0] for i in tqdm(range(n_voxels))])
        # corr_tmp = np.array([skew(y_test[:, i]) for i in range(n_voxels)])
        corr_runs.append(corr_tmp)
        # print(np.max(corr_tmp))
        # corr_voxels = np.array(corr_tmp)
        # imgtmp = nifti_masker.inverse_transform(corr_tmp)
        # plotting.plot_img_on_surf(imgtmp,
        #                           surf_mesh='fsaverage5',
        #                           views=['lateral'],
        #                           hemispheres=['left', 'right'],
        #                           vmin=np.min(corr_tmp), vmax=np.max(corr_tmp),
        #                           cmap='Spectral_r',
        #                           symmetric_cbar=False,
        #                           cbar_tick_format='%.2f',
        #                           colorbar=True,
        #                           title="x")
        # plt.show()
        toc = time.time()

        print('run {}'.format(run_test), '\t', 'mean = {:.03f}'.format(np.mean(corr_tmp)), '\t',
              'max = {:.03f}'.format(np.max(corr_tmp)), '\t',
              'time elapsed = {:.03f}'.format(toc-tic))
    nifti_masker = NiftiMasker(mask_img='mask_lpp_en.nii.gz')
    nifti_masker.fit()
    # corr_voxels = np.array(corr_runs)
    # imgtmp = nifti_masker.inverse_transform(np.mean(corr_voxels, axis=0))
    # plotting.plot_img_on_surf(imgtmp,
    #                           surf_mesh='fsaverage5',
    #                           views=['lateral'],
    #                           hemispheres=['left', 'right'],
    #                           vmin=0, vmax=0.4,
    #                           cmap='Spectral_r',
    #                           symmetric_cbar=False,
    #                           cbar_tick_format='%.2f',
    #                           colorbar=True,
    #                           title="corr rdemb 1024d corrected biais with onset[1:]=offset[:-1]")
    # plt.show()

    print('---->', '\t' 'mean corr = {:.03f}'.format(np.mean(corr_runs)))

    filename = os.path.join(output_folder, 'autre{}_layer-{}_corr.gz'.format(model_name, idx_layer))
    with open(filename, 'wb') as f:
        joblib.dump(np.mean(corr_runs, axis=0), f, compress=4)
