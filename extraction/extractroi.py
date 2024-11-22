from nilearn import datasets, plotting
from nilearn.image import math_img

# Téléchargement d'un atlas avec des régions pertinentes pour les tâches cognitives
# Par exemple, l'atlas Harvard-Oxford
ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

# Chargement de l'atlas
atlas_img = ho_atlas.maps
labels = ho_atlas.labels

# Identification des labels correspondant aux régions d'intérêt
# Cela peut être fait en listant les labels et en les filtrant
for label in labels:
    print(label)

# Extraction des régions d'intérêt
# Par exemple, si on s'intéresse au "Frontal Pole"
roi_img = math_img("img == 1", img=atlas_img)  # Remplacez 1 par l'index du label correspondant

# Visualisation du ROI
plotting.plot_roi(roi_img, title="Frontal Pole")
plotting.show()
