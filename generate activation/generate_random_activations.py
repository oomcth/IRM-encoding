import numpy as np
import pandas as pd
import os
import joblib
import argparse

import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir

output_folder = lbl.llms_activations
make_dir(output_folder)

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='embedding',
                    help='embedding or vector')
parser.add_argument('--n_dims', type=int, default='300',
                    help='number of dimensions')
parser.add_argument('--seed', type=int, default='1',
                    help='seed')
args = parser.parse_args()

random_type = args.type
n_dims = args.n_dims
seed = args.seed

np.random.seed(seed)

model_name = 'plusidee'.format(random_type, n_dims, seed)

filename = os.path.join(lbl.annotation_folder, 'lppEN_word_information.csv')
df_word_onsets = pd.read_csv(filename)

df_word_onsets = df_word_onsets.drop([3919,6775,6781])
# 3919: adhoc removal of repeated line with typo
# 6775: mismatch with full text

word_list_runs = []
onsets_offsets_runs = []
for run in range(lbl.n_runs):
    df_word_onsets_run = df_word_onsets[df_word_onsets.section==(run+1)]
    word_list_tmp = df_word_onsets_run.word.to_numpy()
    onsets_tmp = df_word_onsets_run.onset.to_numpy()
    offsets_tmp = df_word_onsets_run.offset.to_numpy()
    
    word_list = []
    onsets = []
    offsets = []
    
    for idx_word, (word, onset, offset) in enumerate(zip(word_list_tmp, onsets_tmp, offsets_tmp)):
        if isinstance(word, str):
            word_list.append(word)
            onsets.append(onset)
            offsets.append(offset)
            
    onsets_offsets_runs.append((np.array(onsets), np.array(offsets)))
    word_list_runs.append(word_list)

runs_words_activations = []

if random_type == 'vector':
    for run in range(lbl.n_runs):
        words_activations = [np.random.lognormal(1, 1, n_dims) for _ in range(len(word_list_runs[run]))]
        runs_words_activations.append([words_activations])
elif random_type == 'embedding':

    all_words = set()
    word_freq = {}  # Dictionnaire pour stocker les fréquences des mots

    for run in range(lbl.n_runs):
        for word in word_list_runs[run]:
            all_words.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    word_to_index = {word: idx for idx, word in enumerate(all_words)}
    vocab_size = len(word_to_index)

    # Convertir le dictionnaire de fréquences en un array
    word_freq_array = [0] * vocab_size
    for word, freq in word_freq.items():
        word_freq_array[word_to_index[word]] = freq
    word_freq_array = (word_freq_array - np.mean(word_freq_array)) / np.std(word_freq_array)
    word_freq_array = np.exp(word_freq_array)

    word_embeddings = {}
    for word, idx in word_to_index.items():
        one_hot_vector = np.zeros(vocab_size)
        one_hot_vector[idx] = 100 * word_freq_array[idx]
        word_embeddings[word] = one_hot_vector

    np.random.seed(0)  # pour la reproductibilité
    random_matrix = np.random.randn(vocab_size, vocab_size)
    u, _, _ = np.linalg.svd(random_matrix)
    orthogonal_matrix = u

    # Appliquer la matrice orthogonale aux vecteurs one-hot pour obtenir des vecteurs denses orthogonaux
    dense_embeddings = {word: np.dot(orthogonal_matrix, one_hot) for word, one_hot in word_embeddings.items()}    

    runs_words_activations = []
    for run in range(lbl.n_runs):
        words_activations = [dense_embeddings[word] for word in word_list_runs[run]]
        runs_words_activations.append([words_activations])
        from scipy.stats import skew
        from scipy.stats import norm, kurtosis
        print(np.mean(skew(words_activations, axis=0)))
        print(np.mean(kurtosis(words_activations, axis=0)))
        print("---------")
    # word_embeddings = {}        
    # for run in range(lbl.n_runs):
    #     for word in word_list_runs[run]:
    #         if word not in word_embeddings:
    #             word_embeddings[word] = np.random.lognormal(1, 1, n_dims)
    # for run in range(lbl.n_runs):
    #     words_activations = [word_embeddings[word] for word in word_list_runs[run]]
    #     runs_words_activations.append([words_activations])
else:
    raise Exception('Unknown random type')

# n_runs x 1 x n_words x n_neurons
filename = os.path.join(output_folder, '{}.gz'.format(model_name))
with open(filename, 'wb') as f:
     joblib.dump(runs_words_activations, f, compress=4)

if not os.path.exists(os.path.join(lbl.llms_activations, 'onsets_offsets.gz')):
    filename = os.path.join(output_folder, 'onsets_offsets.gz')
    with open(filename, 'wb') as f:
         joblib.dump(onsets_offsets_runs, f, compress=4)