import numpy as np
import pandas as pd
import os
import joblib
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, GPTNeoXForCausalLM
import torch
import os
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm import tqdm
import llms_brain_lateralization as lbl
from llms_brain_lateralization import make_dir
import sys


na = str(globals().get('arg', 1000))
step = str(143_000)
full_name = "EleutherAI/pythia-" + na
revision = "step" + step
cache_dir = "./" + full_name + "step" + step

output_folder = lbl.llms_activations
make_dir(output_folder)


model = GPTNeoXForCausalLM.from_pretrained(
    full_name,
    revision=revision,
    force_download=True,
    cache_dir=cache_dir,
    output_hidden_states=True,  # Ensure hidden states are output
    output_attentions=False
)
tokenizer = AutoTokenizer.from_pretrained(
    full_name
)
tokenizer = AutoTokenizer.from_pretrained(full_name)


n_layers = model.config.num_hidden_layers
try:
    maxlen = model.config.max_position_embeddings
except AttributeError:
    # ssm: infinite
    maxlen = 32000  # this is infinity here

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

stride = maxlen - 64

# Each model is fed with the full original English text, with full punctuation.
# For each run, this text can be found in the lpp_full_text folder as defined
# in the llms_brain_lateralization.py file. In order to align this text and thus
# the activity of the neural networks with what the subjects in the scanner heard,
# we made use of the time-aligned speech segmentation provided in the LPP naturalistic
# fMRI corpus, available in OpenNeuro. This information is given in the file
# lppEN_word_information.csv. Due to some discrepancies between these two sources,
# some ad-hoc heuristics are used in order to improve the alignement.

filename = os.path.join(lbl.annotation_folder, 'lppEN_word_information.csv')
df_word_onsets = pd.read_csv(filename)

df_word_onsets = df_word_onsets.drop([3919, 6775, 6781])
# 3919: adhoc removal of repeated line with typo
# 6775: mismatch with full text

word_list_runs = []
onsets_offsets_runs = []
for run in range(lbl.n_runs):
    df_word_onsets_run = df_word_onsets[df_word_onsets.section == (run+1)]
    word_list_tmp = df_word_onsets_run.word.to_numpy()
    onsets_tmp = df_word_onsets_run.onset.to_numpy()
    offsets_tmp = df_word_onsets_run.offset.to_numpy()
    word_list = []
    onsets = []
    offsets = []
    for idx_word, (word, onset, offset) in enumerate(zip(word_list_tmp,
                                                         onsets_tmp,
                                                         offsets_tmp)):
        if isinstance(word, str):
            word_list.append(word)
            onsets.append(onset)
            offsets.append(offset)

    onsets_offsets_runs.append((np.array(onsets), np.array(offsets)))
    word_list_runs.append(word_list)


def simplify_word(word):
    return word.lower().replace(' ', '').replace('-', '').replace(
        "'", ""
    ).replace("’", "").replace("“", "").replace('—', '')


def do_word_match(word_in_list, word_in_text):
    if len(simplify_word(word_in_text)) > 0 and word.startswith(simplify_word(word_in_text)):
        return True
    if len(word) > 1 and word in simplify_word(word_in_text):
        return True
    if len(simplify_word(word_in_text)) > 1 and simplify_word(word_in_text) in word:
        return True
    if word == 'one' and simplify_word(word_in_text) == '1':
        return True
    if word == 'did' and simplify_word(word_in_text) == 'didn':
        return True
    if word == 'nt' and (simplify_word(word_in_text) == 't'):
        return True
    if word == 'does' and simplify_word(word_in_text) == "doesn":
        return True
    if word == 'do' and simplify_word(word_in_text) == "don":
        return True
    if word == 'is' and simplify_word(word_in_text) == "isn":
        return True
    if word == 'threetwofive' and simplify_word(word_in_text) == "3":
        return True
    if word == 'threetwosix' and simplify_word(word_in_text) == "3":
        return True
    if word == 'threetwoseven' and simplify_word(word_in_text) == "3":
        return True
    if word == 'threetwoeight' and simplify_word(word_in_text) == "3":
        return True
    if word == 'threetwonine' and simplify_word(word_in_text) == "3":
        return True
    if word == 'threethreezero' and simplify_word(word_in_text) == "3":
        return True
    if word == 'na\ive' and simplify_word(word_in_text) == 'naive':
        return True
    return False


runs_layers_words_activations = []  # n_runs x n_layers x n_words x n_neurons

for run in tqdm(range(lbl.n_runs)):
    word_list = word_list_runs[run]

    filename = os.path.join(lbl.lpp_full_text, 'text_english_run{}.txt'.format(run+1))
    with open(filename, 'r') as f:
        fulltext_run = f.read()

    # make a few corrections so as to help aligning the two sources of text
    if run == 3:
        fulltext_run = fulltext_run.replace('Minster', 'Minister')
    if run == 4:
        fulltext_run = fulltext_run.replace('1440', 'one thousand four hundred and forty')
    if run == 5:
        fulltext_run = fulltext_run.replace('111', 'one hundred and eleven')
        fulltext_run = fulltext_run.replace('7000', 'seven thousand')
        fulltext_run = fulltext_run.replace('900000', 'nine hundred thousand')
        fulltext_run = fulltext_run.replace('7500000', 'seven million five hundred thousand')
        fulltext_run = fulltext_run.replace('311000000', 'three hundred and eleven million')
        fulltext_run = fulltext_run.replace('2000000000', 'two billion')
        fulltext_run = fulltext_run.replace('462511', 'four hundred and sixty two thousand five hundred and eleven')
    if run == 7:
        fulltext_run = fulltext_run.replace('did I have this sense', 'did I have to have this sense')
    if run == 8:
        fulltext_run = fulltext_run.replace('price', 'prince')

    fulltext_run.replace('\n', ' ')

    inputs = tokenizer(fulltext_run,
                       return_tensors='pt',
                       return_offsets_mapping=True,
                       truncation=True,
                       padding=True,
                       max_length=maxlen,
                       return_overflowing_tokens=True,
                       stride=stride
                       )

    # from word in list, find position of corresponding tokens
    idx_batch = 0
    idx_token = 0
    idx_word_to_idx_token = []
    for idx_word, word in enumerate(word_list):
        word = simplify_word(word)
        if idx_token == maxlen:
            idx_batch += 1
            idx_token = stride
        i_start, i_stop = inputs['offset_mapping'][idx_batch, idx_token].numpy()

        n = 0
        while n < 15 and not do_word_match(word, fulltext_run[i_start:i_stop]):
            idx_token += 1
            if idx_token == maxlen:
                idx_batch += 1
                idx_token = stride
            i_start, i_stop = inputs['offset_mapping'][idx_batch, idx_token].numpy()
            n += 1
        if n == 15:
            raise Exception('Error while parsing text file -- no corresponding token to word')

        # word idx_word starts at idx_token in batch idx_batch
        idx_word_to_idx_token.append((idx_batch, idx_token))
        idx_token += 1

    # could feed it directly all the batches,
    # but it explodes my memory
    batch_size = inputs['input_ids'].shape[0]
    hidden_states = []  # batch_size x n_layers x n_tokens x n_neurons
    for k in range(batch_size):
        with torch.no_grad():
            model_outputs = model(inputs['input_ids'][k:k+1],
                                  attention_mask=inputs['attention_mask'][k:k+1])
        hidden_states.append([model_outputs['hidden_states'][idx_layer][0].numpy()
                             for idx_layer in range(n_layers+1)])

    layers_words_activations = [[] for _ in range(n_layers+1)]  # n_layers x n_words x n_neurons

    for idx_word in range(len(word_list)-1):
        idx_batch, idx_token = idx_word_to_idx_token[idx_word]
        idx_batch_next, idx_token_next = idx_word_to_idx_token[idx_word+1]
        if idx_batch == idx_batch_next:
            emb_layers = [[] for _ in range(n_layers+1)]  # n_layers x n_tokens x n_neurons
            for i in range(idx_token, idx_token_next):
                for idx_layer in range(n_layers+1):
                    emb_layers[idx_layer].append(hidden_states[idx_batch][idx_layer][i])
            for idx_layer in range(n_layers+1):
                layers_words_activations[idx_layer].append(np.mean(emb_layers[idx_layer], axis=0))
        else:  # idx_batch != idx_batch_next
            emb_layers = [[] for _ in range(n_layers+1)]  # n_layers x n_tokens x n_neurons
            # go to the end of the batch ...
            for i in range(idx_token, np.minimum(maxlen, inputs['input_ids'][idx_batch].shape[0])):
                for idx_layer in range(n_layers+1):
                    emb_layers[idx_layer].append(hidden_states[idx_batch][idx_layer][i])
            # ... and start the next one
            for i in range(stride, idx_token_next):
                for idx_layer in range(n_layers+1):
                    emb_layers[idx_layer].append(hidden_states[idx_batch_next][idx_layer][i])
            for idx_layer in range(n_layers+1):
                layers_words_activations[idx_layer].append(np.mean(emb_layers[idx_layer], axis=0))

    # now the last word
    idx_word += 1
    idx_batch, idx_token = idx_word_to_idx_token[idx_word]
    emb_layers = [[] for _ in range(n_layers+1)]  # n_layers x n_tokens x n_neurons
    for i in range(idx_token, np.minimum(maxlen, inputs['input_ids'][idx_batch].shape[0])):
        token = inputs['input_ids'][idx_batch, i]
        if token == tokenizer.eos_token_id:
            break
        for idx_layer in range(n_layers+1):
            emb_layers[idx_layer].append(hidden_states[idx_batch][idx_layer][i])
    for idx_layer in range(n_layers+1):
        layers_words_activations[idx_layer].append(np.mean(emb_layers[idx_layer], axis=0))

    runs_layers_words_activations.append(layers_words_activations)

# n_runs x n_layers x n_words x n_neurons
filename = os.path.join(output_folder, '{}.gz'.format(full_name + '-step:' + step))
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'wb') as f:
    joblib.dump(runs_layers_words_activations, f, compress=4)

if not os.path.exists(os.path.join(lbl.llms_activations, 'onsets_offsets.gz')):
    filename = os.path.join(output_folder, 'onsets_offsets.gz')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        joblib.dump(onsets_offsets_runs, f, compress=4)

del model, tokenizer, runs_layers_words_activations, hidden_states, model_outputs, inputs, df_word_onsets