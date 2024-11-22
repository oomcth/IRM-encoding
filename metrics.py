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
from datasets import load_dataset
import math
import torch


step = str(globals().get('arg', 143000))
full_name = "EleutherAI/pythia-14m"
revision = "step" + step
cache_dir = "./" + full_name + "step" + step

output_folder = "metrics"
make_dir(output_folder)


model = GPTNeoXForCausalLM.from_pretrained(
    full_name,
    revision=revision,
    cache_dir=cache_dir,
    output_hidden_states=True,
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
    maxlen = 32000

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

stride = maxlen - 64

filename = os.path.join(lbl.annotation_folder, 'lppEN_word_information.csv')
df_word_onsets = pd.read_csv(filename)

df_word_onsets = df_word_onsets.drop([3919, 6775, 6781])

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


runs_layers_words_activations = []

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

    input_ids = inputs['input_ids']
    logperp = []
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        logperp.append(-loss.item())
        del outputs, input_ids, loss
perplexity = math.exp(sum(logperp) / len(logperp))
torch.save(perplexity, output_folder + '/' + full_name + 'perplexity' + str(step) + .'pt')


# Charger le dataset HellaSwag
dataset = load_dataset('hellaswag', split='validation')


def predict_ending(context, endings):
    scores = []
    for ending in endings:
        # Concaténer le contexte et la fin
        input_text = context + " " + ending
        inputs = tokenizer(input_text, return_tensors='pt')
        input_ids = inputs['input_ids']
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            scores.append(-loss.item())  # La perte négative car un score plus élevé est meilleur
    return scores.index(max(scores))


correct_predictions = 0
total_predictions = len(dataset)

for example in tqdm(dataset):
    context = example['ctx_a'] + " " + example['ctx_b']
    endings = example['endings']
    correct_ending = example['label']
    predicted_ending = predict_ending(context, endings)
    if predicted_ending == correct_ending:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
torch.save(accuracy, output_folder + '/' + full_name + 'hellaswag' + str(step) + '.pt')
del model, tokenizer, runs_layers_words_activations, inputs, df_word_onsets
