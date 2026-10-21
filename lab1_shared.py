"""
lab1_shared.py -- code shared by the ICL and LoRA experiments.
Both team members must use this exact file so that data, input format,
decoding, parsing and metrics are identical across methods.

Contents:
  1. Fixed settings (model, seed, decoding length, evaluation data, labels)
  2. Data loading and preparation
  3. Input format (format_input)
  4. Batched generation (generate_batch)
  5. Output parsing and evaluation (parse_output, evaluate)
  6. Saving results (save_results)

Not included (method-specific):
  - ICL: task instructions, few-shot demonstrations, message building
  - LoRA: training instruction and training code

Usage: import unsloth first (`from unsloth import FastLanguageModel`),
then paste this file into a notebook cell, or upload it and `from lab1_shared import *`.
"""

import os
import re
import gzip
import random
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


# =====================================================================
# 1. Fixed settings -- any change must be made by both team members
# =====================================================================

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
SEED = 42
MAX_NEW_TOKENS = 10        # labels are short; 10 new tokens are enough

DEV_SUBSET_N = None        # None = full rand-dev; set an int (e.g. 300) to tune on a fixed subset
DROP_LABELS = []           # hypersenses to exclude (e.g. ["other"]), if we decide to

DATA_URL = ("http://www.linguist.univ-paris-diderot.fr/~mcandito/divers/"
            "super_wiktionary_labeled_data_simplified.noex.25labels.tsv.gz")

# Supersense -> hypersense mapping, copied from the lab notebook
SUPER2HYPER = {
    'artifact': 'inanimate',
    'body': 'inanimate',
    'food': 'inanimate',
    'object': 'inanimate',
    'plant': 'inanimate',
    'substance': 'inanimate',
    'act': 'dynamic_situation',
    'event': 'dynamic_situation',
    'phenomenon': 'dynamic_situation',
    'animal': 'animate',
    'person': 'animate',
    'groupxperson': 'animate',
    'cognition': 'information',
    'communication': 'information',
    'quantity': 'quantity',
    'attribute': 'stative_situation',
    'feeling': 'stative_situation',
    'relation': 'stative_situation',
    'state': 'stative_situation',
    'time': 'time',
    'possession': 'possession',
    'institution': 'institution',
    'act*cognition': 'dynamic_situation*information',
    'artifact*cognition': 'inanimate*information',
    'other': 'other',
}

# Field names of the input, depending on the prompt language
FIELDS = {
    "en": ("Lemma", "Definition"),
    "fr": ("Lemme", "Définition"),
}


def set_seed(seed=SEED):
    """Fix Python, NumPy and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# =====================================================================
# 2. Data loading and preparation
# =====================================================================

def load_data(url=DATA_URL):
    """Download the gzipped TSV (if not already present) and load it."""
    filename = url.split('/')[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        return pd.read_csv(f, sep='\t')


def prepare_data(df):
    """Add the hypersense column, drop protect-frsemcor, split by the 'set' column.
    Returns (splits, labels):
      splits: dict with keys 'train', 'rand-dev', 'rand-test', 'freq-test'
      labels: sorted list of hypersense labels, shared by both methods"""
    df = df.copy()
    df['hypersense'] = df['supersense'].map(SUPER2HYPER)
    missing = df.loc[df['hypersense'].isna(), 'supersense'].unique()
    assert len(missing) == 0, f"Supersenses missing from SUPER2HYPER: {missing}"

    df = df[df['set'] != 'protect-frsemcor']
    if DROP_LABELS:
        df = df[~df['hypersense'].isin(DROP_LABELS)]

    # Sets other than the four below (if any) are not used
    print(df['set'].value_counts(), "\n")

    names = ['train', 'rand-dev', 'rand-test', 'freq-test']
    splits = {n: df[df['set'] == n].reset_index(drop=True) for n in names}
    if DEV_SUBSET_N:
        splits['rand-dev'] = (splits['rand-dev']
                              .sample(DEV_SUBSET_N, random_state=SEED)
                              .reset_index(drop=True))

    labels = sorted(df['hypersense'].unique())
    for n in names:
        print(f"{n:10s}: {len(splits[n])} senses")
        assert len(splits[n]) > 0, f"Split '{n}' is empty: check the values of the 'set' column"
    print("Labels:", labels)
    return splits, labels


# =====================================================================
# 3. Input format -- used by both ICL and LoRA
# =====================================================================

def format_input(row, lang="en", use_lemma=True):
    """Turn one word sense into the input text, e.g.
         Lemma: béruguette
         Definition: Variété d'olive allongée à pulpe ferme."""
    f_lemma, f_def = FIELDS[lang]
    text = f"{f_def}: {row['definition']}"
    if use_lemma:
        text = f"{f_lemma}: {row['lemma']}\n" + text
    return text


# =====================================================================
# 4. Batched generation -- used by both methods at inference time
# =====================================================================

def generate_batch(model, tokenizer, messages_list, batch_size=16):
    """messages_list: one chat (list of messages) per instance.
    Returns the raw generated strings, in the same order as messages_list.
    Greedy decoding, MAX_NEW_TOKENS, left padding, batches sorted by length."""
    import torch

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"              # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
               for m in messages_list]
    # Sort by length so each batch needs little padding; restore order at the end
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs = [None] * len(prompts)

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        # The chat template already contains the special tokens
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt",
                        padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        for i, t in zip(idx, texts):
            outputs[i] = t.strip()

    tokenizer.padding_side = old_side
    return outputs


# =====================================================================
# 5. Parsing and evaluation
# =====================================================================

def parse_output(text, labels):
    """Extract a label from the generated text. Returns (prediction, status) with
    status 'valid' (exactly one label), 'none' (no label) or 'multiple' (several labels).
    Boundaries exclude letters, digits, '_' and '*', so 'animate' does not match
    inside 'inanimate', and 'inanimate' does not match inside 'inanimate*information'."""
    text = text.lower()
    found = [l for l in labels
             if re.search(r'(?<![\w*])' + re.escape(l.lower()) + r'(?![\w*])', text)]
    if len(found) == 1:
        return found[0], "valid"
    return "INVALID", ("none" if not found else "multiple")


def evaluate(rows_df, raw_outputs, labels, verbose=True):
    """Returns (metrics, res):
      metrics: accuracy and macro-F1 (invalid outputs count as errors), invalid rates
      res: one row per instance, for saving and error analysis"""
    parsed = [parse_output(t, labels) for t in raw_outputs]
    res = pd.DataFrame({
        "sense_id":   rows_df["sense_id"].values,
        "lemma":      rows_df["lemma"].values,
        "definition": rows_df["definition"].values,
        "gold":       rows_df["hypersense"].values,
        "raw":        raw_outputs,
        "pred":       [p for p, _ in parsed],
        "status":     [s for _, s in parsed],
    })
    valid = res["status"] == "valid"
    # Macro-F1 is averaged over the labels that occur in this split only,
    # so that a label absent from the split does not pull the average down
    present = [l for l in labels if l in set(res["gold"])]
    metrics = {
        "n":                len(res),
        "accuracy":         accuracy_score(res["gold"], res["pred"]),
        "macro_f1":         f1_score(res["gold"], res["pred"], labels=present,
                                     average="macro", zero_division=0),
        "invalid_rate":     1 - valid.mean(),
        "no_label_rate":    (res["status"] == "none").mean(),
        "multi_label_rate": (res["status"] == "multiple").mean(),
        "acc_on_valid":     accuracy_score(res.loc[valid, "gold"], res.loc[valid, "pred"])
                            if valid.any() else 0.0,
    }
    if verbose:
        for k, v in metrics.items():
            print(f"{k:18s}: {v:.4f}" if isinstance(v, float) else f"{k:18s}: {v}")
        print(classification_report(res["gold"], res["pred"], labels=labels, zero_division=0))
    return metrics, res


# =====================================================================
# 6. Saving results -- {run}_{split}_preds.csv + one shared results_summary.csv
# =====================================================================

def save_results(run_name, split_name, metrics, res, out_dir, config=None):
    """run_name: e.g. 'E1', 'L1'; split_name: 'rand-dev', 'rand-test' or 'freq-test';
    config: optional dict describing the run, e.g. {'method': 'ICL', 'lang': 'en', 'lemma': True}"""
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, f"{run_name}_{split_name}_preds.csv"), index=False)

    row = {"run": run_name, "split": split_name, **(config or {}), **metrics}
    summary_path = os.path.join(out_dir, "results_summary.csv")
    summary = pd.DataFrame([row])
    if os.path.exists(summary_path):
        summary = pd.concat([pd.read_csv(summary_path), summary], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {run_name} / {split_name}")
