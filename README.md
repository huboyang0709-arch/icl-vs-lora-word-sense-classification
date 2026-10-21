# French hypersense classification with LLMs: in-context learning vs. LoRA

Classifying French word senses (lemma + Wiktionary definition) into 12 coarse semantic categories ("hypersenses")
with **Qwen2.5-3B-Instruct (4-bit)**, comparing **in-context learning (ICL)** and **LoRA fine-tuning**.
Course project, M2 Computational Linguistics (Université Paris Cité), NLP for Society, 2026.

## Main results

| System | rand-test acc. | rand-test macro-F1 | freq-test acc. | freq-test macro-F1 |
|---|---|---|---|---|
| Majority baseline (`inanimate`) | 0.316 | 0.040 | 0.209 | 0.029 |
| ICL, best condition (E7: French prompt, lemma, few-shot) | 0.519 | 0.367 | 0.328 | 0.286 |
| LoRA, L2 (selected on rand-dev, no lemma) | 0.854 | 0.717 | 0.721 | 0.648 |
| LoRA, L1 (with lemma) | 0.867 | 0.730 | 0.759 | 0.698 |

- **LoRA (≈1% of the parameters trained) roughly doubles macro-F1 over few-shot ICL** and produces no invalid outputs.
- **The lemma helps mainly for frequent words**: +5.0 macro-F1 on freq-test (paired bootstrap 95% CI [+2.5, +7.6];
  McNemar p < 0.001), no significant effect on rand-test. The splits share no lemma, so the gain comes from the
  model's pretrained knowledge, not from memorisation.
- **Other LoRA choices made no measurable difference**: adapters on attention layers only (¼ of the parameters) and a
  classification head instead of text generation stay within the run-to-run variation (0.6–1.3 macro-F1 points
  between identical runs).
- The gap between the two test sets is mostly explained by their class distributions; the remaining errors
  concentrate on the residual class `other`, neighbouring abstract classes and composite labels.

## Experiments

**ICL** (8 conditions): prompt language (English / French) × lemma (yes / no) × demonstrations (zero-shot /
one example per class).

**LoRA** (QLoRA, r = 16, α = 16, 2 epochs, lr 2e-4, effective batch 16):

| Run | Lemma | Adapted modules | Output |
|---|---|---|---|
| L1 | yes | attention + MLP | label generated as text |
| L2 | no | attention + MLP | label generated as text |
| L3 | yes | attention only | label generated as text |
| L4 | yes | attention + MLP | classification head |

Configurations are selected on rand-dev (macro-F1) and evaluated once on rand-test and freq-test.
Significance between LoRA runs: exact McNemar test (accuracy) and paired bootstrap (macro-F1).

## Repository structure

```
lab1_shared.py              code shared by ICL and LoRA: data preparation, input format,
                            batched greedy decoding, output parsing, metrics
notebooks/
  00_icl_experiments.ipynb              ICL: the 8 conditions on rand-dev and E7 on the test sets
  01_lora_L1-L3_training.ipynb          LoRA training + rand-dev evaluation (L1–L3; one run per config)
  02_lora_L4_classification_head.ipynb  L4: classification head (transformers + PEFT, no Unsloth)
  03_lora_test_evaluation.ipynb         loads saved adapters, evaluates L1/L2 + baseline on the test sets
  04_error_analysis.ipynb               per-class results, confusions, lemma effect, ICL vs LoRA (CPU only)
  executed_runs/                        the same notebooks with the outputs of the actual Kaggle runs
results/
  lora_results_all.csv                  all LoRA runs (incl. repeated runs of L1 and L3) on rand-dev and test sets
  icl_results_summary.csv               the 8 ICL conditions on rand-dev, E7 on the test sets
  icl_paired_tests_rand-dev.csv         paired tests between ICL conditions
  predictions/                          per-instance predictions (icl/, lora_dev/, lora_test/)
  error_analysis/                       per-class scores and lists of errors
  figures/                              learning curves, confusion matrix, accuracy by lemma frequency
data/                                   dataset (not included, see data/README.md)
report/                                 project report (PDF)
```

## Running the code

All GPU experiments were run on **Kaggle notebooks (Tesla T4, 16 GB)**; the notebooks are not a packaged
pipeline but self-contained Kaggle notebooks:

1. Import a notebook into Kaggle (ICL and LoRA notebooks alike), select the **GPU T4** accelerator and enable **Internet**.
2. Run all cells (the first cell installs Unsloth, which pulls compatible versions of transformers, TRL, PEFT and
   bitsandbytes). The contents of `lab1_shared.py` are inlined in each notebook so that it runs on its own.
3. For notebooks 01 and 02, choose the run in the "Experiment settings" cell (e.g. `RUN_NAME`, `USE_LEMMA`,
   `TARGET_MODULES`) and run the whole notebook from the top for each run.

Outputs are written to `/kaggle/working`. Notebook 04 only needs the CSV files of `results/` and runs on a CPU.

Versions used: Unsloth 2026.10.7, transformers 5.16.1, PEFT 0.20.0, bitsandbytes 0.50.2 (see `requirements.txt`).
LoRA adapters are not included (≈120 MB each).

## Data

Word senses labelled with supersenses (Angleraud et al., 2024), mapped to hypersenses following the taxonomy of
Barque et al. (2020). Splits: train 10,604; rand-dev 570; rand-test 678; freq-test 1,521 (1,517 after removing 4
empty definitions).

## Credits

- ICL experiments and error analysis: Boyang Hu
- LoRA experiments and error analysis: Lezhi Zhang
- Course: Marie Candito, Université Paris Cité.

## References

- Angleraud et al. (2024). [COLING 2025 paper describing the dataset](https://aclanthology.org/2025.coling-main.356/).
- Barque et al. (2020). [Supersense taxonomy for French nouns, LREC 2020](https://aclanthology.org/2020.lrec-1.724/).
- Dietterich (1998). Approximate statistical tests for comparing supervised classification learning algorithms.
- Koehn (2004). Statistical significance tests for machine translation evaluation.
