# Multi-target MAMMAL (protein + molecule)

MAMMAL can read a **protein sequence and a molecule in one prompt**, so a single model can cover several
targets instead of one model per target. These notebooks try that out.

**This is not part of the benchmark.** The benchmark compares models on one target at a time, on identical
folds of one dataset, and MMELON cannot take a protein at all — so there would be nothing to compare these
numbers against. They are kept here, outside [../benchmark/](../benchmark/), so the benchmark folder stays
about benchmarking and this work is not lost when
[../archive-to-be-deleted/](../archive-to-be-deleted/) is removed.

```
multitarget/
├── notebooks/
│   ├── MAMMAL_multitarget_finetuning.ipynb   fine-tune on (protein, molecule) pairs
│   └── MAMMAL_multitarget_inference.ipynb    score pairs with that model
├── sample_data/                              mock data: 3 invented proteins, molecules from the MMELON sample
└── tools/make_mock_multitarget.py            rebuilds that mock data
```

**Environment:** the `biomed-mammal` kernel (see [../INSTALL.md](../INSTALL.md)). Run the notebooks from
`notebooks/`; their default paths point at `sample_data/` next door.

**The mock data is for checking that the notebooks run**, not for results: the proteins are random sequences
and the labels are invented, so a model trained on it learns nothing real. Rebuild it with:

```bash
python aircheck/multitarget/tools/make_mock_multitarget.py
```

| Column | |
|---|---|
| `TARGET` | the target's name |
| `PROTEIN` | its amino-acid sequence, which goes in the prompt with the molecule |
| `SMILES`, `LABEL` | the molecule and whether it is a hit for that target |

## If this should become a benchmark method

It would need real multi-target data and a decision about what it is compared with: a multi-target model is
best judged against single-target models **on the same target**, so the fair comparison is "one model for many
targets" against "one model per target", not a row in the usual table. The way in would be a third method in
[../benchmark/methods/](../benchmark/methods/) writing the same `predictions.csv` and `run.json`, with folds
built per target.
