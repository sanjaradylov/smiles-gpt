# Generative Pre-Training from Molecules

Autoregressive transformer language model for drug discovery. (Pre)trained on a large
SMILES corpus. Evaluated on molecular property prediction and low-data de novo design
tasks.


## Installation

Set up [conda](https://conda.io/en/latest/index.html) and create a new environment from
`environment.yml` (if needed, make corresponding edits for GPU-compatibility).
```shell
conda env create -f environment.yml
conda activate smiles-gpt
git clone https://github.com/sanjaradylov/smiles-gpt.git
cd smiles-gpt
```


## Examples

Adapter training for molecular property prediction:
```shell
python3 scripts/classification.py checkpoints/benchmark-5m data/bbbp.csv p_np
```
For language model pretraining, see notebooks.

## Citation
