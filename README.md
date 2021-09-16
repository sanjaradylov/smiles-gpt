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


## Benchmark

### Checkpoint
[checkpoints/benchmark-5m](https://github.com/sanjaradylov/smiles-gpt/tree/master/checkpoints/benchmark-5m)
stores serialized model, tokenizer, and configuration. Do not modify them. Use
`from_pretrained` method to load HuggingFace objects, e.g.,
```python
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

checkpoint = "checkpoints/benchmark-5m"

config = GPT2Config.from_pretrained(checkpoint)
model = GPT2LMHeadModel.from_pretrained(checkpoint)
tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
```

### Data
[data](https://github.com/sanjaradylov/smiles-gpt/tree/master/data) stores
[Blood-Brain Barrier Penetration](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv)
classification dataset and 10K subset of ChemBERTa's
[PubChem-10M](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pubchem_10m.txt.zip).
See [Examples](#Examples).

### Output

[output](https://github.com/sanjaradylov/smiles-gpt/tree/master/output) stores generated
SMILES strings.

## Examples

Adapter training for molecular property prediction
(replace `data/bbbp.csv` and `p_np` arguments with your dataset and taskname(s),
respectively):
```shell
python3 scripts/classification.py checkpoints/benchmark-5m data/bbbp.csv p_np
```
For language model pretraining, see
[notebooks](https://github.com/sanjaradylov/smiles-gpt/tree/master/notebooks).

## Citation

If you use `smiles-gpt` in your research, please consider citing
> https://doi.org/10.33774/chemrxiv-2021-5fwjd