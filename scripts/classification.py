#!/usr/bin/env python3

"""Run single- or multi-task classification using pre-trained transformer decoder and
Pfeiffer adapters.
"""

try:
    import smiles_gpt as gpt
except ImportError:
    import sys
    sys.path.extend(["."])
    import smiles_gpt as gpt

import argparse
import os.path
import statistics

from pandas import read_csv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import GPT2Config, PfeifferConfig

import smiles_gpt as gpt


RANDOM_SEED = 42
EARLY_STOPPING_MAX_THRESHOLD, EARLY_STOPPING_DIV_THRESHOLD = 0.98, 0.5


def main(options: argparse.Namespace):
    model_config = GPT2Config.from_pretrained(options.checkpoint)
    model_config.num_tasks = len(options.tasks)

    tokenizer_file = os.path.join(options.checkpoint, "tokenizer.json")
    tokenizer = gpt.SMILESBPETokenizer.get_hf_tokenizer(
        tokenizer_file, model_max_length=model_config.n_positions)
    model_config.pad_token_id = tokenizer.pad_token_id

    model = gpt.GPT2ForSequenceClassification.from_pretrained(
        options.checkpoint, config=model_config)

    if options.tl_strat == "adapters":
        adapter_config = PfeifferConfig(
            mh_adapter=False,
            output_adapter=True,
            reduction_factor=options.reduction_factor,
            non_linearity=options.adapter_act,
            original_ln_before=True,
            original_ln_after=True,
            ln_before=False,
            ln_after=False,
            init_weights="bert",
            is_parallel=False,
            scaling=1.0,
            residual_before_ln=True,
            adapter_residual_before_ln=False,
            cross_adapter=False,
        )
        adapter_name = os.path.splitext(options.csv)[0]
        model.add_adapter(adapter_name, config=adapter_config)
        model.train_adapter(adapter_name)
        model.set_active_adapters(adapter_name)
    elif options.tl_strat == "freeze":
        for parameter in model.transformer.parameters():
            parameter.requires_grad = False
    elif options.tl_strat == "update":
        """The weights of `GPT2` are updated by default."""

    data_frame = read_csv(options.csv)
    splitter = gpt.CVSplitter(mode=options.split)
    data_module = gpt.CSVDataModule(
        data_frame, tokenizer, target_column=options.tasks,
        has_empty_target=options.has_empty, num_workers=options.workers,
        batch_size=options.batch_size, splitter=splitter)

    early_stopping_roc = EarlyStopping(
        "val_roc",
        min_delta=options.min_delta,
        patience=options.patience,
        mode="max",
        stopping_threshold=EARLY_STOPPING_MAX_THRESHOLD,
        divergence_threshold=EARLY_STOPPING_DIV_THRESHOLD,
    )
    early_stopping_prc = EarlyStopping(
        "val_prc",
        min_delta=options.min_delta,
        patience=options.patience,
        mode="max",
        stopping_threshold=EARLY_STOPPING_MAX_THRESHOLD,
    )
    logger = CSVLogger("logs",
                       name=f'{options.csv.split(".")[0]}-{options.tl_strat}')
    trainer = Trainer(
        logger=logger,
        log_every_n_steps=10,
        accelerator="gpu" if options.devices > 0 else "cpu",
        devices=None if options.devices == 0 else options.devices,
        auto_select_gpus=True,
        max_epochs=options.max_epochs,
        callbacks=[early_stopping_roc, early_stopping_prc],
    )

    lit_model = gpt.ClassifierLitModel(
        model, num_tasks=len(options.tasks), has_empty_labels=options.has_empty,
        batch_size=options.batch_size, learning_rate=options.learning_rate,
        scheduler_lambda=options.scheduler_lambda, weight_decay=options.weight_decay,
        scheduler_step=options.scheduler_step)
    trainer.fit(lit_model, datamodule=data_module)

    return trainer.test(lit_model, dataloaders=data_module.test_dataloader())


def process_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run single- or multi-task classification tasks using pre-trained "
                     "transformer decoder and Pfeiffer adapters."),
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data_options = parser.add_argument_group("dataset and task(s)")
    data_options.add_argument(
        "checkpoint",
        help=("The directory that stores HuggingFace transformer "
              "configuration and tokenizer file `tokenizer.json`."))
    data_options.add_argument("csv", help="CSV file with SMILES entries and task labels")
    data_options.add_argument("tasks", help="Task names", nargs="+")
    data_options.add_argument("-t", "--has_empty",
                              help="Whether tasks contain empty values",
                              action="store_true")

    early_stopping_options = parser.add_argument_group("early stopping")
    early_stopping_options.add_argument(
        "-e", "--max_epochs",
        help="The maximum number of epochs",
        type=int, default=30)
    early_stopping_options.add_argument(
        "-m", "--min_delta",
        help="Minimum change of validation AUCs to qualify as improvement",
        type=float, default=2e-3)
    early_stopping_options.add_argument(
        "-p", "--patience",
        help="Number of checks with no improvement before stopping",
        type=int, default=3)

    device_options = parser.add_argument_group("device and workers")
    device_options.add_argument("-d", "--devices",
                                help="`0` for CPU and `>= 1` for GPU",
                                default=0, type=int)
    device_options.add_argument("-w", "--workers",
                                help="Number of workers",
                                default=20, type=int)

    training_options = parser.add_argument_group("training hyperparams")
    training_options.add_argument("-b", "--batch_size",
                                  help="Train/Val/Test data loader batch size",
                                  type=int, default=64)
    training_options.add_argument("-l", "--learning_rate",
                                  help="The initial learning rate",
                                  type=float, default=7e-4)
    training_options.add_argument("-c", "--weight-decay",
                                  help="AdamW optimizer weight decay",
                                  type=float, default=0.0)
    training_options.add_argument("-a", "--scheduler_lambda",
                                  help="Lambda parameter of the exponential lr scheduler",
                                  type=float, default=0.995)
    training_options.add_argument("-s", "--scheduler_step",
                                  help="Step parameter of the exponential lr scheduler",
                                  type=float, default=5)

    val_test_options = parser.add_argument_group("validation/testing strategies")
    val_test_options.add_argument("-i", "--split",
                                  help="Train/val/test splitter",
                                  choices=("random", "scaffold"), default="scaffold")
    val_test_options.add_argument("-k", "--num_folds",
                                  help="Number of runs w/ different random seeds",
                                  type=int, default=5)

    tl_options = parser.add_argument_group("transfer learning strategies")
    tl_options.add_argument("-f", "--tl_strat",
                            help=("Whether to fine-tune w/ adapters, "
                                  "freeze backbone and update only outputs, or "
                                  "update whole model"),
                            choices=("adapters", "freeze", "update"), default="adapters")
    tl_options.add_argument("-r", "--reduction_factor",
                            help="Adapter bottleneck reduction factor",
                            type=int, default=18)
    tl_options.add_argument("-u", "--adapter_act",
                            help="Non-linearity in adapters",
                            choices=("swish", "relu", "gelu", "tanh"), default="gelu")

    return parser.parse_args()


if __name__ == "__main__":
    from warnings import filterwarnings
    from pytorch_lightning import seed_everything
    from rdkit.RDLogger import DisableLog

    filterwarnings("ignore", category=UserWarning)
    DisableLog("rdApp.*")

    options = process_options()

    prc_results, roc_results = [], []
    for fold_i in range(options.num_folds):
        seed_everything(seed=RANDOM_SEED + fold_i)
        results = main(options)[0]
        prc_results.append(results["test_prc"])
        roc_results.append(results["test_roc"])

    if options.num_folds > 1:
        mean_roc, std_roc = statistics.mean(roc_results), statistics.stdev(roc_results)
        mean_prc, std_prc = statistics.mean(prc_results), statistics.stdev(prc_results)
    else:
        mean_roc, std_roc = roc_results[0], 0.
        mean_prc, std_prc = prc_results[0], 0.

    print(f"Mean AUC-ROC: {mean_roc:.3f} (+/-{std_roc:.3f})")
    print(f"Mean AUC-PRC: {mean_prc:.3f} (+/-{std_prc:.3f})")
