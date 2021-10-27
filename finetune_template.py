#!/usr/bin/env python

import argparse
import sys
from runner import SuperRunner, SuperRunnerConfig
from transformers import TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument(
    "--unit",
    type=str,
    default="char",
)
parser.add_argument(
    "--num-epoch",
    type=int,
    default=30,
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.005,
)
parser.add_argument(
    "--scale",
    type=float,
    default=0,
)
args = parser.parse_args()
print(args)

super_runner_config = SuperRunnerConfig(
    task="ASR",
    model="facebook/wav2vec2-base",
    unit="phone",
    lexicon="",
    output_dir="log",
    vocab="vocab.json",
    eval_set_name="test",
    dataset="dataset/timit",
    trainer="nc",
    scale=args.scale,
)

training_args = TrainingArguments(
    output_dir="model/timit",
    group_by_length=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    num_train_epochs=args.num_epoch,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    warmup_steps=1000,
    save_total_limit=2,
)

super_runner = SuperRunner(super_runner_config, training_args)
super_runner.run()
