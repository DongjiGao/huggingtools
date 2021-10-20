#!/usr/bin/env python

import sys
from transformers import TrainingArguments
from runner import SuperRunner, SuperRunnerConfig

scale=sys.argv[1]

super_runner_config = SuperRunnerConfig(
    task="ASR",
    model="facebook/wav2vec2-base",
    unit="char",
    output_dir="log",
    eval_set_name="eval",
    dataset="dataset/librispeech_toy",
    trainer="nc",
    scale=scale,
)

training_args = TrainingArguments(
    output_dir="model/test",
    group_by_length=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    save_total_limit=2,
)

super_runner = SuperRunner(super_runner_config, training_args)
super_runner.run()
