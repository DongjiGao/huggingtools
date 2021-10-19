#!/usr/bin/env python

from transformers import TrainingArguments
from runner import SuperRunner, SuperRunnerConfig

super_runner_config = SuperRunnerConfig(
    task="ASR",
    model="facebook/wav2vec2-base",
    unit="char",
    output_dir="log",
    eval_set_name="train",
    dataset="dataset/librispeech_toy"
)

training_args = TrainingArguments(
    output_dir="model/test",
    group_by_length=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    save_steps=500,
    eval_steps=50,
    logging_steps=50,
    learning_rate=5e-4,
    weight_decay=0.005,
    warmup_steps=500,
    save_total_limit=2,
)

super_runner = SuperRunner(super_runner_config, training_args)
super_runner.run()
