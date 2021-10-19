# 2021 Dongji Gao

import datasets
import json
import lhotse
import numpy as np
import os
from datasets import load_dataset, load_metric
from mapping import get_chars, text_to_phones, tokenize_data
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from typing import Any, Dict, List, Optional, Union
from utils import (
    lhotse_to_huggingface,
    get_dataset_from_disk,
    split_dataset,
    DataCollatorCTCWithPadding,
)


class SuperRunnerConfig():
    def __init__(self,
                 task,
                 model,
                 unit,
                 output_dir,
                 training_set_name="train",
                 eval_set_name="eval",
                 vocab="",
                 data="",
                 dataset="",
                 lexicon="",
                 trainer="trainer",
                 sampling_rate=16e3
                 ):
        self.task = task
        self.model = model
        self.unit = unit
        self.training_set_name = training_set_name,
        self.eval_set_name = eval_set_name,
        self.output_dir = output_dir
        self.vocab = vocab
        self.data = data
        self.dataset = dataset
        self.lexicon = lexicon
        self.trainer = trainer
        self.sampling_rate = sampling_rate


class SuperRunner():
    def __init__(self, super_runner_config, training_args):
        self.task = super_runner_config.task
        self.model = super_runner_config.model
        self.unit = super_runner_config.unit
        self.training_set_name = super_runner_config.training_set_name[0]
        self.eval_set_name = super_runner_config.eval_set_name[0]

        self.output_dir = super_runner_config.output_dir

        assert self.unit in ["char", "phone"]
        if self.unit == "phone":
            assert os.path.exits(super_runner_config.lexicon)
            self.lexicon = lexicon

        self.vocab = super_runner_config.vocab

        self.data = super_runner_config.data
        self.dataset = super_runner_config.dataset
        self.is_kaldi_format = True
        if not os.path.exists(self.data):
            self.is_kaldi_format = False
            assert os.path.exists(self.dataset)

        assert self.data or self.dataset
        self.sampling_rate = super_runner_config.sampling_rate
        self.trainer = super_runner_config.trainer
        self.training_args = training_args

    # TODO: whether move this function to utils?
    def process_data(self, data, dataset, is_kaldi_format):
        hf_dataset = get_dataset_from_disk(data, dataset, is_kaldi_format)
        if self.eval_set_name not in hf_dataset:
            print(f"eval set {self.eval_set_name} not found, spliting dataset")
            hf_dataset = split_dataset(hf_dataset, self.eval_set_name, num_eval=500)
            hf_dataset.save_to_disk(dataset)

        if self.unit == "char":
            if not self.vocab:
                vocabs = hf_dataset.map(get_chars, batched=True, batch_size=-1,
                                        remove_columns=hf_dataset.column_names["train"])
                vocab_list = list(
                    set(vocabs["train"]["vocab"][0]) | set(vocabs["eval"]["vocab"][0]))
                vocab_dict = {v: k + 1 for k, v in enumerate(vocab_list)}

                vocab_dict["|"] = vocab_dict[" "]
                del vocab_dict[" "]
                vocab_dict["<eps>"] = 0
                vocab_dict["<UNK>"] = len(vocab_dict)

                with open('vocab.json', 'w') as vocab_file:
                    json.dump(vocab_dict, vocab_file)
        else:
            if not self.vocab:
                phone_list = list()
                lexicon_dict = dict()
                with open(self.lexicon, "r") as lex:
                    for line in lex.readlines():
                        line_list = lints.split()
                        word = line_list[0]
                        phones = line_list[1:]
                        lexicon_dict[word] = phones
                        phone_list += phones

                vocab_list = list(set(phones_list))
                vocab_dict = {v: k + 1 for k, v in enumerate(vocab_list)}
                vocab_dict["<eps>"] = 0
                vocab_dict["<UNK>"] = len(vocab_dict)

                with open('vocab.json', 'w') as vocab_file:
                    json.dump(vocab_dict, vocab_file)
            if "phones" not in hf_dataset:
                hf_dataset = hf_dataset.map(text_to_phones)

        return hf_dataset

    def run(self):
        hf_dataset = self.process_data(self.data, self.dataset, self.is_kaldi_format)

        is_split_into_word = True if self.unit == "char" else False
        tokenizer = Wav2Vec2CTCTokenizer("./vocab.json",
                                         unk_token="<UNK>",
                                         pad_token="<eps>",
                                         word_dilimiter_token="|",
                                         is_split_into_word=is_split_into_word)
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                     sampling_rate=self.sampling_rate,
                                                     padding_value=0.0,
                                                     do_normalize=True,
                                                     return_attention_mask=False)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                      tokenizer=tokenizer)
        tk_dict = {"processor": processor, "unit": self.unit, "sampling_rate": self.sampling_rate}
        hf_tokened_dataset = hf_dataset.map(tokenize_data,
                                            fn_kwargs=tk_dict,
                                            )
        # padding
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

        model = Wav2Vec2ForCTC.from_pretrained(
            self.model,
            gradient_checkpointing=True,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer.get_vocab())
        )

        # metric
        wer_metric = load_metric("wer")

        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)
            pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.batch_decode(pred_ids)
            label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
            print("hyp: {}".format(pred_str[0]))
            print("ref: {}".format(label_str[0]))
            wer = wer_metric.compute(predictions=pred_str, references=label_str)

            return {"wer": wer}

        # training
        SpeechTrainer = NCTrainer if self.trainer == "nc" else Trainer
        trainer = SpeechTrainer(
            model=model,
            data_collator=data_collator,
            args=self.training_args,
            compute_metrics=compute_metrics,
            train_dataset=hf_tokened_dataset[self.training_set_name],
            eval_dataset=hf_tokened_dataset[self.eval_set_name],
            tokenizer=processor.feature_extractor
        )
        trainer.train()
