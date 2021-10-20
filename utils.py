# 2021 Dongji Gao

import lhotse
import random
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict, load_from_disk, load_metric
from transformers import Wav2Vec2Processor
from typing import Any, Dict, List, Optional, Union


def lhotse_to_huggingface(cuts, sampling_rate=16e3):
    dataset_dict = defaultdict(list)
    for cut in cuts:
        audio = cut.load_audio()
        for segment in cut.to_dict()["supervisions"]:
            id = segment["id"]
            spk_id = segment["speaker"]
            text = segment["text"]
            duration = float(segment["segment"])

            start_frame = int(sampling_rate * float(segment["start"]))
            end_frame = start_frame + int(sampling_rate * duration)
            speech = audio[0][start_frame:end_frame + 1]

            for item in [id, spk_id, text, duration, speech]:
                dataset_dict[f"{item}"].append(item)

    hf_dataset = Dataset.from_dict(dataset_dict)

    return hf_dataset


def get_dataset_from_disk(data, dataset, is_kaldi_format, sampling_rate=16e3):
    if not is_kaldi_format:
        print(f"loading {dataset} form disk")
        hf_dataset = load_from_disk(dataset)
    else:
        print(f"building huggingface datset from {data}")
        recordings, supervisions, _ = lhotse.kaldi.load_kaldi_data_dir(data,
                                                                       sampling_rate)
        cuts = lhotse.CutSet.from_manifests(recordings=recordings,
                                            supervisions=supervisions)
        hf_dataset = lhotse_to_huggingface(cuts)

    return hf_dataset


# TODO: split by duration?
def split_dataset_randomly(dataset, eval_name, num_eval=100):
    hf_dataset = DatasetDict()
    dataset_length = len(dataset)
    assert num_eval <= dataset_length

    eval_indexes = list()
    train_indexes = list()

    for _ in range(num_eval):
        pick = random.randint(0, dataset_length - 1)
        while pick in eval_indexes:
            pick = random.randint(0, dataset_length - 1)
        eval_indexes.append(pick)

    for index in range(dataset_length):
        if index not in eval_indexes:
            train_indexes.append(index)

    hf_dataset["train"] = Dataset.from_dict(dataset[train_indexes])
    hf_dataset[eval_name] = Dataset.from_dict(dataset[eval_indexes])

    return hf_dataset

def get_subset_by_duration(dataset, duration=3600):
    # be careful about duration
    total_duration = 0
    dataset_length = len(dataset)
    pick_list = list()

    while total_duration <= duration:
        pick = random.randint(0, dataset_length - 1)
        while pick in pick_list:
            pick = random.randint(0, dataset_length - 1)
        pick_list.append(pick)
        batch = dataset[pick]
        if "duration" in batch:
            total_duration += batch["duration"]
        else:
            if "speech" in batch and "sampling_rate" in batch:
                cur_duration = len(batch["speech"]) / batch["sampling_rate"]
            else:
                raw_wav, sampling_rate = torchaudio.load(batch["file"])
                raw_wav = raw_wav[0]
                cur_duration = len(raw_wav) / sampling_rate

            total_duration += cur_duration

    sub_dataset = datasets.Dataset.from_dict(dataset[pick_list])

    return sub_dataset

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self,
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[
        str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1),
                                                       -100)
        batch["labels"] = labels
        return batch



