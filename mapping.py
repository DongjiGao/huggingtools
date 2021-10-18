# 2021 Dongji Gao

import librosa
import numpy as np
import re
import torchaudio

SAMPLING_RATE = 16e3


def speech_to_input(batch):
    raw_wav, sampling_rate = torchaudio.load(batch["file"])
    if sampling_rate != SAMPLING_RATE:
        raw_wav = librosa.resample(np.asarray(raw_wav), sampling_rate, SAMPLING_RATE)
    batch["speech"] = raw_wav
    batch["sampling_rate"] = SAMPLING_RATE

    return batch


def get_duration(batch):
    batch["duration"] = len(batch["speech"]) / batch["sampling_rate"]
    return batch


def normalize_text(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch


def get_chars(batch):
    texts = " ".join(batch["text"])
    vocab = list(set(texts))
    return {"vocab": [vocab]}


def text_to_phones(batch, **kwargs):
    phones = list()
    text = batch["text"]
    lexicon_dict = kwargs["lexicon"]
    for word in text.split():
        phone_list = ["<UNK>"] if word not in lexicon_dict else lexicon_dict[word]
        for phone in phone_list:
            phones.append(phone)
            phones.append("|")
    batch["phones"] = phones[:-1]

    return batch


def tokenize_data(batch, **kwargs):
    sampling_rate = kwargs["sampling_rate"]
    processor = kwargs["processor"]
    unit = kwargs["unit"]

    input_values = keyprocessor(batch["speech"], sampling_rate=sampling_rate).input_values
    batch["input_values"] = input_values[0]

    with processor.as_target_processor():
        if unit == "char":
            labels = processor(batch["text"]).input_ids
        else:
            raw_ids = processor(batch["phones"]).input_ids
            labels = [x[0] for x in raw_ids]
    return batch
