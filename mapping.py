# 2021 Dongji Gao

import torchaudio
import librosa
import numpy as np

SAMPLING_RATE = 16e3

def speech_to_input(batch):
    raw_wav, sampling_rate = torchaudio.load(batch["file"])
    if sampling_rate != SAMPLING_RATE:
        raw_wav = librosa.resample(np.asarray(raw_wav), sampling_rate, SAMPLING_RATE)
    batch["speech"] = raw_wav
    batch["sampling_rate"] = SAMPLING_RATE

    return batch

def get_duration(batch):
    batch["duration"] = len(batch["speech"])/batch["sampling_rate"]
    return batch


