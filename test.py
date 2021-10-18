#!/usr/bin/env python3
# 2021 Dongji Gao

import datasets
from checking import check_data, get_subset

libri_toy = datasets.load_dataset("librispeech_asr", "clean", split="train.100")
check_data(libri_toy, play_audio=False)
subset = get_subset(libri_toy, duration=600)
print(subset)
