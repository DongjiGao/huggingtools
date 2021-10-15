#!/usr/bin/env python3
# 2021 Dongji Gao

import datasets
from checking import check_data

libri_toy = datasets.load_dataset("dgao/librispeech_nc_test", "clean", split="validation")
check_data(libri_toy, play_audio=True)
