# 2021 Dongji Gao

import datasets
import mapping
import numpy as np
import random
import sys
import torchaudio
from mapping import speech_to_input

SAMPLING_RATE = 16e3


def check_data(dataset, dis_number=2, play_audio=False):
    if isinstance(dataset, datasets.DatasetDict):
        print("Please specify Dataset from DatasetDict")
        return

    dataset_length = len(dataset)
    assert dataset_length >= dis_number

    print("==== sampling {} examples from {} in total ====".format(dis_number, dataset_length))
    pick_list = list()
    for _ in range(dis_number):
        pick = random.randint(0, dataset_length - 1)
        while pick in pick_list:
            pick = random.randint(0, dataset_length - 1)
        pick_list.append(pick)

    # convert dict to Dataset
    sub_dataset = datasets.Dataset.from_dict(dataset[pick_list])
    if play_audio:
        from IPython.display import display
        import IPython.display as ipd

        if "speech" not in sub_dataset[0]:
            sub_dataset = sub_dataset.map(speech_to_input)

    for index in range(dis_number):
        for column in sub_dataset.column_names:
            if column == "speech":
                print("{}: {} (first 10 values)".format(column, sub_dataset[column][index][:10]))
            else:
                print("{}: {}".format(column, sub_dataset[column][index]))
        print("")

        display(ipd.Audio(data=np.asarray(sub_dataset[index]["speech"]),
                          autoplay=False, rate=SAMPLING_RATE))