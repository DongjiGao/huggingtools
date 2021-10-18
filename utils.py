# 2021 Dongji Gao

import lhotse
from datasets import load_from_disk
from collections import defaultdict

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
            speech = audio[0][start_frame:end_frame+1]


def get_dataset(data, dataset, is_kaldi_format, sampling_rate=16e3):
    if not is_kaldi_format:
        print(f"loading {dataset} form disk")
        hf_dataset = load_from_disk(dataset)
    else:
        print(f"building huggingface datset from {data}")
        recordings, supervisions, _ = lhotse.kaldi.load_kaldi_data_dir(data, sampling_rate)
        cuts = lhotse.CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
