# 2021 Dongji Gao

import datasets
import lhotse
import os


class SuperRunnerConfig():
    def __init__(self,
                 task,
                 unit,
                 output_dir,
                 vocab="",
                 data="",
                 dataset="",
                 lexicon="",
                 ):
        self.task = task
        self.unit = unit
        self.output_dir = output_dir
        self.vocab = vocab
        self.data = data
        self.dataset = dataset
        self.lexicon = lexicon


class SuperRunner():
    def __init__(self, SuperRunnerConfig):
        self.task = SuperRunnerConfig.task
        self.output_dir = SuperRunnerConfig.output_dir

        assert unit in ["char", "phone"]
        if unit is "phone":
            assert SuperRunnerConfig.lexicon
            self.lexicon = lexicon

        self.vocab = SuperRunnerConfig.vocab

        self.data = SuperRunnerConfig.data
        self.dataset = SuperRunnerConfig.dataset
        self.kaldi_format = True
        if not os.path.exits(self.data):
            self.kaldi_format = False
            assert os.path.exists(self.dataset)


        assert self.data or self.dataset

    def process_data(self):
        dataset = get_dataset(self.data, self.dataset)

    def get_dataset(self, data, dataset):
        if not self.kaldi_format:
            print(f"loading {dataset} form disk")
        else:

