#!/usr/bin/env python3

import torch
from torch.linalg import lstsq
from datasets import load_dataset, load_from_disk


# 2021 Dongji Gao

class AnalyzerBase():
    def __init__(self, model_name, model_type, dataset):
        self.model_name = model_name
        self.model_type = model_type
        self.model = model_type.from_pretrained(model_name, output_hidden_states=True)
        self.dataset = load_from_disk(dataset)

    def set_processer(self, processer):
        assert self.tokenier and self.feature_exatrctor
        self.processer = processer(tokinzer=self.tokenizer,
                                   feature_extractor=feature_extractor)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer.from_pretrained(self.model_name)

    def set_feature_extractor(self, feature_extractor):
        self.feature_extractor = feature_extractor.from_pretained(self.model_name)

    def set_device(self):
        self.device = device

    def infer(self):
        raise NotImplementedError


class LMAnalyzer(AnalyzerBase):
    def __init__(self, model_name, model_type, dataset):
        super().__init__(model_name, model_type, dataset)

    def get_weight(self, model):
        return model.get_output_embeddings().weight

    def infer(self):
        model = self.model
        tokenizer = self.tokenizer
        dataset = self.dataset

        hyp_ids_list = list()
        features_list = list()
        ref_ids_list = list()

        def input_to_result(batch):
            input_text = batch["sentence"]
            input_tokened = tokenizer(input_text, return_tensors="pt")

            with torch.no_grad():
                output = model(**input_tokened)
            logits = output.logits
            # logits.shape = [1, num_words, feature_dim]
            logits = logits.squeeze()
            # logits.shape = [num_words, feature_dim]

            hyp_ids = torch.argmax(logits, dim=-1)[:-1]
            features = output.hidden_states[-1].squeeze()[:-1]
            ref_ids = input_tokened.input_ids.squeeze()[1:]

            hyp_ids_list.append(hyp_ids)
            features_list.append(features)
            ref_ids_list.append(ref_ids)

        dataset.map(input_to_result)

        hyp_ids = torch.cat(hyp_ids_list)
        features = torch.cat(features_list)
        ref_ids = torch.cat(ref_ids_list)

        return hyp_ids, features, ref_ids

    def compute_nc1(self):
        hyp_ids, features, ref_ids = self.infer()
        features_dict = dict()
        features_mean_dict = dict()

        num_features, feature_size = features.shape
        global_mean = torch.mean(features, dim=0)

        for id in hyp_ids:
            if id not in features_dict:
                features_dict[id] = features[hyp_ids == id]
        for id in features_dict:
            features_mean_dict[id] = torch.mean(features_dict[id], dim=0)

        num_classes = len(features_dict)
        covariance_b = torch.zeros(feature_size, feature_size)
        for id in features_mean_dict:
            tmp_vector = (features_mean_dict[id] - global_mean).unsqueeze(1)
            cur_corvariance = torch.matmul(tmp_vector, tmp_vector.T)
            covariance_b += cur_corvariance
        covariance_b /= num_classes

        covariance_w = torch.zeros(feature_size, feature_size)
        for id in features_dict:
            num_features_in_class, _ = features_dict[id].shape
            for index in range(num_features_in_class):
                tmp_vector = (features_dict[id][index] - features_mean_dict[id]).unsqueeze(1)
                cur_corvariance = torch.matmul(tmp_vector, tmp_vector.T)
                covariance_w += cur_corvariance
        covariance_w /= num_features

        nc_1 = torch.trace(lstsq(covariance_b, covariance_w).solution) / num_classes

        print(nc_1)
