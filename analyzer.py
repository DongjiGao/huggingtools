#!/usr/bin/env python3

# 2021 Dongji Gao

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, load_from_disk
from torch.linalg import lstsq
import numpy as np


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

    def get_weights(self):
        weights_dict = dict()
        weights = self.model.get_output_embeddings().weight
        num_ids, weight_size = weights.shape
        for id in range(num_ids):
            weights_dict[id] = weights[id]
        self.weights_dict = weights_dict

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

        self.hyp_ids = hyp_ids
        self.features = features
        self.ref_ids = ref_ids

    def compute_nc1(self):
        hyp_ids = self.hyp_ids
        features = self.features
        ref_ids = self.ref_ids

        features_dict = dict()
        features_mean_dict = dict()

        num_features, feature_size = features.shape
        global_mean = torch.mean(features, dim=0)

        for id in hyp_ids:
            id = id.item()
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

        self.features_mean_dict = features_mean_dict
        print(nc_1)
        return nc_1

    def plot_angle(self):
        Cos = torch.nn.CosineSimilarity()
        weights_dict = self.weights_dict
        features_mean_dict = self.features_mean_dict
        hyp_ids = self.hyp_ids

        features_angle_list = list()
        weights_angle_list = list()
        features_product_list = list()
        weights_product_list = list()
        number_list = list()
        ids_list = list()

        id_number_dict = dict()

        for id in features_mean_dict:
            id_number_dict[id] = torch.sum(hyp_ids == id).item()
            ids_list.append(id)


        for i in range(len(ids_list)):
            id_1 = ids_list[i]
            for j in range(i + 1, len(ids_list)):
                id_2 = ids_list[j]
                features_angle = Cos(features_mean_dict[id_1].unsqueeze(dim=0),
                                     features_mean_dict[id_2].unsqueeze(dim=0))
                features_product = torch.matmul(features_mean_dict[id_1].unsqueeze(dim=0),
                                                features_mean_dict[id_2].unsqueeze(dim=0).T)
                weights_angle = Cos(weights_dict[id_1].unsqueeze(dim=0),
                                    weights_dict[id_2].unsqueeze(dim=0))
                weights_product = torch.matmul(weights_dict[id_1].unsqueeze(dim=0),
                                               weights_dict[id_2].unsqueeze(dim=0).T)
                number = id_number_dict[id_1] + id_number_dict[id_2]

                features_angle_list.append(features_angle.item())
                weights_angle_list.append(weights_angle.item())
                features_product_list.append(features_product.item())
                weights_product_list.append(weights_product.item())
                number_list.append(number)

        features_angle = np.array(features_angle_list)
        weights_angle = np.array(weights_angle_list)
        features_product = np.array(features_product_list)
        weights_product = np.array(weights_product_list)
        numbers = np.array(number_list)
        fig, ax = plt.subplots()
        ax.scatter(features_product, weights_product, s=numbers/100)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.grid()
        plt.savefig("nc_1_product.png")

    def tmp_variance(self):
        weight = self.model.get_output_embeddings().weight
        num_ids, weight_dize = weight.shape
        mean = torch.mean(weight, dim=0)
        var = torch.var(weight, dim=0)

        ratio = var / torch.square(mean)
        mean_sorted, index = torch.sort(mean)
        var_sorted = var[index]
        ratio[torch.argmax(ratio)] = 0

        plt.plot(ratio.detach().numpy())
        plt.savefig("ratio.png")

