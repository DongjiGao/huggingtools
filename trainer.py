# 2021 Dongji Gao

import torch
from transformers import Trainer


class NCTrainer(Trainer):
    # notice this trainer should only work for Wav2Vec2ForCTC model
    def compute_loss(self, model, inputs, return_outputs=False):
        scale = self.scale
        outputs = model(**inputs)
        ctc_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # regularization
        for index, layer in enumerate(model.lm_head.parameters()):
            if index == 0:
                weights = layer
        weights_norm = torch.linalg.norm(weights, dim=1).unsqueeze(dim=1)
        norm_matrix = torch.matmul(weights_norm, weights_norm.T)
        cos_matrix = torch.div(torch.matmul(weights, weights.T), norm_matrix)
        regularization = torch.sum(cos_matrix)

        loss = ctc_loss + scale * regularization

        return (loss, outputs) if return_outputs else loss

    def set_scale(self, scale=0.0):
        self.scale = scale
