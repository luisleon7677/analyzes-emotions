"""Cabeza de clasificación personalizada de UMUTeam (wav2vec2-bert español)."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Wav2Vec2BertForSequenceClassification, Wav2Vec2BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

_HIDDEN_STATES_START_POSITION = 2


class ClassificationHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        dims = config.classifier_proj_size
        self.dense = nn.Linear(dims, dims)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(dims, config.num_labels)
        self.layernorm = nn.LayerNorm(dims)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class CustomAudioClassification(Wav2Vec2BertForSequenceClassification):
    """Arquitectura usada por UMUTeam/w2v-bert-emotion-es."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = ClassificationHead(config)
        if getattr(config, "use_weighted_layer_sum", False):
            num_layers = config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            True if self.config.use_weighted_layer_sum else output_hidden_states
        )

        outputs = self.wav2vec2_bert(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )
            hidden_states = hidden_states.clone()
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
