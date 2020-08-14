import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_electra import ElectraModel, ElectraConfig, ElectraPreTrainedModel


class KoELECTRASpacingModel(nn.Module):
    def __init__(self):
        super(KoELECTRASpacingModel, self).__init__()
        
        self.bert = ElectraModel.from_pretrained("monologg/koelectra-small-v2-discriminator")
        config = self.bert.config
        self.pad_idx = config.pad_token_id

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.token_cls = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def forward(self, input_ids, token_type_ids):
        attention_mask = input_ids.ne(self.pad_idx).float()
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        outputs = outputs[0]
        token_output = self.token_cls(self.dropout(outputs))

        return token_output