import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PreTrainedModel, PretrainedConfig

def collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch])
    lengths = torch.tensor([len(x["input_ids"]) for x in batch])
    labels = torch.tensor([int(x["labels"]) for x in batch])
    return {"input_ids": input_ids, "lengths":lengths, "labels": labels}

def tokenizes(examples, tokenizer):
    return tokenizer(examples, truncation=True, max_length=128, padding="max_length")
	
class NLIConfig(PretrainedConfig):
    model_type = "NLI"
    def __init__(self, vocab_size=20000, hidden_size=1024, nclass=3, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.nclass = nclass


class NLI(PreTrainedModel):
    config_class = NLIConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.fc = nn.Linear(config.hidden_size, config.nclass)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, lengths, labels=None, **kwargs):
        # Support two modes:
        # - if this model instance has a pretrained encoder attribute (e.g., self.encoder),
        #   use it with possible attention_mask passed via kwargs.
        # - otherwise, fallback to the original embedding+LSTM classifier.
        attention_mask = kwargs.get("attention_mask", None)
        if hasattr(self, "encoder"):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0, :]
            if hasattr(self, "dropout"):
                pooled = self.dropout(pooled)
            logits = self.fc(pooled)
        else:
            # original LSTM path
            # if lengths not provided, try to compute from attention_mask
            if (lengths is None or (isinstance(lengths, torch.Tensor) and lengths.numel() == 0)) and attention_mask is not None:
                lengths = attention_mask.long().sum(dim=1)

            x = self.embedding(input_ids)

            # use packed sequence for LSTM to ignore padding positions
            packed = pack_padded_sequence(
                x,
                lengths.to("cpu"),
                batch_first=True,
                enforce_sorted=False,
            )

            packed_out, (h, c) = self.lstm(packed)
            # h shape: (num_layers * num_directions, batch, hidden_size)
            # take last layer's hidden state
            last_hidden = h[-1]
            logits = self.fc(last_hidden)  # (batch, nclass)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
