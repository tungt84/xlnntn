import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation

from transformers import (
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EvalPrediction,
)

# `get_last_checkpoint` location changed across transformers versions —
# try importing it from the main package, then `trainer_utils`, then
# fall back to a simple filesystem-based implementation.
try:
    from transformers import get_last_checkpoint
except Exception:
    try:
        from transformers.trainer_utils import get_last_checkpoint
    except Exception:
        import os
        import re

        def get_last_checkpoint(output_dir):
            if not os.path.isdir(output_dir):
                return None
            checkpoints = [d for d in os.listdir(output_dir)
                           if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint")]
            if not checkpoints:
                return None

            def keyfn(name):
                m = re.search(r"checkpoint-?(\d+)", name)
                if m:
                    return int(m.group(1))
                return 0

            checkpoints.sort(key=keyfn)
            return os.path.join(output_dir, checkpoints[-1])

from datasets import load_from_disk, Dataset
from pathlib import Path
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from model import *

import evaluate

# Load pre-downloaded dataset splits from disk (saved under DATASETS/multi_nli)
data_base = Path(__file__).resolve().parents[1] / "DATASETS" / "multi_nli"
train_ds = load_from_disk(str(data_base / "train"))
val_ds = load_from_disk(str(data_base / "validation_matched"))

df_train = train_ds.to_pandas()
df_train = df_train.dropna()

df_val = val_ds.to_pandas()
df_val = df_val.dropna()

# Use pretrained English tokenizer (small model to meet 40M param limit)
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
tokenizer.save_pretrained("./MODEL")

def collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch])
    attention_mask = torch.tensor([x["attention_mask"] for x in batch])
    labels = torch.tensor([x["labels"] for x in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def tokenizes_pair(premise, hypothesis):
    return tokenizer(premise, hypothesis, truncation=True, max_length=128, padding="max_length")
    
def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    pred_labels = np.argmax(preds, axis=-1)
    targ = p.label_ids
    res = {}
    metric = evaluate.load("accuracy")
    res["accuracy"] = metric.compute(predictions=pred_labels, references=targ)["accuracy"]
    metric = evaluate.load("precision")
    res["precision"] = metric.compute(predictions=pred_labels, references=targ, average="macro", zero_division=0)["precision"]
    metric = evaluate.load("recall")
    res["recall"] = metric.compute(predictions=pred_labels, references=targ, average="macro", zero_division=0)["recall"]
    metric = evaluate.load("f1")
    res["f1"] = metric.compute(predictions=pred_labels, references=targ, average="macro")["f1"]
    return res
	
tokenized_train = df_train.apply(lambda x: tokenizes_pair(x["premise"], x["hypothesis"]), axis=1)
train_set = Dataset.from_dict({
    "input_ids": [t["input_ids"] for t in tokenized_train],
    "attention_mask": [t["attention_mask"] for t in tokenized_train],
    "labels": df_train["label"].tolist(),
})
tokenized_val = df_val.apply(lambda x: tokenizes_pair(x["premise"], x["hypothesis"]), axis=1)
val_set = Dataset.from_dict({
    "input_ids": [t["input_ids"] for t in tokenized_val],
    "attention_mask": [t["attention_mask"] for t in tokenized_val],
    "labels": df_val["label"].tolist(),
})

model = AutoModelForSequenceClassification.from_pretrained("albert-base-v2", num_labels=3)

allparams = sum(p.numel() for p in model.parameters())
trainparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("All Param:", allparams, "Train Params:", trainparams)
args = TrainingArguments(output_dir="./NLIMODEL",
                         load_best_model_at_end= True,
                         dataloader_pin_memory=True,
                         per_device_train_batch_size=16,
                         learning_rate=2e-5,
                         weight_decay=0.01,
                         num_train_epochs=3,
                         eval_strategy="epoch",
                         save_strategy="epoch"
                         )

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=val_set,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

last_checkpoint = None
try:
    last_checkpoint = get_last_checkpoint(args.output_dir)
except Exception:
    last_checkpoint = None

if last_checkpoint is not None:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()
model.save_pretrained("./MODEL")
