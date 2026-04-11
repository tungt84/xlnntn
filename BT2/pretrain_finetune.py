import os
from pathlib import Path
import numpy as np
import torch
import shutil

from datasets import load_from_disk, Dataset
from tokenizers import Tokenizer, processors
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments

from model import NLI, NLIConfig
import evaluate


# NOTE: Use the same TrainingArguments parameter names as in BT2/train.py
# No fallback: construct TrainingArguments directly to match train.py.


def load_disk_dataset(base_path: Path, split_name: str):
    p = base_path / split_name
    if not p.exists():
        return None
    return load_from_disk(str(p))


def df_from_dataset(ds):
    if ds is None:
        return None
    df = ds.to_pandas()
    return df.dropna()


def normalize_label_column(df):
    if df is None:
        return None

    if "label" in df.columns:
        # labels may already be integers; filter invalid
        if df["label"].dtype.kind in "iu":
            df = df[df["label"] >= 0]
            return df
        # strings -> map
        mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
        if df["label"].dtype == object:
            df = df[df["label"].isin(mapping.keys())]
            df["label"] = df["label"].map(mapping)
            return df

    # try common SNLI names
    if "sentence1" in df.columns and "sentence2" in df.columns:
        df = df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis"})

    if "gold_label" in df.columns:
        mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
        df = df[df["gold_label"].isin(mapping.keys())]
        df = df.rename(columns={"gold_label": "label"})
        df["label"] = df["label"].map(mapping)
        return df

    return df


def build_tokenizer(sentences, save_dir: Path):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[UNK]", 0),
            ("[CLS]", 1),
            ("[SEP]", 2),
            ("[PAD]", 3),
            ("[MASK]", 4),
        ],
    )

    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], min_frequency=2)
    tokenizer.train_from_iterator(sentences, trainer=trainer, length=len(sentences))

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(str(save_dir))
    return hf_tokenizer


def tokenizes(examples, tokenizer):
    return tokenizer(examples, truncation=True, max_length=128, padding="max_length")


def compute_metrics(results):
    pred, targ = results
    pred = np.argmax(pred, axis=-1)
    res = {}
    res["accuracy"] = evaluate.load("accuracy").compute(predictions=pred, references=targ)["accuracy"]
    res["f1"] = evaluate.load("f1").compute(predictions=pred, references=targ, average="macro")["f1"]
    return res


def prepare_dataset_for_trainer(df, tokenizer):
    if df is None:
        return None
    # ensure columns
    if "premise" not in df.columns or "hypothesis" not in df.columns:
        # try to find alternatives
        if "sentence1" in df.columns and "sentence2" in df.columns:
            df = df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis"})

    texts = (df["premise"] + " [CLS] " + df["hypothesis"]).astype(str)
    tokenized = texts.apply(lambda x: tokenizes(x, tokenizer))
    ds = Dataset.from_dict({"input_ids": [t["input_ids"] for t in tokenized], "labels": df["label"].astype(int).tolist()})
    return ds


def main():
    root = Path(__file__).resolve().parents[1]
    snli_base = root / "DATASETS" / "snli"
    multi_base = root / "DATASETS" / "multi_nli"

    # load datasets
    snli_train = load_disk_dataset(snli_base, "train")
    snli_val = load_disk_dataset(snli_base, "validation") or load_disk_dataset(snli_base, "dev")

    multi_train = load_disk_dataset(multi_base, "train")
    multi_val = load_disk_dataset(multi_base, "validation_matched") or load_disk_dataset(multi_base, "validation")

    df_snli_train = df_from_dataset(snli_train)
    df_snli_val = df_from_dataset(snli_val)
    df_multi_train = df_from_dataset(multi_train)
    df_multi_val = df_from_dataset(multi_val)

    df_snli_train = normalize_label_column(df_snli_train)
    df_snli_val = normalize_label_column(df_snli_val)
    df_multi_train = normalize_label_column(df_multi_train)
    df_multi_val = normalize_label_column(df_multi_val)

    # collect sentences for tokenizer
    sentences = []
    for d in (df_snli_train, df_snli_val, df_multi_train, df_multi_val):
        if d is None:
            continue
        if "premise" in d.columns:
            sentences.extend(d["premise"].astype(str).tolist())
        if "hypothesis" in d.columns:
            sentences.extend(d["hypothesis"].astype(str).tolist())

    if not sentences:
        raise SystemExit("No sentences found for tokenizer training. Ensure DATASETS/snli and DATASETS/multi_nli exist on disk.")

    model_dir = root / "MODEL"
    hf_tokenizer = build_tokenizer(sentences, model_dir)

    # prepare HF datasets for trainer
    snli_train_ds = prepare_dataset_for_trainer(df_snli_train, hf_tokenizer)
    snli_val_ds = prepare_dataset_for_trainer(df_snli_val, hf_tokenizer)
    multi_train_ds = prepare_dataset_for_trainer(df_multi_train, hf_tokenizer)
    multi_val_ds = prepare_dataset_for_trainer(df_multi_val, hf_tokenizer)

    # Pretrain on SNLI
    pretrain_out = root / "NLIMODEL" / "pretrain"
    pretrain_out.mkdir(parents=True, exist_ok=True)

    model = NLI(NLIConfig(vocab_size=len(hf_tokenizer.get_vocab())))

    pretrain_args = TrainingArguments(
        output_dir=str(pretrain_out),
        load_best_model_at_end=True,
        dataloader_pin_memory=True,
        per_device_train_batch_size=32,
        learning_rate=0.001,
        weight_decay=0.01,
        num_train_epochs=5,
        eval_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=pretrain_args,
        train_dataset=snli_train_ds,
        eval_dataset=snli_val_ds,
        data_collator=lambda b: {"input_ids": torch.tensor([x["input_ids"] for x in b]),
                                 "lengths": torch.tensor([len(x["input_ids"]) for x in b]),
                                 "labels": torch.tensor([int(x["labels"]) for x in b])},
        compute_metrics=lambda p: compute_metrics((p.predictions, p.label_ids)) if p.predictions is not None else {},
    )

    print("Starting pretraining on SNLI...")
    trainer.train()
    trainer.save_model(str(pretrain_out))

    # remove training snapshot/checkpoint folders to avoid accidental resume
    def clear_checkpoints(dir_path: Path):
        if not dir_path.exists():
            return
        for p in dir_path.iterdir():
            if p.is_dir() and p.name.startswith("checkpoint"):
                try:
                    print(f"Removing checkpoint snapshot: {p}")
                    shutil.rmtree(p)
                except Exception as e:
                    print(f"Failed to remove {p}: {e}")

    clear_checkpoints(pretrain_out)

    # Fine-tune on MultiNLI
    finetune_out = root / "NLIMODEL" / "finetune"
    finetune_out.mkdir(parents=True, exist_ok=True)

    # load the pretrained weights into a fresh model instance
    model_finetune = NLI(NLIConfig(vocab_size=len(hf_tokenizer.get_vocab())))
    model_finetune = model_finetune.from_pretrained(str(pretrain_out))

    finetune_args = TrainingArguments(
        output_dir=str(finetune_out),
        load_best_model_at_end=True,
        dataloader_pin_memory=True,
        per_device_train_batch_size=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_train_epochs=10,
        eval_strategy="epoch",
        save_strategy="epoch",
    )

    trainer_ft = Trainer(
        model=model_finetune,
        args=finetune_args,
        train_dataset=multi_train_ds,
        eval_dataset=multi_val_ds,
        data_collator=lambda b: {"input_ids": torch.tensor([x["input_ids"] for x in b]),
                                 "lengths": torch.tensor([len(x["input_ids"]) for x in b]),
                                 "labels": torch.tensor([int(x["labels"]) for x in b])},
        compute_metrics=lambda p: compute_metrics((p.predictions, p.label_ids)),
    )

    print("Starting fine-tuning on MultiNLI...")
    trainer_ft.train()
    trainer_ft.save_model(str(finetune_out))

    print("Pretrain + finetune complete. Models saved to:")
    print(pretrain_out)
    print(finetune_out)


if __name__ == "__main__":
    main()
