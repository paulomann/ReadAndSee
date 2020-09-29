from readorsee import settings
from pathlib import Path
import random
from typing import Optional
from transformers import (
    BertTokenizer,
    LineByLineTextDataset,
    set_seed,
    BertForPreTraining,
    Trainer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)


def get_dataset(file_path: Path, tokenizer: BertTokenizer) -> LineByLineTextDataset:
    # block_size = the number of tokens we are going to use for the sequence.
    return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=150)


def get_data_collator(
    tokenizer: BertTokenizer, mlm: bool = True, mlm_prob: float = 0.15
) -> DataCollatorForLanguageModeling:
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_prob
    )
    return data_collator

set_seed(42)
bert_size = "base"
gradient_acc_steps = 8
batch_size = 32
epochs = 100
adam_epsilon = 1e-6
weight_decay = 0.1
lr = 1e-5
# Parameters from https://www.aclweb.org/anthology/2020.acl-main.740.pdf
train_path = Path(settings.PATH_TO_PROCESSED_DATA, "bert", "bert_train.txt")
val_path = Path(settings.PATH_TO_PROCESSED_DATA, "bert", "bert_val.txt")
tokenizer = BertTokenizer.from_pretrained(settings.PATH_TO_BERT[bert_size])

train_dataset = get_dataset(train_path, tokenizer)
eval_dataset = get_dataset(val_path, tokenizer)
data_collator = get_data_collator(tokenizer)
t_total = (batch_size // gradient_acc_steps) * epochs
print(f"====>TOTAL NUMBER OF STEPS: {t_total}")
warmup_steps = int(t_total * 0.10)  # 10% of total steps during fine-tuning
print(f"====>WARMUP STEPS: {warmup_steps}")
training_args = TrainingArguments(
    output_dir=str(Path(settings.PATH_TO_BERT_MODELS_FOLDER, f"{bert_size}-TAPT")),
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    learning_rate=lr,
    weight_decay=weight_decay,
    num_train_epochs=epochs,
    adam_epsilon=adam_epsilon,
    evaluate_during_training=True,
    gradient_accumulation_steps=gradient_acc_steps,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    logging_steps=10,
    eval_steps=10,
    save_total_limit=1,
    save_steps=10
)
model = BertForMaskedLM.from_pretrained(settings.PATH_TO_BERT[bert_size])
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    prediction_loss_only=True,
)
trainer.train(model_path=settings.PATH_TO_BERT[bert_size])
trainer.save_model(str(Path(settings.PATH_TO_BERT_MODELS_FOLDER, f"{bert_size}-TAPT")))