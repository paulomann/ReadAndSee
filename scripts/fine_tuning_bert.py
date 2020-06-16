from readorsee.data.models import Config
from readorsee.data.dataset import DepressionCorpusTransformer
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import BertForSequenceClassification, BertForSequenceClassificationPooling, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from readorsee import settings
import numpy as np
import time
import datetime
import random
import pandas as pd
import argparse
import pickle as pk
import itertools
from readorsee import settings
import os
from pathlib import Path
import copy

parser = argparse.ArgumentParser(description="Arguments for fine-tuning.")
parser.add_argument("--gpu", type=int)
parser.add_argument("--period", type=int)
parser.add_argument("--dataset", type=int)
parser.add_argument("--save-stats", type=int, default=0)
parser.add_argument("--wi", type=int, default=None)
parser.add_argument("--do", type=int, default=None)
parser.add_argument("--save-name", type=str, default=None)
parser.add_argument("--bert-pooling", type=int, default=0)
parser.add_argument("--pooling-method", type=str, default="concat") #can be "mean" or "concat"
parser.add_argument("--save-model", type=int, default=0)
parser.add_argument("--bert-size", type=str, default="base") # "base" or "large"

args = parser.parse_args()

# do for Data Order
# seed_do = args.sdo
# wi for Weight initialization
# seed_wi = args.swi
gpu = args.gpu
period = args.period
dataset = args.dataset
save_stats = args.save_stats
wi = args.wi
do = args.do
save_name = args.save_name
bert_pooling = args.bert_pooling
pooling_method = args.pooling_method
save_model = args.save_model
bert_size = args.bert_size


save_data = []


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

device = torch.device(f"cuda:{gpu}")

config = Config()

print(f"Parameters: Dataset={dataset}; GPU={gpu}; Period={period}; Save Stats={save_stats}; WI={wi}; DO={do}; Save Name={save_name}; Bert Pooling={bert_pooling}; Pooling Method={pooling_method}; Save Model={save_model}; Bert Size={bert_size}.")

if wi is not None and do is not None:
    combinations = [(do, wi)]
else:
    # combinations = itertools.product(list(range(10)), list(range(10)))
    combinations = list(itertools.product([0], list(range(0, 100))))

best_val_acc = float("-inf")
best_model_wts = None
# for comb in :
for comb in combinations:
    print(f"Running experiment for combination {comb}")
    seed_do = comb[0]
    seed_wi = comb[1]

    def _init_fn(worker_id):
        np.random.seed(seed_do)

    train_corpus = DepressionCorpusTransformer(period, dataset, "train", config)
    val_corpus = DepressionCorpusTransformer(period, dataset, "val", config)
    test_corpus = DepressionCorpusTransformer(period, dataset, "test", config)

    # HYPERPARAMS
    batch_size = 32
    lr = 2e-5
    epochs = 3
    dropout = 0.1

    set_seed(seed_do)
    train_dataloader = DataLoader(
                train_corpus,
                sampler = RandomSampler(train_corpus),
                batch_size = batch_size,
                pin_memory=True,
                worker_init_fn=_init_fn,
                num_workers=0
    )

    validation_dataloader = DataLoader(
                val_corpus, 
                sampler = RandomSampler(val_corpus),
                batch_size = batch_size,
                pin_memory=True,
                worker_init_fn=_init_fn,
                num_workers=0
    )

    test_dataloader = DataLoader(
                test_corpus,
                sampler = RandomSampler(test_corpus),
                pin_memory=True,
                worker_init_fn=_init_fn,
                num_workers=0
    )

    # bert_size = config.general["bert_size"].lower()
    bert_path = settings.PATH_TO_BERT[bert_size]
    if bert_pooling:
        print(f"Using BertForSequenceClassificationPooling with {pooling_method} method")
        model = BertForSequenceClassificationPooling.from_pretrained(
            bert_path,
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = True,
            pooling_mode=pooling_method,
            last_pooling_layers = 4
        )
        # Here we use last_pooling_layers = 4, i.e., we get the
        # last 4 layers and concat their CLS token representation
    else:
        print(f"Using BertForSequenceClassification")
        model = BertForSequenceClassification.from_pretrained(
            bert_path,
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False
        )
    model = model.to(device)

    set_seed(seed_wi)
    nn.init.normal_(model.classifier.weight.data, 0, 0.02)
    nn.init.zeros_(model.classifier.bias.data)
    optimizer = Adam(model.parameters(), lr = lr, eps = 1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))


    training_stats = []
    total_t0 = time.time()
    mean_val_acc_over_epochs = 0
    threshold = 0.5
    logit_threshold = torch.tensor(threshold / (1 - threshold), device=device).log()
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            # print(batch[2])
            batch_inputs, batch_labels = batch
            b_input_ids = batch_inputs["input_ids"].squeeze().to(device)
            if b_input_ids.dim() == 1:
                b_input_ids = b_input_ids.unsqueeze(0)
            b_input_mask = batch_inputs["attention_mask"].squeeze().to(device)
            if b_input_mask.dim() == 1:
                b_input_mask = b_input_mask.unsqueeze(0)
            b_labels = batch_labels.float().to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.

            loss, logits, *_ = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item() * len(b_labels)
            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        # print(f"====>SIZE OF TRAIN DATALOADER={len(train_corpus)}")
        avg_train_loss = total_train_loss / len(train_corpus)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0.0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            batch_inputs, batch_labels = batch
            b_input_ids = batch_inputs["input_ids"].squeeze().to(device)
            if b_input_ids.dim() == 1:
                b_input_ids = b_input_ids.unsqueeze(0)
            b_input_mask = batch_inputs["attention_mask"].squeeze().to(device)
            if b_input_mask.dim() == 1:
                b_input_mask = b_input_mask.unsqueeze(0)
            b_labels = batch_labels.float().to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                loss, logits, *_ = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item() * len(b_labels)
            preds = (logits > logit_threshold).squeeze()
            total_eval_accuracy += torch.sum(preds.int() == b_labels.data.int()).float()

            # Move logits and labels to CPU
            # logits = logits.detach().cpu().numpy()
            # label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            # total_eval_accuracy += flat_accuracy(logits, label_ids)
            

        # Report the final accuracy for this validation run.
        # print(f"====> Avg_val_accuracy type={}")
        avg_val_accuracy = total_eval_accuracy / len(val_corpus)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_corpus)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy.item(),
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        mean_val_acc_over_epochs += avg_val_accuracy.item()

    mean_val_acc_over_epochs /= epochs
    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    print(f"Training stats: {training_stats}")

    print(f"Mean Valid. Acc. over epochs: {mean_val_acc_over_epochs}")

    save_data.append({"seed_do":seed_do, "seed_wi":seed_wi, "training_stats":training_stats})

    if mean_val_acc_over_epochs > best_val_acc:
        best_val_acc = mean_val_acc_over_epochs
        best_model_wts = copy.deepcopy(model.state_dict())

    del train_dataloader
    del validation_dataloader
    del test_dataloader
    del train_corpus
    del val_corpus
    del test_corpus
    del model
    torch.cuda.empty_cache()


if save_model:

    if bert_pooling:
        print(f"Recreating BertForSequenceClassificationPooling with {pooling_method} method for saving...")
        model = BertForSequenceClassificationPooling.from_pretrained(
            bert_path,
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = True,
            pooling_mode=pooling_method,
            last_pooling_layers = 4,
            state_dict=best_model_wts
        )
        # Here we use last_pooling_layers = 4, i.e., we get the
        # last 4 layers and concat their CLS token representation
    else:
        print(f"Recreating BertForSequenceClassification for saving...")
        model = BertForSequenceClassification.from_pretrained(
            bert_path,
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
            state_dict=best_model_wts
        )
    model_path = Path(settings.PATH_TO_BERT_MODELS_FOLDER, save_name)
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)

if save_stats:
    if save_name:
        save_path = os.path.join(settings.PATH_TO_BERT_FINE_TUNING_DATA, f"{save_name}.pk")
    else:
        save_path = os.path.join(settings.PATH_TO_BERT_FINE_TUNING_DATA, f"dataset-{dataset}.pk")
    with open(save_path, "wb") as f:
        pk.dump(save_data, f)