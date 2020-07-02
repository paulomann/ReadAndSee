from readorsee.data.models import Config
from readorsee.data.dataset import DepressionCorpusTransformer
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
# from transformers import BertForSequenceClassification, BertForSequenceClassificationPooling, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_bert import BertLayerNorm
from readorsee.models.models import BertForSequenceClassificationWithPooler
from readorsee import settings
from readorsee.optim.adamw import mAdamW
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
import wandb
import re
from scipy.special import expit

parser = argparse.ArgumentParser(description="Arguments for fine-tuning.")
parser.add_argument("--gpu", type=int)
parser.add_argument("--period", type=int)
parser.add_argument("--dataset", type=int)
parser.add_argument("--save-stats", type=int, default=0)
parser.add_argument("--wi", type=int, default=None)
parser.add_argument("--do", type=int, default=None)
parser.add_argument("--save-name", type=str, default=None)
parser.add_argument("--bert-pooling", type=int, default=0)
parser.add_argument("--save-model", type=int, default=0)
parser.add_argument("--bert-size", type=str, default="base") # "base" or "large"
parser.add_argument("--layerwise-lr", type=int, default=0.9)
parser.add_argument("--wandb", type=int, default=0)
parser.add_argument("--freeze-bert", type=int, default=0)
parser.add_argument("--reset-layers", type=int, default=0)


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
save_model = args.save_model
bert_size = args.bert_size
layerwise_lr = args.layerwise_lr
log_wandb = args.wandb
freeze_bert_weights = args.freeze_bert
reset_layers = args.reset_layers

if log_wandb:
    wandb.init(project="readandsee", config=vars(args))


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

def get_lr_by_layer(name: str, base_lr: float = 2e-5, decay: int = 0.9, bert_size = "base"):
    match = re.search("^.*\\.(\d+)\\..*$", name)
    last_layer_idx = 23 if bert_size == "large" else 11
    if match is None:
        if "embeddings" in name:
            # last_layer_idx + 2 is because there is one more 
            # pooler and classifier layer on top of the 11
            # stack of encoders.
            return 2e-5 * (decay ** (last_layer_idx + 2))
        elif "pooler" in name:
            return 2e-5 * (decay ** 1)
        elif "classifier" in name:
            return 2e-5
    else:
        layer = int(match.group(1))
        return 2e-5 * (decay ** (last_layer_idx + 2 - layer))


# device = torch.device(f"cuda:{gpu}")
device = torch.device(f"cuda:0")

config = Config()
bestwi = -1
bestdo = -1
config.general["bert_size"] = bert_size

print(f"Parameters: Dataset={dataset}; GPU={gpu}; Period={period}; Save Stats={save_stats}; WI={wi}; DO={do}; Save Name={save_name}; Bert Pooling={bert_pooling}; Save Model={save_model}; Bert Size={bert_size}.")

if wi is not None and do is not None:
    combinations = [(do, wi)]
else:
    # combinations = itertools.product(list(range(10)), list(range(10)))
    combinations = list(itertools.product([0], list(range(0, 25))))

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

    # HYPERPARAMS
    batch_size = 16
    lr = 2e-5
    epochs = 3
    gradient_accumulation_steps = 1

    set_seed(seed_do)
    print(f"Loading dataset with bert_size = {config.general['bert_size']}")
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


    # bert_size = config.general["bert_size"].lower()
    bert_path = settings.PATH_TO_BERT[bert_size]
    print(f"Loading bert_model = {bert_path}")
    if bert_pooling:
        print(f"Using BertForSequenceClassificationWithPooler")
        model = BertForSequenceClassificationWithPooler.from_pretrained(
            bert_path,
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = True
        )
    else:
        print(f"Using BertForSequenceClassification")
        model = BertForSequenceClassification.from_pretrained(
            bert_path,
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False
        )
    
    set_seed(seed_wi)
    nn.init.normal_(model.classifier.weight.data, 0, 0.02)
    nn.init.zeros_(model.classifier.bias.data)

    # We reset the last layer dense and LayerNorm parameters
    def reset_last_layers(n_layers: int):
        last_layer_idx = 23 if bert_size == "large" else 11
        first = last_layer_idx - n_layers + 1
        last = last_layer_idx + 1
        for i in range(first, last):
            print(f"====>Reseting layer {i}!")
            for name, module in model.bert.encoder.layer[last_layer_idx].named_modules():
                if any(x in name for x in ["dense", "query", "key", "value"]):
                    module.weight.data.normal_(0, 0.02)
                    module.bias.data.zero_()

            model.bert.encoder.layer[last_layer_idx].output.LayerNorm = BertLayerNorm(model.bert.encoder.layer[last_layer_idx].output.dense.out_features, eps=1e-12)

        model.bert.pooler.reset_parameters()
    
    if reset_layers > 0:
        reset_last_layers(n_layers=reset_layers)

    model = model.to(device)
    if log_wandb:
        wandb.watch(model)

    # optimizer = Adam(model.parameters(), lr = lr, eps = 1e-8)
    no_decay = ["bias", "LayerNorm.weight"]  # no weight decay for these params



    if layerwise_lr != 0:
        print(f"====>Using layerwise learning rate with decay={layerwise_lr}")
        optimizer_grouped_parameters = []
        for name, p in model.named_parameters():
            if not any(nd in name for nd in no_decay) and p.requires_grad:
                d = {"params": p, "weight_decay": 0.01, "lr": get_lr_by_layer(name, lr, layerwise_lr, bert_size)}
            elif any(nd in name for nd in no_decay) and p.requires_grad:
                d = {"params": p, "weight_decay": 0.00, "lr": get_lr_by_layer(name, lr, layerwise_lr, bert_size)}
            
            optimizer_grouped_parameters.append(d)

    else:
        print(f"====>Using fixed learning rate={lr}")
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],  # keep only params that require a gradient
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],  # keep only params that require a gradient
                "weight_decay": 0.0
            },
        ]
    optimizer = mAdamW(
        optimizer_grouped_parameters, 
        lr=lr,
        betas=(0.9, 0.999),
        eps=0.000001, 
        correct_bias=True,
        local_normalization=False,
        max_grad_norm=1.0
    )
    t_total = (len(train_dataloader) // gradient_accumulation_steps) * epochs
    print(f"====>TOTAL NUMBER OF STEPS: {t_total}")
    warmup_steps = int(t_total * 0.1) # 10% of total steps during fine-tuning
    print(f"====>WARMUP STEPS: {warmup_steps}")

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = warmup_steps,
        num_training_steps = t_total # We used total_steps before.... Why?
    )

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
    global_step = 0
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
            # Perform a backward pass to calculate the gradients.
            loss.backward()
                # grads = param.grad
                # grads = grads.view(-1)
                # grads_norm = torch.norm(grads, p=2, dim=0)
                # weight_norm = torch.norm(param.view(-1), p=2, dim=0)
                # # compute ratio ||w|| / ||grad L(w)||
                # grad_weight_ratio = weight_norm / grads_norm

            #     param_histo = param.detach().cpu().numpy().reshape(-1)
            #     param_histo = wandb.Histogram(sequence=param_histo)
            #     wandb.log({f"{name}_grad_histo": param_histo}, step=global_step)
            #     wandb.log({f"{name}_grad_norm": grads_norm}, step=global_step)           

            total_train_loss += loss.item() * len(b_labels)

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Calculate layerwise gradients
            if log_wandb and global_step % 20 == 0:
                layer_grads = {}
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    grads = param.grad
                    grads = grads.view(-1)
                    match = re.search("^.*\\.(\d+)\\..*$", name)
                    if match is not None:
                        layer = f"Layer {match.group(1)}"
                        if layer not in layer_grads: layer_grads[layer] = []
                        layer_grads[layer].append(grads)
                    else:
                        if "classifier" in name:
                            layer = "classifier"
                            if layer not in layer_grads: layer_grads[layer] = []
                            layer_grads[layer].append(grads)
                        elif "pooler" in name:
                            layer = "pooler"
                            if layer not in layer_grads: layer_grads[layer] = []
                            layer_grads[layer].append(grads)
                
                layer_norms = {}
                for k, grads in layer_grads.items():
                    layer_norms[k] = torch.cat(grads).norm(p=2)
                wandb.log(layer_norms)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            # if global_step % 40 == 0:
                # print(f"====>Current learning rate: {optimizer.param_groups[0]['lr']}")
                # Another way to get the current learning rate
                # print(f"====>Current learning rate: {scheduler.get_lr()[0]}")


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
        # wandb.log({'train_loss': avg_train_loss, 'val_loss': avg_val_loss}, step=epoch_i)
        if log_wandb:
            wandb.log({"epoch": epoch_i, "loss": avg_train_loss, "val_loss": avg_val_loss, "val_acc": avg_val_accuracy})
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

    model.to("cpu")
    if mean_val_acc_over_epochs > best_val_acc:
        bestdo = seed_do
        bestwi = seed_wi
        best_val_acc = mean_val_acc_over_epochs
        best_model_wts = copy.deepcopy(model.state_dict())

    del train_dataloader
    del validation_dataloader
    del train_corpus
    del val_corpus
    del model
    torch.cuda.empty_cache()


if save_model:

    if bert_pooling:
        print(f"Recreating BertForSequenceClassificationWithPooler for saving...")
        model = BertForSequenceClassificationWithPooler.from_pretrained(
            bert_path,
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = True,
            state_dict=best_model_wts
        )
    else:
        print(f"Recreating BertForSequenceClassification for saving...")
        model = BertForSequenceClassification.from_pretrained(
            bert_path,
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
            state_dict=best_model_wts
        )
    if save_name:
        model_path = Path(settings.PATH_TO_BERT_MODELS_FOLDER, f"{bert_size}-dataset-{dataset}-{period}-do-{bestdo}-wi-{bestwi}-{save_name}")
    else:
        model_path = Path(settings.PATH_TO_BERT_MODELS_FOLDER, f"{bert_size}-dataset-{dataset}-{period}-do-{bestdo}-wi-{bestwi}")
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)

if save_stats:
    if save_name:
        save_path = os.path.join(settings.PATH_TO_BERT_FINE_TUNING_DATA, f"{bert_size}-dataset-{dataset}-{period}-do-{bestdo}-wi-{bestwi}-{save_name}.pk")
    else:
        save_path = os.path.join(settings.PATH_TO_BERT_FINE_TUNING_DATA, f"{bert_size}-dataset-{dataset}-{period}-do-{bestdo}-wi-{bestwi}.pk")
    with open(save_path, "wb") as f:
        pk.dump(save_data, f)