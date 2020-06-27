import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import copy
import time
from readorsee.data.dataset import DepressionCorpus, DepressionCorpusTwitter
from gensim.models.fasttext import load_facebook_model
import json
import numpy as np

__all__ = ["Trainer", "train_model", "train_model_twitter"]

class Trainer():

    def __init__(self, model, dataloaders, dataset_sizes, criterion, optimizer, 
                 scheduler, config, num_epochs=100, threshold=0.5):
        self.config = config
        general_config = self.config.general
        gpus = general_config["gpus"]
        self.acc_loss = {"train": {"loss": [], "acc": []}, 
                         "val": {"loss": [], "acc": []}}
        self.device = torch.device(
            f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if len(gpus) > 1:
           print(f"Using {gpus} GPUs!")
           self.model = nn.DataParallel(model, device_ids=gpus)
        else:
            print("Using device ", self.device)
        self.dataset_sizes = dataset_sizes
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.logit_threshold = torch.tensor(threshold / (1 - threshold)).log()
        self.logit_threshold = self.logit_threshold.to(self.device)

    def train_model(self, verbose=True):

        since = time.time()

        self.acc_loss = {"train": {"loss": [], "acc": []}, 
                         "val": {"loss": [], "acc": []}}

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            if verbose:
                print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
                print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for *inputs, labels in self.dataloaders[phase]:
                    inputs = [i.to(self.device) for i in inputs]
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(*inputs)
                        # _, preds = torch.max(outputs, 1)
                        preds =  outputs > self.logit_threshold
                        loss = self.criterion(outputs, labels.float())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs[0].size(0)
                    running_corrects += torch.sum(preds.long() == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = (running_corrects.double() / 
                             self.dataset_sizes[phase])

                if verbose:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
                self.acc_loss[phase]["loss"].append(epoch_loss)
                self.acc_loss[phase]["acc"].append(epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            if verbose:
                print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

def _init_fn(worker_id):
    np.random.seed(12 + worker_id)

def train_model(model, days, dataset, fasttext, config, verbose):
    print("======================")
    print("Training...")
    media_type = config.general["media_type"]
    media_config = getattr(config, media_type)
    embedder = media_config.get("txt_embedder", "").lower()

    train = DepressionCorpus(
        observation_period=days,
        subset="train",
        fasttext=fasttext,
        dataset=dataset,
        config=config
    )

    if embedder == "bow":
        model.set_out_ftrs(train.bow_ftrs_size)

    criterion = nn.BCEWithLogitsLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    general = config.general
    optimizer_name = media_config["optimizer"]["type"]
    opt_params = media_config["optimizer"]["params"]
    scheduler = media_config["scheduler"]
    optimizer_ft = getattr(optim, optimizer_name)(parameters, **opt_params)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, **scheduler)

    train_loader = DataLoader(train,
                              batch_size=general["batch_size"],
                              shuffle=general["shuffle"],
                              pin_memory=True,
                              worker_init_fn=_init_fn)

    val = DepressionCorpus(observation_period=days,
                            subset="val",
                            fasttext=fasttext,
                            dataset=dataset,
                            config=config
                            )

    val_loader = DataLoader(val,
                            batch_size=general["batch_size"],
                            shuffle=general["shuffle"],
                            pin_memory=True,
                            worker_init_fn=_init_fn)

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train), "val": len(val)}

    trainer = Trainer(model,
                      dataloaders,
                      dataset_sizes,
                      criterion,
                      optimizer_ft,
                      exp_lr_scheduler,
                      config,
                      general["epochs"])

    trained_model = trainer.train_model(verbose)
    return trained_model

def train_model_twitter(model, days, dataset, fasttext, config, verbose):
    print("======================")
    print("Training...")
    media_type = config.general["media_type"]
    media_config = getattr(config, media_type)
    embedder = media_config.get("txt_embedder", "").lower()

    train = DepressionCorpusTwitter(
        observation_period=days,
        subset="train",
        fasttext=fasttext,
        dataset=dataset,
        config=config
    )

    if embedder == "bow":
        model.set_out_ftrs(train.bow_ftrs_size)

    criterion = nn.BCEWithLogitsLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    general = config.general
    optimizer_name = media_config["optimizer"]["type"]
    opt_params = media_config["optimizer"]["params"]
    scheduler = media_config["scheduler"]
    optimizer_ft = getattr(optim, optimizer_name)(parameters, **opt_params)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, **scheduler)

    train_loader = DataLoader(train,
                              batch_size=general["batch_size"],
                              shuffle=general["shuffle"],
                              pin_memory=True,
                              worker_init_fn=_init_fn)

    val = DepressionCorpusTwitter(observation_period=days,
                            subset="val",
                            fasttext=fasttext,
                            dataset=dataset,
                            config=config
                            )

    val_loader = DataLoader(val,
                            batch_size=general["batch_size"],
                            shuffle=general["shuffle"],
                            pin_memory=True,
                            worker_init_fn=_init_fn)

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train), "val": len(val)}

    trainer = Trainer(model,
                      dataloaders,
                      dataset_sizes,
                      criterion,
                      optimizer_ft,
                      exp_lr_scheduler,
                      config,
                      general["epochs"])

    trained_model = trainer.train_model(verbose)
    return trained_model


# def fine_tune_XLM(model, days, dataset, config, verbose):
