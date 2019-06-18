import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import copy
import time
from readorsee.data.dataset import DepressionCorpus
from readorsee.models.models import ELMo, ResNet, FastText
from gensim.models.fasttext import load_facebook_model
from readorsee.data import config
import json


class Trainer():

    def __init__(self, model, dataloaders, dataset_sizes, criterion, optimizer, 
                 scheduler, num_epochs=100, threshold=0.5):
        self.acc_loss = {"train": {"loss": [], "acc": []}, 
                         "val": {"loss": [], "acc": []}}
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        print("Using device ", self.device)
        if torch.cuda.device_count() > 1:
           print("Using {} GPUs!".format(torch.cuda.device_count()))
           self.model = nn.DataParallel(model)
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
                for inputs, sif_weights, labels in self.dataloaders[phase]:
                    # if inputs.size()[0] < 4: continue
                    inputs = inputs.to(self.device)
                    sif_weights = sif_weights.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs, sif_weights)
                        # _, preds = torch.max(outputs, 1)
                        preds =  outputs > self.logit_threshold
                        loss = self.criterion(outputs, labels.float())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
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


class Experiment():
    """ This is a class to run all the experiments for this study """
    def __init__(self, experiment_type="img", 
                 observation_periods=[60, 212, 365]):
        """ 
        experiment_type: can be any of "img", "txt", "both"
        observation_periods: list of integers with observation period
        """
        with open(config.PATH_TO_CLFS_OPTIONS, "r") as f:
            hyperparameters = json.load(f)

        if not hyperparameters.keys():
            raise ValueError

        self.epochs = int(hyperparameters["general"]["num_epochs"])
        self.hyperparameters = hyperparameters

        print("Loading FastText pretrained vectors...")
        fasttext = self._load_fasttext_model()

        self.type_to_model = {"img": [ResNet(18), ResNet(34), ResNet(50)],
                              "txt": [ELMo(), FastText(fasttext)],
                              "both": []}

        self.experiment_type = experiment_type
        self.models = self.type_to_model[self.experiment_type]

        if not isinstance(observation_periods, list):
            raise ValueError
        self.observation_periods = observation_periods

    def _load_fasttext_model(self):
        fasttext = load_facebook_model(
            config.PATH_TO_FASTTEXT_PT_EMBEDDINGS, encoding="utf-8")
        return fasttext

    def _get_dataloaders(self, days, dset):
        hparameters = self.hyperparameters["general"]
        batch = int(hparameters["batch"])
        shuffle = bool(hparameters["shuffle"])

        print("GENERAL PARAMETERS ->")
        print("Batch: {}\nShuffle: {}".format(batch, shuffle))

        train_loader = DataLoader(
            DepressionCorpus(observation_period=days, subset="train",
                             data_type=self.experiment_type, dataset=dset),
            batch_size=batch,
            shuffle=shuffle
        )
        val_loader = DataLoader(
            DepressionCorpus(observation_period=days, subset="val",
                             data_type=self.experiment_type, dataset=dset),
            batch_size=batch,
            shuffle=shuffle
        )
        test_loader = DataLoader(
            DepressionCorpus(observation_period=days, subset="test",
                             data_type=self.experiment_type, dataset=dset),
            batch_size=batch,
            shuffle=shuffle
        )
        dataloaders = {"train": train_loader,
                       "val": val_loader,
                       "test": test_loader}
        return dataloaders

    def _get_criterion(self):
        """ Loss function """
        return nn.CrossEntropyLoss()

    def _get_optimizer(self, params):
        """ Algorithm that updates the weights of the network """
        trainable_params = filter(lambda p: p.requires_grad, params)
        hparameters = self.hyperparameters[self.experiment_type]["optimizer"]

        if hparameters["type"] == "sgd":
            print("Params for experiment type: ", self.experiment_type)
            lr = float(hparameters["lr"])
            momentum = float(hparameters["momentum"])
            weight_decay = float(hparameters["weight_decay"])
            nesterov = bool(hparameters["nesterov"])
            print("OPTIMIZER PARAMS ->"
                "lr: {}\nmomentum: {}\nweight_decay: {}\nnesterov: {}"
                    .format(lr, momentum, weight_decay, nesterov))
            return optim.SGD(trainable_params, lr=lr, momentum=momentum,
                            weight_decay=weight_decay, nesterov=nesterov)
        
        elif hparameters["type"] == "adam":
            # TODO
            pass
        else:
            raise ValueError

    def _get_scheduler(self, optimizer):
        print("GENERAL PARAMETERS ->")
        hparameters = self.hyperparameters[self.experiment_type]["scheduler"]
        step_size = int(hparameters["step_size"])
        gamma = float(hparameters["gamma"])
        print("step_size: {}\ngamma: {}".format(step_size, gamma))
        scheduler = lr_scheduler.StepLR(optimizer, 
                                        step_size=step_size, gamma=gamma)
        return scheduler

    def run_experiment(self):

        for days in self.observation_periods:
            print("Executing training for {} days...".format(days))
            for model in self.models:
                for dataset in range(0,10):
                    print("Training model: {} for dataset {}..."
                          .format(model.__name__, dataset))
                    dataloaders = self._get_dataloaders(days, dataset)
                    criterion = self._get_criterion()
                    optimizer = self._get_optimizer(model.parameters())
                    scheduler = self._get_scheduler(optimizer)
                    trainer = Trainer(model, dataloaders, criterion, 
                                      optimizer, scheduler, self.epochs)
                    trainer.train_model()