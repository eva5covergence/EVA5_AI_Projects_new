from __future__ import print_function, with_statement, division
import copy
import os
import torch
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from .lr_finder import StateCacher, ExponentialLR, LinearLR

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
del logging


class LRRangeFinder(object):
    """Learning rate range test.
    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.
    Arguments:
        model (torch.nn.Module): wrapped model.
        optimizer (torch.optim.Optimizer): wrapped optimizer where the defined learning
            is assumed to be the lower boundary of the range test.
        criterion (torch.nn.Module): wrapped loss function.
        device (str or torch.device, optional): a string ("cpu" or "cuda") with an
            optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
            Alternatively, can be an object representing the device on which the
            computation will take place. Default: None, uses the same device as `model`.
        memory_cache (boolean, optional): if this flag is set to True, `state_dict` of
            model and optimizer will be cached in memory. Otherwise, they will be saved
            to files under the `cache_dir`.
        cache_dir (string, optional): path for storing temporary files. If no path is
            specified, system-wide temporary directory is used. Notice that this
            parameter will be ignored if `memory_cache` is True.
    Example:
        >>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
        >>> lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
        >>> lr_finder.plot() # to inspect the loss-learning rate graph
        >>> lr_finder.reset() # to reset the model and optimizer to their initial state
    Reference:
    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    fastai/lr_find: https://github.com/fastai/fastai
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device=None,
        memory_cache=True,
        cache_dir=None,
    ):
        # Check if the optimizer is already attached to a scheduler
        self.optimizer = optimizer
        self._check_for_scheduler()

        self.model = model
        self.criterion = criterion
        self.history = {"lr": [], "loss": [], "acc": []}
        self.best_loss = None
        self.best_acc = None
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir

        # Save the original state of the model and optimizer so they can be restored if
        # needed
        self.model_device = next(self.model.parameters()).device
        self.state_cacher = StateCacher(memory_cache, cache_dir=cache_dir)
        self.state_cacher.store("model", self.model.state_dict())
        self.state_cacher.store("optimizer", self.optimizer.state_dict())

        # If device is None, use the same as the model
        if device:
            self.device = device
        else:
            self.device = self.model_device

    def reset(self):
        """Restores the model and optimizer to their initial states."""

        self.model.load_state_dict(self.state_cacher.retrieve("model"))
        self.optimizer.load_state_dict(self.state_cacher.retrieve("optimizer"))
        self.model.to(self.model_device)

    def range_test(
        self,
        train_loader,
        val_loader=None,
        start_lr=None,
        end_lr=10,
        epochs=100,
        step_mode="exp",
    ):
        """Performs the learning rate range test.
        Arguments:
            train_loader (torch.utils.data.DataLoader): the training set data laoder.
            val_loader (torch.utils.data.DataLoader, optional): if `None` the range test
                will only use the training loss. When given a data loader, the model is
                evaluated after each iteration on that dataset and the evaluation loss
                is used. Note that in this mode the test takes significantly longer but
                generally produces more precise results. Default: None.
            start_lr (float, optional): the starting learning rate for the range test.
                Default: None (uses the learning rate from the optimizer).
            end_lr (float, optional): the maximum learning rate to test. Default: 10.
            epoch (int, optional): the number of iterations over which the test
                occurs. Default: 100.
            step_mode (str, optional): one of the available learning rate policies,
                linear or exponential ("linear", "exp"). Default: "exp".
        """

        # Reset test results
        self.history = {"lr": [], "loss": [], "acc": []}
        self.best_loss = None
        self.best_acc = None

        # Move the model to the proper device
        self.model.to(self.device)

        # Check if the optimizer is already attached to a scheduler
        self._check_for_scheduler()

        # Set the starting learning rate
        if start_lr:
            self._set_learning_rate(start_lr)

        total_steps = epochs * len(train_loader)

        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            self.lr_schedule = ExponentialLR(self.optimizer, end_lr, total_steps)
        elif step_mode.lower() == "linear":
            self.lr_schedule = LinearLR(self.optimizer, end_lr, total_steps)
        else:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))

        for epoch in tqdm(range(epochs)):
            # Train on batch and retrieve loss
            loss, acc = self._train_epoch(train_loader)
            if val_loader:
                loss, acc = self._validate(val_loader)

            self.history["lr"].append(self.lr_schedule.get_lr()[0])

            # Track the best loss and smooth it if smooth_f is specified
            if epoch == 0:
                self.best_loss = loss
                self.best_acc = acc
            else:
                if loss < self.best_loss:
                    self.best_loss = loss
                if acc > self.best_acc:
                    self.best_acc = acc

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            self.history["acc"].append(acc)

        print("Learning rate search finished. See the graph with {finder_name}.plot()")

    def _set_learning_rate(self, new_lrs):
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Length of `new_lrs` is not equal to the number of parameter groups "
                + "in the given optimizer"
            )

        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr

    def _check_for_scheduler(self):
        for param_group in self.optimizer.param_groups:
            if "initial_lr" in param_group:
                raise RuntimeError("Optimizer already has a scheduler attached to it")

    def _train_epoch(self, train_loader):
        self.model.train()
        correct = 0
        processed = 0
        avg_loss = 0
        for inputs, labels in train_loader:
            # get samples
            inputs, labels = self._move_to_device(inputs, labels)
            # Init
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            # Update the learning rate
            self.lr_schedule.step()
            # get the index of the max log-probability
            pred = outputs.argmax(dim=1, keepdim=True)  
            correct += pred.eq(labels.view_as(pred)).sum().item()
            processed += len(inputs)
            avg_loss += loss.item()
        
        avg_loss /= len(train_loader)
        avg_acc = 100*correct/processed
        return avg_loss, avg_acc

    def _move_to_device(self, inputs, labels):
        def move(obj, device):
            if isinstance(obj, tuple):
                return tuple(move(o, device) for o in obj)
            elif torch.is_tensor(obj):
                return obj.to(device)
            elif isinstance(obj, list):
                return [move(o, device) for o in obj]
            else:
                return obj

        inputs = move(inputs, self.device)
        labels = move(labels, self.device)
        return inputs, labels

    def _validate(self, dataloader):
        # Set model to evaluation mode and disable gradient computation
        loss = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                # Move data to the correct device
                inputs, labels = self._move_to_device(inputs, labels)
                # Forward pass and loss computation
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                is_correct = pred.eq(labels.view_as(pred))
                correct += is_correct.sum().item()
                
        loss /= len(dataloader.dataset)
        acc = 100. * correct / len(dataloader.dataset)
        return loss, acc

    def plot(self, skip_start=10, skip_end=5, log_lr=True, show_lr=None,
        ax=None, metric='acc'):
        """Plots the learning rate range test.
        Arguments:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
            show_lr (float, optional): if set, adds a vertical line to visualize the
                specified learning rate. Default: None.
            ax (matplotlib.axes.Axes, optional): the plot is created in the specified
                matplotlib axes object and the figure is not be shown. If `None`, then
                the figure and axes object are created in this method and the figure is
                shown . Default: None.
        Returns:
            The matplotlib.axes.Axes object that contains the plot.
        """
        if metric not in ['loss', 'acc']:
            raise ValueError("metric can only be 'loss' or 'acc'")
        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
        if show_lr is not None and not isinstance(show_lr, float):
            raise ValueError("show_lr must be float")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        accs = self.history["acc"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
            accs = accs[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]
            accs = accs[skip_start:-skip_end]

        # Create the figure and axes object if axes was not already given
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        labels = {
            'loss': "Loss",
            'acc': "Accuracy"
        }

        # Plot loss as a function of the learning rate
        y_val = losses if metric == 'loss' else accs 
        ax.plot(lrs, y_val)
        if log_lr:
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel(labels[metric])
        plt.title("Learning Rate Schedule")

        if show_lr is not None:
            ax.axvline(x=show_lr, color="red")

        # Show only if the figure was created internally
        if fig is not None:
            plt.show()

        return ax
