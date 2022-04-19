from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import utils

"""
Simple ConvNet for MNIST
"""
from module import MLP, Encoder


class SimpleConvNet(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.encoder = Encoder(kernel_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x, return_feats=False):
        feats = self.encoder(x)
        logits = self.avgpool(feats)
        logits = torch.flatten(logits, 1)
        logits = self.fc(logits)

        if return_feats:
            return logits, feats

        return logits

    def _get_params(self):
        return list(self.parameters())

    def estimate_fisher(self, data_loader, sample_size, batch_size=32):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        for x, y in data_loader:
            x = x.view(batch_size, -1)
            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
            loglikelihoods.append(
                F.log_softmax(self(x), dim=1)[range(batch_size), y.data]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        loglikelihood_grads = zip(
            *[
                autograd.grad(
                    l, self.parameters(), retain_graph=(i < len(loglikelihoods))
                )
                for i, l in enumerate(loglikelihoods, 1)
            ]
        )
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g**2).mean(0) for g in loglikelihood_grads]
        param_names = [n.replace(".", "__") for n, p in self.named_parameters()]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace(".", "__")
            self.register_buffer("{}_mean".format(n), p.data.clone())
            self.register_buffer("{}_fisher".format(n), fisher[n].data.clone())

    def ewc_loss(self, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace(".", "__")
                mean = getattr(self, "{}_mean".format(n))
                fisher = getattr(self, "{}_fisher".format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            return (self.lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return Variable(torch.zeros(1)).cuda() if cuda else Variable(torch.zeros(1))

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


class SimpleConvNet_plus(SimpleConvNet):
    def __init__(self, kernel_size=3):
        super().__init__(kernel_size)

        self.mlp = MLP(hidden_size=128, output_size=128)

    def forward(self, x):
        feats = self.encoder(x)
        gap = self.avgpool(feats)
        gap = torch.flatten(gap, 1)
        logits = self.mlp(gap)
        logits = self.fc(logits)

        return logits


class SimpleMLP(nn.Module):
    def __init__(self, input_size=28*28*3, hidden_size=128, output_size=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size*2),
            nn.BatchNorm1d(hidden_size*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size*2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x, return_feats=False):
        return self.mlp(x)

    def _get_params(self):
        return list(self.parameters())

    def estimate_fisher(self, data_loader, sample_size, batch_size=32):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        for x, y in data_loader:
            x = x.view(batch_size, -1)
            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
            loglikelihoods.append(
                F.log_softmax(self(x), dim=1)[range(batch_size), y.data]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        loglikelihood_grads = zip(
            *[
                autograd.grad(
                    l, self.parameters(), retain_graph=(i < len(loglikelihoods))
                )
                for i, l in enumerate(loglikelihoods, 1)
            ]
        )
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g**2).mean(0) for g in loglikelihood_grads]
        param_names = [n.replace(".", "__") for n, p in self.named_parameters()]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace(".", "__")
            self.register_buffer("{}_mean".format(n), p.data.clone())
            self.register_buffer("{}_fisher".format(n), fisher[n].data.clone())

    def ewc_loss(self, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace(".", "__")
                mean = getattr(self, "{}_mean".format(n))
                fisher = getattr(self, "{}_fisher".format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            return (self.lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return Variable(torch.zeros(1)).cuda() if cuda else Variable(torch.zeros(1))

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
