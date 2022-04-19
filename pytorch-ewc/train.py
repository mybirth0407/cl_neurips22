# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2022-04-05 16:20:48
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2022-04-16 04:06:39


import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
import utils
import visual


def train(
    model,
    dic_tr_dl,  # dict of dataloader
    dic_val_dl,  # dict of dataloader
    unbiased_te_dl,  # dataloader
    wandb_writer,
    epochs_per_task=20,
    batch_size=32,
    consolidate=True,
    fisher_estimation_sample_size=512,
    lr=1e-3,
    weight_decay=1e-5,
    cuda=False,
):
    # prepare the loss criteriton and the optimizer.
    criteriton = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # set the model's mode to training mode.
    model.train()

    for i in range(1, 11):  # task number
        print(f"{i}th task")
        logs = dict()
        for epoch in range(1, epochs_per_task + 1):
            # train[i]에 대해 train
            train_loader = dic_tr_dl[i]

            loss_sum = 0.0
            total_num_train = 0
            for images, labels in train_loader:
                images = images.view(images.shape[0], -1)
                images = Variable(images.cuda()) if cuda else Variable(images)
                labels = Variable(labels.cuda()) if cuda else Variable(labels)
                bsize = images.shape[0]
                total_num_train += bsize

                # run the model and backpropagate the errors.
                optimizer.zero_grad()
                scores = model(images)
                ce_loss = criteriton(scores, labels)
                ewc_loss = model.ewc_loss(cuda=cuda)
                loss = ce_loss + ewc_loss
                loss_sum += bsize * loss.item()
                loss.backward()
                optimizer.step()

            avg_loss = loss_sum / total_num_train
            print(f"{epoch}/{epochs_per_task}, train loss: {avg_loss}")

        if consolidate:
            # estimate the fisher information of the parameters and consolidate
            # them in the network.
            print(
                "=> Estimating diagonals of the fisher information matrix...",
                flush=True,
                end="",
            )
            model.consolidate(
                model.estimate_fisher(train_dataset, fisher_estimation_sample_size)
            )
            print(" Done!")

        # unbiased에 대해 validation
        accuracy_unbiased = utils.validate(
            model, optimizer, criteriton, unbiased_te_dl, cuda=cuda
        )
        print(f"accuracy of unbiased {accuracy_unbiased}")
        logs["test/acc"] = accuracy_unbiased

        if i > 1:
            accuracy_prev_task = utils.validate(
                model, optimizer, criteriton, dic_val_dl[i - 1], cuda=cuda
            )
            print(f"accuracy of prev task {accuracy_prev_task}")
            logs["valid/task_pre_acc_after"] = accuracy_prev_task
        # val[i]에 대해 validation
        accuracy_current_task = utils.validate(
            model, optimizer, criteriton, dic_val_dl[i], cuda=cuda
        )
        print(f"accuracy of current task {accuracy_current_task}")
        logs["valid/task_acc"] = accuracy_current_task
        wandb_writer.log(logs)
