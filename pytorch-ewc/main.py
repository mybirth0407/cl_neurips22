#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
import torch
from train import train
from model import SimpleConvNet, SimpleMLP
import utils
from loader import ContraAugTransform, get_data_loader, transform, transform_aug
import wandb

parser = ArgumentParser("EWC PyTorch Implementation")

parser.add_argument("--epochs-per-task", type=int, default=20)
parser.add_argument("--lamda", type=float, default=40)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--fisher-estimation-sample-size", type=int, default=512)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--no-gpus", action="store_false", dest="cuda")
parser.add_argument("--consolidate", action="store_true")

parser.add_argument("--biased_r1", default=0.0, type=float, help="biased ratio")
parser.add_argument("--biased_r2", default=None, type=float, help="biased ratio")

parser.add_argument("--batch_size", default=32, type=int, help="mini-batch size")
parser.add_argument("--task_1", default="BG", type=str, help="fg or bg")
parser.add_argument("--task_2", default="BG", type=str, help="fg or bg")
parser.add_argument("--test_db", default="BG", type=str, help="fg or bg")
parser.add_argument(
    "--save_dir", default="./exps", help="save directory for checkpoint"
)

if __name__ == "__main__":
    args = parser.parse_args()

    # decide whether to use cuda or not.
    cuda = torch.cuda.is_available() and args.cuda

    # transform
    transform_tr = transform
    transform_val = transform

    indices = np.random.permutation(np.arange(60000))

    num_step = 10
    dic_tr_indices = {}
    dic_val_indices = {}
    part_tr = int(50000 / num_step)
    part_val = int(10000 / num_step)
    for i in range(1, 11):
        dic_tr_indices[i] = indices[
            (i - 1) * part_tr + (i - 1) * part_val : i * part_tr + (i - 1) * part_val
        ]
        dic_val_indices[i] = indices[
            i * part_tr + (i - 1) * part_val : i * part_tr + i * part_val
        ]

    dic_tr_dl = {}
    dic_val_dl = {}
    colormap_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(1, 11):
        # different bias on each task
        tmp = colormap_idxs[0]
        colormap_idxs = colormap_idxs[1:]
        colormap_idxs.append(tmp)

        dic_tr_dl[i] = get_data_loader(
            batch_size=args.batch_size,
            mode=args.task_1,
            train=True,
            transform=transform_tr,
            data_label_correlation=args.biased_r1,
            data_indices=dic_tr_indices[i],
            colormap_idxs=colormap_idxs,
        )

        dic_val_dl[i] = get_data_loader(
            batch_size=args.batch_size,
            mode=args.task_1,
            train=True,
            transform=transform_val,
            data_label_correlation=args.biased_r1,
            data_indices=dic_val_indices[i],
            colormap_idxs=colormap_idxs,
        )

    unbiased_te_dl = get_data_loader(
        batch_size=args.batch_size,
        mode=args.test_db,
        train=False,
        transform=transform_val,
        data_label_correlation=0.1,
    )

    # prepare the model.
    simple_model = SimpleMLP()

    # prepare the cuda if needed.
    if cuda:
        simple_model.cuda()

    # Define Tensorboard Writer
    writer = wandb.init(name="ewc/biased_mnist/mlp", project="CL_forget")

    # run the experiment.
    train(
        simple_model,
        dic_tr_dl,
        dic_val_dl,
        unbiased_te_dl,
        writer,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        consolidate=args.consolidate,
        fisher_estimation_sample_size=args.fisher_estimation_sample_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cuda=cuda,
    )
