import warnings
warnings.filterwarnings("ignore")

from model import Model
import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
# import torchvision.datasets as dsets
# import torchvision.models as models
# import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import numpy as np
# import subprocess
from numpy import random
import copy
import os

from loader import get_data_loader, transform
import wandb

parser = argparse.ArgumentParser(description="Continuum learning")
parser.add_argument(
    "--outfile", default="temp_0.1.csv", type=str, help="Output file name"
)
parser.add_argument(
    "--matr", default="results/acc_matr.npz", help="Accuracy matrix file name"
)
parser.add_argument(
    "--num_classes",
    default=10,
    help="Number of new classes introduced each time",
    type=int,
)
parser.add_argument("--init_lr", default=0.1, type=float, help="Init learning rate")

parser.add_argument("--num_epochs", default=40, type=int, help="Number of epochs")
parser.add_argument(
    "--num_task", default=10, type=int, help="the number of tasks (default: 10)"
)
parser.add_argument("--biased_r1", default=0.95, type=float, help="biased ratio")
parser.add_argument("--batch_size", default=32, type=int, help="mini-batch size")
parser.add_argument("--task_1", default="BG", type=str, help="fg or bg")
parser.add_argument("--test_db", default="BG", type=str, help="fg or bg")
parser.add_argument(
    "--save_dir", default="./exps", help="save directory for checkpoint"
)

args = parser.parse_args()

num_classes = args.num_classes
# all_train = cifar100(root='./data',
#                  train=True,
#                  classes=range(100),
#                  download=True,
#                  transform=None)
# mean_image = all_train.mean_image
# np.save("cifar_mean_image.npy", mean_image)
# mean_image = np.load("cifar_mean_image.npy")

total_classes = 100
task_classes = 10
# perm_id = np.random.permutation(task_classes)
all_classes = np.arange(task_classes)
# for i in range(len(all_classes)):
#     all_classes[i] = perm_id[all_classes[i]]

n_cl_temp = 0
num_iters = total_classes // num_classes
class_map = {}
map_reverse = {}
for i, cl in enumerate(all_classes):
    if cl not in class_map:
        class_map[cl] = int(n_cl_temp)
        n_cl_temp += 1

print("Class map:", class_map)

for cl, map_cl in class_map.items():
    map_reverse[map_cl] = int(cl)

print("Map Reverse:", map_reverse)

print("all_classes:", all_classes)
# else:
# perm_id = np.arange(args.total_classes)

indices = np.random.permutation(np.arange(60000))

dic_tr_indices = {}
dic_val_indices = {}
part_tr = int(50000 / args.num_task)
part_val = int(10000 / args.num_task)
for i in range(1, args.num_task + 1):
    dic_tr_indices[i] = indices[
        (i - 1) * part_tr + (i - 1) * part_val : i * part_tr + (i - 1) * part_val
    ]
    dic_val_indices[i] = indices[
        i * part_tr + (i - 1) * part_val : i * part_tr + i * part_val
    ]

dic_tr_dl = {}
dic_val_dl = {}
colormap_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(1, args.num_task + 1):
    # different bias on each task
    tmp = colormap_idxs[0]
    colormap_idxs = colormap_idxs[1:]
    colormap_idxs.append(tmp)

    dic_tr_dl[i] = get_data_loader(
        batch_size=args.batch_size,
        mode=args.task_1,
        train=True,
        transform=transform,
        data_label_correlation=args.biased_r1,
        data_indices=dic_tr_indices[i],
        colormap_idxs=colormap_idxs,
    )

    dic_val_dl[i] = get_data_loader(
        batch_size=args.batch_size,
        mode=args.task_1,
        train=True,
        transform=transform,
        data_label_correlation=args.biased_r1,
        data_indices=dic_val_indices[i],
        colormap_idxs=colormap_idxs,
    )

unbiased_te_dl = get_data_loader(
    batch_size=args.batch_size,
    mode=args.test_db,
    train=False,
    transform=transform,
    data_label_correlation=0.1,
)

wandb.init(name=f"lwf/biased_mnist/SimpleConvNet", project="CL_forget")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

with open(args.outfile, "w") as file:
    print("Classes, Train Accuracy, Test Accuracy", file=file)

    # shuffle classes
    # random.shuffle(all_classes)
    # class_map = {j: int(i) for i, j in enumerate(all_classes)}
    # map_reverse = {i: int(j) for i, j in enumerate(all_classes)}
    # print('Map reverse: ', map_reverse)
    # print('Class map: ', class_map)
    # print('All classes: ', all_classes)

    model = Model(10, class_map, args)
    model.cuda()
    # acc_matr = np.zeros((int(total_classes / num_classes), num_iters))
    for i, s in enumerate(range(args.num_task)):
        logs = dict()
        # Load Datasets
        print("Iteration: ", s)
        # print('Algo running: ', args.algo)
        # print("Loading training examples for classes", all_classes[s : s + num_classes])
        # train_set = cifar100(
        #     root="./data",
        #     train=True,
        #     classes=all_classes[s : s + num_classes],
        #     download=True,
        #     transform=transform,
        #     mean_image=mean_image,
        # )
        # train_loader = torch.utils.data.DataLoader(
        #     train_set, batch_size=args.batch_size, shuffle=True, num_workers=12
        # )

        # test_set = cifar100(
        #     root="./data",
        #     train=False,
        #     classes=all_classes[: s + num_classes],
        #     download=True,
        #     transform=None,
        #     mean_image=mean_image,
        # )
        # test_loader = torch.utils.data.DataLoader(
        #     test_set, batch_size=args.batch_size, shuffle=False, num_workers=12
        # )

        # Update representation via BackProp
        model.update(dic_tr_dl[i+1], class_map, args)
        model.eval()

        model.n_known = model.n_classes
        print("%d, " % model.n_known, file=file, end="")
        print("model classes : %d, " % model.n_known)

        total = 0.0
        correct = 0.0
        for images, labels in unbiased_te_dl:
            images = Variable(images).cuda()
            preds = model.classify(images)
            preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            total += labels.size(0)
            correct += (preds == labels.numpy()).sum()

        acc_test = correct / total
        # Train Accuracy
        print("%.2f ," % (100.0 * acc_test), file=file, end="")
        print(" Accuracy : %.2f ," % (100.0 * acc_test))

        total = 0.0
        correct = 0.0
        for images, labels in dic_val_dl[i+1]:
            images = Variable(images).cuda()
            preds = model.classify(images)
            preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            total += labels.size(0)
            correct += (preds == labels.numpy()).sum()

        acc_task_i = correct / total
        print("%.2f" % (100.0 * acc_task_i), file=file)
        print(f"Task [{i+1}] Accuracy : %.2f" % (100.0 * acc_task_i))

        if i > 0:
            total = 0.0
            correct = 0.0
            for images, labels in dic_val_dl[i]:
                images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()
            
            acc_task_prev = correct / total
            print("%.2f" % (100.0 * acc_task_prev), file=file)
            print(f"Task [{i+1}] Accuracy : %.2f" % (100.0 * acc_task_prev))
            logs["valid/task_pre_acc_after"] = acc_task_prev

        logs["test/acc"] = acc_test
        logs["valid/task_acc"] = acc_task_i
        wandb.log(logs)
        # # Accuracy matrix
        # for i in range(model.n_known):
        #     test_set = cifar100(
        #         root="./data",
        #         train=False,
        #         classes=all_classes[i * num_classes : (i + 1) * num_classes],
        #         download=True,
        #         transform=None,
        #         mean_image=mean_image,
        #     )
        #     test_loader = torch.utils.data.DataLoader(
        #         test_set,
        #         batch_size=min(500, len(test_set)),
        #         shuffle=False,
        #         num_workers=12,
        #     )

        #     total = 0.0
        #     correct = 0.0
        #     for indices, images, labels in test_loader:
        #         images = Variable(images).cuda()
        #         preds = model.classify(images)
        #         preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
        #         total += labels.size(0)
        #         correct += (preds == labels.numpy()).sum()
        #     acc_matr[i, int(s / num_classes)] = 100 * correct / total

        # print(
        #     "Accuracy matrix",
        #     acc_matr[: int(s / num_classes + 1), : int(s / num_classes + 1)],
        # )
        torch.save(model.state_dict(), f'{args.save_dir}/task_{i+1}.pth')
        model.train()
        # githash = subprocess.check_output(["git", "describe", "--always"])
        # np.savez(args.matr, acc_matr=acc_matr, hyper_params=args, githash=githash)
