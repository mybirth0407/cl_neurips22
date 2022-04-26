import os
import os.path
import shutil
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn import init


def save_checkpoint(model, model_dir, epoch, precision, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, "{}-best".format(model.name))

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(
        {
            "state": model.state_dict(),
            "epoch": epoch,
            "precision": precision,
        },
        path,
    )

    # override the best model if it's the best.
    if best:
        shutil.copy(path, path_best)
        print(
            "=> updated the best model of {name} at {path}".format(
                name=model.name, path=path_best
            )
        )

    # notify that we successfully saved the checkpoint.
    print("=> saved the model {name} to {path}".format(name=model.name, path=path))


def load_checkpoint(model, model_dir, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, "{}-best".format(model.name))

    # load the checkpoint.
    checkpoint = torch.load(path_best if best else path)
    print(
        "=> loaded checkpoint of {name} from {path}".format(
            name=model.name, path=(path_best if best else path)
        )
    )

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint["state"])
    epoch = checkpoint["epoch"]
    precision = checkpoint["precision"]
    return epoch, precision


@torch.no_grad()
def validate(model, optimizer, criteriton, data_loader, cuda=False, verbose=True):
    mode = model.training
    model.train(mode=False)

    total_num_correct = 0.0
    total_num_test = 0.0
    total_loss = 0.0

    for images, labels in data_loader:
        # if mlp
        # images = images.view(images.shape[0], -1)
        images = Variable(images.cuda()) if cuda else Variable(images)
        labels = Variable(labels.cuda()) if cuda else Variable(labels)
        batch_size = images.shape[0]
        total_num_test += batch_size

        scores = model(images)
        ce_loss = criteriton(scores, labels)
        ewc_loss = model.ewc_loss(cuda=cuda)
        loss = ce_loss + ewc_loss

        # calculate the training precision.
        _, predicted = scores.max(1)
        total_num_correct += (predicted == labels).sum().float().data
        total_loss += loss.data * batch_size

    avg_loss = total_loss / total_num_test
    avg_acc = float(total_num_correct) / total_num_test

    return avg_acc
