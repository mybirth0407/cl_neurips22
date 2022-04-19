import random
from PIL import ImageFilter
from torchvision import transforms
from torch.utils.data import DataLoader

from mnist import ColourBiasedMNIST_BG, ColourBiasedMNIST_FG


def get_data_loader(
    batch_size,
    mode="BG",
    train=True,
    transform=None,
    data_label_correlation=1.0,
    data_indices=None,
    colormap_idxs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
):
    """
    mode: 'FG'(foreground) or 'BG'(background)
    """
    if mode == "FG":
        Colored_MNIST = ColourBiasedMNIST_FG
    elif mode == "BG":
        Colored_MNIST = ColourBiasedMNIST_BG
    else:
        raise NotImplemented

    dataset = Colored_MNIST(
        root="./",
        train=train,
        transform=transform,
        download=True,
        data_label_correlation=data_label_correlation,
        data_indices=data_indices,
        colormap_idxs=colormap_idxs,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader


class ContraAugTransform:
    def __init__(self, aug_transform, base_transform) -> None:
        self.aug_transform = aug_transform
        self.base_transform = base_transform

    def __call__(self, x):
        t1 = self.aug_transform(x)
        t2 = self.aug_transform(x)
        x = self.base_transform(x)

        return [t1, t2, x]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

transform_aug = transforms.Compose(
    [
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        # transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)
