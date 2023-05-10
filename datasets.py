import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST


class IsingTransform:
    def __call__(self, x, noise_scale=1e-3):
        """
        Turn grayscale images between 0 and 1 into binary variables of +/- 1.

        Args:
            x: input (tensors)
            noise_scale (float): scalar to set noise scale.
                Between 1e-3 and 1e-2 leads to same training convergence.
                Needs to be non-zero due to degenerate variance of some pixels in unmodified MNIST.

        Returns:  output (tensors)

        """
        x_clamped = torch.clamp(x, noise_scale, 1 - noise_scale)
        return 2 * torch.bernoulli(x_clamped) - 1


class GaussianTransform:
    def __call__(self, x, noise_scale=1e-3):
        """
        Turn grayscale images between 0 and 1 into centered Gaussian variables.

        Args:
            x (tensor): input
            noise_scale (float): scalar to set noise scale.
                Between 1e-3 and 1e-2 leads to same training convergence.
                Needs to be non-zero due to degenerate variance of some pixels in unmodified MNIST.

        Returns: output (tensors)

        """
        center = 2.0 * x - 1.0
        noise = noise_scale * torch.randn_like(x)

        return center + noise


class TransformTargetsToOneHot:
    def __init__(self, num_classes):
        """
        Transforms targets to one_hot variables.

        Args:
            num_classes (int): number of classes
        """
        self.num_classes = num_classes

    def __call__(self, x):
        """
        Args:
            x (tensors): input

        Returns: output (tensors)

        """
        return (
            F.one_hot(torch.tensor(x).view(-1, 1), self.num_classes).squeeze().float()
        )


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=32,
        num_workers=1,
        output_type="gaussian",
        dataset_name="MNIST",
    ):
        """
        Create DataModule.

        Args:
            data_dir (str): directory to download data
            batch_size (int) : size of a batch in a dataloader
            num_workers (int) : number of workers in dataloader
            output_type (str) : whether outputs should be gaussian or ising
            dataset_name (str) : which dataset to create
        """

        super().__init__()

        self.num_classes = 10
        self.dims = (1, 28, 28)
        self._val_size = self.num_classes * 500

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if output_type == "gaussian":
            output_transform = GaussianTransform()
        elif output_type == "ising":
            output_transform = IsingTransform()
        else:
            raise ValueError(f"Unrecognized output type {output_type}")
        self.output_type = output_type

        if dataset_name == "MNIST":
            dataset = MNIST
        elif dataset_name == "FashionMNIST":
            dataset = FashionMNIST
        else:
            raise ValueError(f"Unrecognized dataset name {dataset_name}")
        self.dataset = dataset

        self.transform = transforms.Compose([transforms.ToTensor(), output_transform])
        self.target_transform = TransformTargetsToOneHot(self.num_classes)

        # These are datasets that will get created in self.setup
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        # Download the data
        self.dataset(self.data_dir, train=True, download=True)
        self.dataset(self.data_dir, train=False, download=True)

    def setup(self, stage):
        train_val = self.dataset(
            self.data_dir,
            train=True,
            transform=self.transform,
            target_transform=self.target_transform,
        )

        train_size = len(train_val) - self._val_size

        self.train, self.val = random_split(train_val, [train_size, self._val_size])

        self.test = self.dataset(
            self.data_dir,
            train=False,
            transform=self.transform,
            target_transform=self.target_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
