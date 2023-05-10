import matplotlib.cm as cm
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback


def accelerator():
    """
    Function to return what is the best accelerator available at the moment.

    Will return "gpu" if possible, then "mps", then "cpu".

    Returns: accelerate (str)

    """
    if torch.cuda.is_available():
        accelerate = "gpu"
    elif torch.device("mps"):
        accelerate = "mps"
    else:
        accelerate = "cpu"

    return accelerate


def plot_image_grid(image_array, shape, filename=None, show=True, row_labels=None):
    """
    Plots a grid of image.
    Args:
        image_array: Array of images (np.array)
        shape: Shape of each image (number of columns, number of weights)
        filename: (Optional) name to save file (str)
        show: (Optional) whether to display image (bool)
        row_labels: (Optional) labels for each row (list of str)

    Returns: None

    """
    ncols, nrows = image_array.shape[:-1]

    # Prep for image
    f = plt.figure(figsize=(2 * ncols, 2 * nrows))
    grid = gs.GridSpec(nrows, ncols)
    axes = [[plt.subplot(grid[i, j]) for j in range(ncols)] for i in range(nrows)]

    # Prep for labels
    fontdict = {"fontsize": 48}
    tick = int(shape[0] // 2)
    if row_labels is not None:
        assert (
            len(row_labels) == nrows
        ), f"Need to have a label for each of the {nrows} rows."

    for i in range(nrows):
        for j in range(ncols):
            axes[i][j].imshow(
                np.reshape(image_array[j][i], shape),
                cmap=cm.gray,
                interpolation="none",
            )
            if j == 0 and row_labels is not None:
                axes[i][j].set_yticks([tick], [row_labels[i]], fontdict=fontdict)
                axes[i][j].set(xticks=[])
            else:
                axes[i][j].set(yticks=[])
                axes[i][j].set(xticks=[])

    plt.tight_layout(pad=0.5, h_pad=0.2, w_pad=0.2)

    if show:
        plt.show()
    if filename is not None:
        f.savefig(filename)
    plt.close()


def plot_samples(
    model,
    mcsteps,
    img_shape=(28, 28),
    denoise=False,
    num_classes=10,
    filename=None,
    num_examples=4,
):
    """
    Plots a grid of generated images.

    Args:
        model: instance of a model (NBM)
        mcsteps: number of sampling steps (int)
        img_shape: size of image (optional; (28,28) for MNIST)
        denoise: whether to denoise Gaussian images (bool)
        num_classes: number of classes in the dataset (int)
        filename: (optional) name of where to save images (str)
        num_expamples: (optional int) number of examples to generate

    Returns:
        None
    """
    grid = []
    for i in range(num_classes):
        one_hot_input = torch.full((num_examples,), i, dtype=int)
        x = (
            torch.nn.functional.one_hot(one_hot_input, num_classes)
            .float()
            .to(accelerator())
        )
        sample = model.sample(x, mcsteps, denoise=denoise)
        grid.append(sample.detach().cpu().numpy())
    plot_image_grid(np.array(grid), img_shape, filename=filename, show=False)


def plot_weights(
    model,
    mcsteps,
    img_shape=(28, 28),
    denoise=False,
    num_classes=10,
    filename=None,
):
    """
    Plots a grid of generated digits.

    Args:
        model: instance of a model (NBM)
        mcsteps: number of sampling steps (int)
        img_shape: size of image (optional; (28,28) for MNIST)
        denoise: whether to denoise Gaussian images (bool)
        num_classes: number of classes in the dataset (int)
        filename: (optional) name of where to save images (str)

    Returns:
        None
    """
    grid = []
    for i in range(num_classes):
        one_hot_input = torch.full((1,), i, dtype=int)
        x = (
            torch.nn.functional.one_hot(one_hot_input, num_classes)
            .float()
            .to(accelerator())
        )
        sample = model.sample(x, mcsteps, denoise=denoise)
        bias = model.bias_net(x)
        variance = 1 / model.precision_net(x)
        weights = model.weights_net(x).mean(-1)
        images = torch.concat((sample, bias, variance, weights))
        grid.append(images.detach().cpu().numpy())

    row_labels = ["Sample", "Bias", "1/Precision", "Avg Weights"]
    plot_image_grid(
        np.array(grid), img_shape, filename=filename, show=False, row_labels=row_labels
    )


class PlotSamples(Callback):
    """
    Callback to plot figures at the end of each epoch.
    """

    def __init__(self, logger_dir, image_shape, num_classes, denoise=True):
        """

        Args:
            logger_dir: directory where we should log figures (str)
            image_shape: shape of the images (tuple of int x int)
            num_classes: number of classes in dataset (int)
            denoise: (Optional) whether to denoise Gaussian samples (bool)
        """
        self.logger_dir = logger_dir
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.denoise = denoise

    def on_train_epoch_end(self, trainer, model):
        if trainer.is_global_zero:
            # Plot samples
            sample_dir = self.logger_dir / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)

            filename = sample_dir / f"sample_epoch_{trainer.current_epoch}"
            plot_samples(
                model.model,
                model.hparams.mc_steps,
                img_shape=self.image_shape[1:],
                denoise=True,
                num_classes=self.num_classes,
                filename=filename,
            )

            # Plot weights
            weights_dir = self.logger_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)

            filename = weights_dir / f"weights_epoch_{trainer.current_epoch}"
            plot_weights(
                model.model,
                model.hparams.mc_steps,
                img_shape=self.image_shape[1:],
                denoise=True,
                num_classes=self.num_classes,
                filename=filename,
            )
