import math

import torch
from torch import nn


def logcosh(x):
    """
    Calculates log of cosh.
    Args:
        x: torch.tensor

    Returns: torch.tensor

    """
    return x - math.log(2.0) + torch.nn.functional.softplus(-2.0 * x)


def visible_times_weights(visible, weights):
    """
    Multiply the visible and weight tensors.

    Args:
        visible (torch.tensor ~ (batch_size, num_visible))
        weights (torch.tensor ~ (batch_size, num_visible, num_hidden))

    Returns:
        torch.tensor ~ (batch_size, num_hidden)

    """
    return torch.bmm(visible.unsqueeze(dim=1), weights).squeeze(dim=1)


def hidden_times_weights(hidden, weights):
    """
    Multiply the hidden and weight tensors.

    Args:
        hidden (torch.tensor ~ (batch_size, num_hidden))
        weights (torch.tensor ~ (batch_size, num_visible, num_hidden))

    Returns:
        torch.tensor ~ (batch_size, num_visible)

    """
    return torch.bmm(weights, hidden.unsqueeze(dim=2)).squeeze(dim=2)


class BiasNet(nn.Module):
    """
    Neural network that parametrizes the bias of the NBM
    Maps from nx to ny.
    """

    def __init__(self, nx, ny, visible_unit_type):
        """
        Args:
            nx (int): input data size
            ny (int): output_size
            visible_unit_type (str): what type of visible units are being modeled
        """
        super().__init__()

        net = nn.Sequential(nn.Linear(nx, 32), nn.ReLU(), nn.Linear(32, ny))

        if visible_unit_type == "ising":
            net.append(nn.Tanh())

        self.net = net

    def forward(self, x):
        """
        Args:
            x (torch.tensor ~ (batch_size, nx))

        Returns:
            y (torch.tensor ~ (batch_size, ny))
        """
        return self.net(x)


class PrecisionNet(nn.Module):
    """
    Neural network that parametrizes the precision of the NBM
    Maps from nx to ny.
    """

    def __init__(self, nx, ny, pmin=1e-3, pmax=1e3):
        """
        Args:
            nx (int): input data size
            ny (int): output_size
            pmin (optional; float = 1e-3)
            pmax (optional; float = 1e3)
        """
        super().__init__()

        self.net = nn.Sequential(nn.Linear(nx, 32), nn.ReLU(), nn.Linear(32, ny))

        self.lpmin = math.log(pmin)
        self.lpmax = math.log(pmax)

    def forward(self, x):
        """
        Args:
            x (torch.tensor ~ (batch_size, nx))

        Returns:
            P (torch.tensor ~ (batch_size, ny))
        """
        return self.net(x).clip(self.lpmin, self.lpmax).exp()


class WeightsNet(nn.Module):
    """
    Neural network that parametrizes the weights of the NBM
    Maps from nx to (ny, nh).
    """

    def __init__(self, nx, ny, nh):
        """
        Args:
            nx (int): input data size
            ny (int): output_size
            nh (int): hidden unit size
        """
        super().__init__()
        self._ny = ny
        self._nh = nh
        self._norm = math.sqrt(self._ny)
        self.net = nn.Sequential(
            nn.Linear(nx, ny * nh),
        )

    def forward(self, x):
        """
        Args:
            x (torch.tensor ~ (batch_size, nx))

        Returns:
            W (torch.tensor ~ (batch_size, ny, nh))
        """
        return self.net(x).reshape((-1, self._ny, self._nh)) / self._norm


class NBM(nn.Module):
    """
    Basic Neural Boltzmann Machine implementation.
    """

    def __init__(self, nx, ny, nh, visible_unit_type="gaussian"):
        """
        Constructs the NBM.

        Args:
            nx (int): input data shape
            ny (int): output data shape
            nh (int): hidden dim
            visible_unit_type (str): what type of visible units are being modeled
        """
        super().__init__()

        assert visible_unit_type in {"gaussian", "ising"}
        self.visible_unit_type = visible_unit_type
        self.bias_net = BiasNet(nx, ny, visible_unit_type=visible_unit_type)
        self.precision_net = PrecisionNet(nx, ny)
        self.weights_net = WeightsNet(nx, ny, nh)

    def _free_energy(self, y, bias, precision, weights):
        """
        Computes the free energy (after tracing over h)

        U(y; x) = 1/2 (y - \mu(x))^T P (y - \mu(x)) - \sum \log \cosh W^T (y - \mu(x))

        Args:
            y (torch.tensor ~ (batch_size, ny))
            bias (torch.tensor ~ (batch_size, ny))
            precision (torch.tensor ~ (batch_size, ny))
            weights (torch.tensor ~ (batch_size, ny, nh))

        Returns:
            free energy (torch.tensor ~ (batch_size, ))
        """
        diff = y - bias
        self_energy = 0.5 * (diff * precision * diff).sum(dim=-1)
        phi = logcosh(visible_times_weights(diff, weights)).sum(dim=-1)
        free_energy = self_energy - phi
        return free_energy

    @torch.no_grad()
    def _sample_hid(self, y, bias, precision, weights):
        """
        Sample hidden state given visible state and context.

        Args:
            y (torch.tensor ~ (batch_size, ny))
            bias (torch.tensor ~ (batch_size, ny))
            precision (torch.tensor ~ (batch_size, ny))
            weights (torch.tensor ~ (batch_size, ny, nh))

        Returns:
            h (torch.tensor ~ (batch_size, nh))
        """
        diff = y - bias
        logits = visible_times_weights(diff, weights)
        proba = torch.sigmoid(2.0 * logits)
        sample = 2.0 * torch.bernoulli(proba) - 1.0
        return sample

    @torch.no_grad()
    def _sample_vis(self, h, bias, precision, weights, denoise=False):
        """
        Sample hidden state given visible state and context.

        Args:
            h (torch.tensor ~ (batch_size, nh))
            bias (torch.tensor ~ (batch_size, ny))
            precision (torch.tensor ~ (batch_size, ny))
            weights (torch.tensor ~ (batch_size, ny, nh))
            denoise (optional; bool = False)

        Returns:
            y (torch.tensor ~ (batch_size, ny))
        """

        if self.visible_unit_type == "gaussian":
            y = self._sample_vis_gaussian(h, bias, precision, weights, denoise=denoise)
        elif self.visible_unit_type == "ising":
            y = self._sample_vis_ising(h, bias, precision, weights, denoise=denoise)
        else:
            raise ValueError(f"Unrecognized visible type: {self.visible_unit_type}")

        return y

    @torch.no_grad()
    def _sample_vis_gaussian(self, h, bias, precision, weights, denoise=False):
        """
        Sample hidden state given visible state and context.

        Args:
            h (torch.tensor ~ (batch_size, nh))
            bias (torch.tensor ~ (batch_size, ny))
            precision (torch.tensor ~ (batch_size, ny))
            weights (torch.tensor ~ (batch_size, ny, nh))
            denoise (optional; bool = False)

        Returns:
            y (torch.tensor ~ (batch_size, ny))
        """
        field = hidden_times_weights(h, weights)
        y = bias + field / precision
        if not denoise:
            noise = torch.randn_like(bias)
            y = y + noise / precision.sqrt()
        return y

    @torch.no_grad()
    def _sample_vis_ising(self, h, bias, precision, weights, denoise=False):
        field = hidden_times_weights(h, weights)
        logits = precision * bias + field
        proba = torch.sigmoid(2.0 * logits)
        if denoise:
            sample = torch.tanh(proba)
        else:
            sample = 2.0 * torch.bernoulli(proba) - 1.0

        return sample

    def compute_loss(self, y, x, mc_steps):
        """
        Evaluate the losses used for training the model

        Args:
            y (torch.tensor ~ (batch_size, ny))
            x (torch.tensor ~ (batch_size, nx))
            mc_steps (int): number of Gibbs sampling steps

        Returns:
            loss: the usual CD loss
            MSE_bias_loss: the bias_net reconstruction loss
            MSE_variance_loss: the precision_net variance reconstruction loss
        """
        nsteps = max(1, mc_steps)

        bias = self.bias_net(x)
        precision = self.precision_net(x)
        weights = self.weights_net(x)

        y_model = bias.clone()
        for _ in range(nsteps):
            h_model = self._sample_hid(y_model, bias, precision, weights)
            y_model = self._sample_vis(h_model, bias, precision, weights)

        # This is the loss used for backprop
        y_flat = torch.flatten(y, start_dim=1)
        pos_phase = self._free_energy(y_flat, bias, precision, weights)
        neg_phase = self._free_energy(y_model, bias, precision, weights)
        CD_loss = (pos_phase - neg_phase).mean()

        # This is a supplemental loss to understand how the bias net is learning
        diff = y_flat - bias
        bias_mse = (diff * diff).mean()

        # This is a supplemental loss to understand how the precision net is learning
        # It is more natural to interpret as a variance instead of as precision
        variance = 1 / precision
        y_variance = torch.pow((y_flat - y_flat.mean()), 2)
        diff = y_variance - variance
        variance_mse = (diff * diff).mean()

        return {
            "CD_loss": CD_loss,
            "MSE_bias_loss": bias_mse,
            "MSE_variance_loss": variance_mse,
        }

    @torch.no_grad()
    def sample(self, x, mc_steps, denoise=False):
        """
        Generate (noisy) samples from the model.

        Args:
            x (torch.tensor ~ (batch_size, nx))
            mc_steps (int): number of Gibbs sampling steps
            denoise (optional; bool = False)

        Returns:
            y (torch.tensor ~ (batch_size, ny))
        """
        nsteps = max(1, mc_steps)

        bias = self.bias_net(x)
        precision = self.precision_net(x)
        weights = self.weights_net(x)

        y_model = bias.clone()
        for _ in range(nsteps):
            h_model = self._sample_hid(y_model, bias, precision, weights)
            y_model = self._sample_vis(h_model, bias, precision, weights)

        if denoise:
            y_model = self._sample_vis(h_model, bias, precision, weights, denoise=True)

        return y_model
