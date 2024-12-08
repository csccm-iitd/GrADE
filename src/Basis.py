
import math
import torch
import torch.nn as nn
import numpy as np


class GaussianRBF(nn.Module):
    """Eigenbasis expansion using gaussian radial basis functions. $phi(r) = e^{-(\eps r)^2}$ with $r := || x - x0 ||_2$"
    :param deg: degree of the eigenbasis expansion
    :type deg: int
    :param adaptive: whether to adjust `centers` and `eps_scales` during training.
    :type adaptive: bool
    :param eps_scales: scaling in the rbf formula ($\eps$)
    :type eps_scales: int
    :param centers: centers of the radial basis functions (one per degree). Same center across all degrees. x0 in the radius formulas
    :type centers: int
    """

    def __init__(self, deg, adaptive=False, eps_scales=2, centers=0):
        super().__init__()
        self.deg, self.n_eig = deg, 1
        if adaptive:
            self.centers = torch.nn.Parameter(centers * torch.ones(deg + 1))
            self.eps_scales = torch.nn.Parameter(eps_scales * torch.ones((deg + 1)))
        else:
            self.centers = 0
            self.eps_scales = 2

    def forward(self, n_range, s):
        n_range_scaled = (n_range - self.centers) / self.eps_scales
        r = torch.norm(s - self.centers, p=2)
        basis = [math.e ** (-(r * n_range_scaled) ** 2)]
        return basis


class VanillaRBF(nn.Module):
    """Eigenbasis expansion using vanilla radial basis functions."
    :param deg: degree of the eigenbasis expansion
    :type deg: int
    :param adaptive: whether to adjust `centers` and `eps_scales` during training.
    :type adaptive: bool
    :param eps_scales: scaling in the rbf formula ($\eps$)
    :type eps_scales: int
    :param centers: centers of the radial basis functions (one per degree). Same center across all degrees. x0 in the radius formulas
    :type centers: int
    """

    def __init__(self, deg, adaptive=False, eps_scales=2, centers=0):
        super().__init__()
        self.deg, self.n_eig = deg, 1
        if adaptive:
            self.centers = torch.nn.Parameter(centers * torch.ones(deg + 1))
            self.eps_scales = torch.nn.Parameter(eps_scales * torch.ones((deg + 1)))
        else:
            self.centers = 0
            self.eps_scales = 2

    def forward(self, n_range, s):
        n_range_scaled = n_range / self.eps_scales
        r = torch.norm(s - self.centers, p=2)
        basis = [r * n_range_scaled]
        return basis


class MultiquadRBF(nn.Module):
    """Eigenbasis expansion using multiquadratic radial basis functions."
    :param deg: degree of the eigenbasis expansion
    :type deg: int
    :param adaptive: whether to adjust `centers` and `eps_scales` during training.
    :type adaptive: bool
    :param eps_scales: scaling in the rbf formula ($\eps$)
    :type eps_scales: int
    :param centers: centers of the radial basis functions (one per degree). Same center across all degrees. x0 in the radius formulas
    :type centers: int
    """

    def __init__(self, deg, adaptive=False, eps_scales=2, centers=0):
        super().__init__()
        self.deg, self.n_eig = deg, 1
        if adaptive:
            self.centers = torch.nn.Parameter(centers * torch.ones(deg + 1))
            self.eps_scales = torch.nn.Parameter(eps_scales * torch.ones((deg + 1)))
        else:
            self.centers = 0
            self.eps_scales = 2

    def forward(self, n_range, s):
        n_range_scaled = n_range / self.eps_scales
        r = torch.norm(s - self.centers, p=2)
        basis = [1 + torch.sqrt(1 + (r * n_range_scaled) ** 2)]
        return basis


class Fourier(nn.Module):
    """Eigenbasis expansion using fourier functions."
    :param deg: degree of the eigenbasis expansion
    :type deg: int
    :param adaptive: does nothing (for now)
    :type adaptive: bool
    """

    def __init__(self, deg, adaptive=False):
        super().__init__()
        self.deg, self.n_eig = deg, 2

    def forward(self, n_range, s):
        s_n_range = s * n_range
        basis = [torch.cos(s_n_range), torch.sin(s_n_range)]
        return basis


class Polynomial(nn.Module):
    """Eigenbasis expansion using polynomials."
    :param deg: degree of the eigenbasis expansion
    :type deg: int
    :param adaptive: does nothing (for now)
    :type adaptive: bool
    """

    def __init__(self, deg, adaptive=False):
        super().__init__()
        self.deg, self.n_eig = deg, 1

    def forward(self, n_range, s):
        basis = [s ** n_range]
        return basis


class Chebychev(nn.Module):
    """Eigenbasis expansion using chebychev polynomials."
    :param deg: degree of the eigenbasis expansion
    :type deg: int
    :param adaptive: does nothing (for now)
    :type adaptive: bool
    """

    def __init__(self, deg, adaptive=False):
        super().__init__()
        self.deg, self.n_eig = deg, 1

    def forward(self, n_range, s):
        max_order = n_range[-1].int().item()
        basis = [1]
        # Based on numpy's Cheb code
        if max_order > 0:
            s2 = 2 * s
            basis += [s.item()]
            for i in range(2, max_order):
                basis += [basis[-1] * s2 - basis[-2]]
        return [torch.tensor(basis).to(n_range)]


class PiecewiseConstant(nn.Module):
    """Eigenbasis expansion using PiecewiseConstant."
        :param deg: degree of the eigenbasis expansion
        :type deg: int
        :param adaptive: does nothing (for now)
        :type adaptive: bool
        """

    def __init__(self, deg, adaptive=False):
        super().__init__()
        self.deg, self.n_eig = deg, 1

    def forward(self, n_range, s):
        idx = self.idx
        zero_tnsor = torch.zeros((len(n_range))).to(n_range.device)
        zero_tnsor[idx] = 1
        basis = [zero_tnsor]
        return basis
