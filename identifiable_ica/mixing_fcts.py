from abc import abstractmethod

import torch


class AbstractFct:
    def __init__(self, t):
        self.t = t

    @abstractmethod
    def __call__(self, x):
        pass


class PolarSampleFct(AbstractFct):
    def __call__(self, x):
        return polar_2(precomposition_sin(x, self.t))


class RotationSampleFct(AbstractFct):
    def __call__(self, x):
        w = torch.tensor([[0., 1.], [-1., 0.]])
        return rotation(precomposition_scale(x), self.t * w)


def precomposition_sin(x, t):
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] + t / 2 * torch.sin(x[:, 0] + t) + 3
    y[:, 1] = (x[:, 1] + t) / 2
    return y


def precomposition_scale(x):
    y = torch.zeros_like(x)
    y[:, 0] = 4 * x[:, 0]
    y[:, 1] = x[:, 1]
    return y


def polar_2(x):
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] * torch.sin(x[:, 1])
    y[:, 1] = x[:, 0] * torch.cos(x[:, 1])
    return y


def parabolic(x):
    y = torch.zeros_like(x)
    sigma = x[:, 0]
    tau = x[:, 1]
    y[:, 0] = sigma * tau
    y[:, 1] = .5 * (tau ** 2 - sigma ** 2)
    return y


def rotation(x, w=None):
    if w is None:
        return x
    r = torch.linalg.matrix_exp(w)
    return torch.matmul(x, r)

