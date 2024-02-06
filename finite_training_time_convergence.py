import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import LogNorm


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# We define our ResNet with fully connected hidden layers.
# We could use nn.Sequential directly,  but here it's easier to keep track of the parameters


class ResNet(nn.Module):
    def __init__(self, V, W, d, N):
        super(ResNet, self).__init__()
        self.V = V
        self.W = W
        self.d = d
        self.N = N
        self.sigma = torch.nn.GELU()

    def forward(self, x):
        # Iterates x_{t+1} = x_{t} + V_t \sigma(W_t x_t) / N
        N = self.N
        sigma = self.sigma
        for W_i, V_i in zip(self.W, self.V):
            x = x + sigma(x.mm(W_i)).mm(V_i) / N
        return x


def make_data(n, d):
    return torch.randn(n, d)


set_all_seeds(0)

# hidden layer size
M = 32
d = 16

# Adjust the number of epochs here if you want to train for longer
n_epochs = 50
n = 100
W_to_repeat = torch.randn(d, M)

X = make_data(n, d=d)
Y = make_data(n, d=d)

Ns = [2**i for i in range(1, 11)] + [2**14]

V_list = []
W_list = []


def loss(pred, Y):
    return ((pred - Y) ** 2).mean()


D = {}
for N in Ns:
    V = torch.zeros(N, M, d)
    V = nn.Parameter(V)
    W = W_to_repeat.repeat(N, 1, 1)
    W = nn.Parameter(W)
    net = ResNet(V, W, d, N)
    optimizer = torch.optim.SGD(net.parameters(), lr=N * 0.01)

    V_t = []
    W_t = []
    for i in range(n_epochs):
        V_t.append(V.clone().detach())
        W_t.append(W.clone().detach())
        optimizer.zero_grad()

        output = loss(net(X), Y)
        output.backward()
        optimizer.step()

    V_list.append(V_t)
    W_list.append(W_t)


norms_t = []
for t in range(len(V_list[-1])):
    V_true = V_list[-1][t]
    N = len(V_true)
    norms = []
    for V in V_list[:-1]:
        norm = 0
        for i, p in enumerate(V[t]):
            norm += (p - V_true[int(i * N / len(V[t]))]).square().sum()
        norms.append(np.sqrt(norm / len(V[t])))
    norms_t.append(norms)
norms_t = np.array(norms_t)

norms_t = norms_t[4:, :]  # Remove the very beginning to avoids 0s

vmin = np.min(norms_t[np.where(norms_t > 0)])

p = 0.5

plt.figure(figsize=(5 * p, 5 * p))
plt.imshow(
    norms_t[:].T,
    interpolation="nearest",
    aspect="auto",
    norm=LogNorm(vmin=vmin),
    cmap="BuPu",
)
plt.xlabel("Training iteration")
plt.ylabel("Depth $L$")
plt.yticks(ticks=np.arange(len(Ns))[:-1], labels=Ns[:-1])
plt.colorbar()
plt.title("$\|\mathcal{V}^L(., t)  - \mathcal{V}(., t)\|$")
plt.tight_layout()
plt.savefig("figures/fig_V.pdf")

norms_t = []
for t in range(len(W_list[-1])):
    W_true = W_list[-1][t]
    N = len(W_true)
    norms = []
    for W in W_list[:-1]:
        norm = 0
        for i, p in enumerate(W[t]):
            norm += (p - W_true[int(i * N / len(W[t]))]).square().sum()
        norms.append(np.sqrt(norm / len(W[t])))
    norms_t.append(norms)
norms_t = np.array(norms_t)

norms_t = norms_t[4:, :]

p = 0.5
opt_indices = np.arange(1, n_epochs - 1, n_epochs // 10)

vmin = np.min(norms_t[np.where(norms_t > 0)])
vmax = np.max(norms_t[np.where(norms_t > 0)])

plt.figure(figsize=(5 * p, 5 * p))
plt.imshow(
    norms_t[:].T,
    interpolation="nearest",
    aspect="auto",
    norm=LogNorm(vmax=vmax, vmin=vmin),
    cmap="BuPu",
)
plt.xlabel("Training iteration")
plt.ylabel("Depth $L$")
plt.yticks(ticks=np.arange(len(Ns))[:-1], labels=Ns[:-1])
plt.colorbar()
plt.title("$\|\mathcal{W}^L(., t)  - \mathcal{W}(., t)\|$")
plt.tight_layout()
plt.savefig("figures/fig_W.pdf")
