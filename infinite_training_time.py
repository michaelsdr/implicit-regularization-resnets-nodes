import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Adjust the number of training epochs here
n_epochs = 10000


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

M = 64
d = 16
W_to_repeat = torch.randn(d, M)
n = 50

B = torch.eye(d, d) + 0.4 * torch.randn(d, d)
X = make_data(n, d=d)
Y = make_data(n, d=d)

N = 64

V_list = []
W_list = []


def loss(pred, Y):
    return ((pred - Y) ** 2).mean()


losses = []


V = torch.zeros(N, M, d)
V = nn.Parameter(V)
W = W_to_repeat.repeat(N, 1, 1)
W = nn.Parameter(W)
net = ResNet(V, W, d, N)

optimizer = torch.optim.SGD(net.parameters(), lr=N * 0.005)


V_t = []
W_t = []
for k in range(n_epochs):
    V_t.append(V.clone().detach())
    W_t.append(W.clone().detach())
    optimizer.zero_grad()
    output = loss(net(X), Y)
    output.backward()
    optimizer.step()
    losses.append(output)

V_list.append(V_t)
W_list.append(W_t)
lw = 0.5
plot_epochs = np.arange(0, len(V_t), 100)
p = 0.7
k_plot = 4
num_colors = len(plot_epochs)
os.makedirs("figures", exist_ok=True)
for i in range(k_plot):  # M
    for j in range(k_plot):  # d
        plt.figure(figsize=(4 * p, 2.5 * p))
        _ = 0
        for k, a in enumerate(V_list[-1]):
            if k in plot_epochs:
                color = plt.cm.viridis(_ / num_colors)
                plt.plot(a[:, i, j], color=color, lw=lw)
                _ += 1
        plt.xlabel("Layer $k$")
        plt.ylabel("A coeff. of ${V}_k^L(t)$")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("figures/V_%d_%d.pdf" % (i, j))


num_colors = len(plot_epochs)
for i in range(k_plot):  # d
    for j in range(k_plot):  # M
        plt.figure(figsize=(4 * p, 2.5 * p))
        _ = 0
        for k, a in enumerate(W_list[-1]):
            if k in plot_epochs:
                color = plt.cm.viridis(_ / num_colors)
                plt.plot(a[:, i, j], color=color, lw=lw)

                _ += 1
        plt.xlabel("Layer $k$")
        plt.ylabel("A coeff. of ${W}_k^L(t)$")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("figures/W_%d_%d.pdf" % (i, j))

p = 0.8
plt.figure(figsize=(3 * p, 2 * p))
plt.semilogy(np.arange(n_epochs)[::10], torch.tensor(losses[::10]))
plt.xlabel("Training iteration")
plt.ylabel("$\ell^L$")
plt.yticks(
    ticks=[10 ** (-i) for i in [1, 4, 7]],
    labels=["$10^{-1}$", "$10^{-4}$", "$10^{-7}$"],
)
plt.tight_layout()
plt.grid(True)
plt.savefig("figures/losses.pdf")
