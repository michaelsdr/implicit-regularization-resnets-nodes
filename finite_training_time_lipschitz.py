import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


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


# hidden layer size

set_all_seeds(2)

M = 32
d = 16

# Adjust the number of epochs here if you want to train for longer
n_epochs = 50
n = 100
W_to_repeat = torch.randn(d, M)

X = make_data(n, d=d)
Y = make_data(n, d=d)


def loss(pred):
    return ((pred - Y) ** 2).mean()


opt_indices = np.arange(1, n_epochs - 1, n_epochs // 10)

D = {}
Ns = [2, 16, 64, 256, 1014]
for N in Ns:
    V = torch.zeros(N, M, d)
    V = nn.Parameter(V)
    W = W_to_repeat.repeat(N, 1, 1)
    W = nn.Parameter(W)

    net = ResNet(V, W, d, N)

    optimizer = torch.optim.SGD(net.parameters(), lr=N * 0.01)

    V_list = []
    W_list = []

    for i in range(n_epochs):
        optimizer.zero_grad()
        V_list.append(V.clone().detach().numpy())
        W_list.append(W.clone().detach().numpy())
        pred = net(X)
        # output = loss(pred, x)
        output = loss(pred)
        output.backward()
        optimizer.step()

    for x in opt_indices:
        x = int(x)
        v = np.array(V_list)[:x, 1:, :, :] - np.array(V_list)[:x, :-1, :, :]
        w = np.array(W_list)[:x, 1:, :, :] - np.array(W_list)[:x, :-1, :, :]
        D[N, "V", x] = ((v**2).sum()) ** (1 / 2)
        D[N, "W", x] = ((w**2).sum()) ** (1 / 2)

p = 0.7
plt.figure(figsize=(p * 5, p * 4))

for i, x in enumerate(opt_indices):
    x = int(x)
    vs = []
    for N in Ns:
        vs.append(max(D[N, "V", x], D[N, "W", x]))
    color = plt.cm.viridis(i / len(opt_indices))
    plt.loglog(Ns, vs, label=str(int(opt_indices[i])), lw=2, color=color)
plt.xlabel("Depth $L$")
plt.ylabel("$\mathrm{max}_{k,t}(\|Z^L_k(t) - Z^L_{k+1}(t)\|_F)$")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/max_v_w.pdf")
