"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EulerStep(nn.Module):
    def __init__(self, function, depth):
        super().__init__()
        self.function = function
        self.depth = depth

    def forward(self, x):
        depth = self.depth
        return x + self.function(x) / depth


class Net(nn.Module):
    def __init__(self, functions):
        super().__init__()
        for i, function in enumerate(functions):
            self.add_module(str(i), function)
        self.functions = functions
        self.depth = len(functions)
        depth = len(functions)
        self.nets = [EulerStep(functions[i], depth) for i in range(depth)]

    def forward(self, x):
        for net in self.nets:
            x = net(x.clone())
        return x

    def __getitem__(self, idx):
        return self.functions[idx]


class TinyResidual(nn.Module):
    expansion = 1

    def __init__(
        self, planes, n_layers, stride=1, use_relu=True, use_bn=True, non_lin="relu"
    ):
        super(TinyResidual, self).__init__()
        self.conv1 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            nn.init.constant_(self.bn2.weight, 0)
            nn.init.constant_(self.bn2.bias, 0)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            nn.init.constant_(self.conv2.weight, 0)
            nn.init.constant_(self.conv2.bias, 0)
            nn.init.xavier_uniform_(self.conv1.weight, gain=1e-4)
        # Fixup Init
        # nn.init.xavier_uniform_(self.conv2.weight, gain=1e-4)
        # nn.init.xavier_uniform_(self.conv1.weight, gain=1e-4)
        self.n_layers = n_layers
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.non_lin = non_lin
        if self.non_lin == "relu":
            self.non_lin = nn.ReLU()
        elif self.non_lin == "tanh":
            self.non_lin = nn.Tanh()
        elif self.non_lin == "softplus":
            self.non_lin = nn.Softplus()
        elif self.non_lin == "leaky_relu":
            self.non_lin = nn.LeakyReLU()
        elif self.non_lin == "gelu":
            self.non_lin = nn.GELU()
        else:
            self.non_lin = nn.Identity()

    def forward(self, x):
        out = self.non_lin(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out


class iTinyResnet(nn.Module):
    def __init__(
        self,
        n_layers,
        in_planes=64,
        num_classes=10,
        use_relu=True,
        use_bn=True,
        non_lin="relu",
    ):
        super(iTinyResnet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(
            3, in_planes, kernel_size=5, stride=3, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_planes)
        functions = [
            TinyResidual(
                in_planes,
                n_layers=n_layers,
                use_relu=use_relu,
                use_bn=use_bn,
                non_lin=non_lin,
            )
            for _ in range(n_layers)
        ]
        self.residual_layers = Net(functions)
        self.linear = nn.Linear(4 * in_planes, num_classes)
        self.n_layers = n_layers
        self.use_relu = use_relu

    def forward(self, x, early_exit=False):
        out = F.relu(self.bn1(self.conv1(x)))
        if early_exit:
            in_ = out.clone()
        out = self.residual_layers(out)
        if early_exit:
            return in_, out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    net = iTinyResnet(
        100,
        16,
        use_relu=True,
    )
    # net.eval()
    x = torch.randn(10, 3, 32, 32)
    in_, out_ = net(x, early_exit=True)
    # z = net.backward(out_)
    loss = (out_**2).sum()
    loss.backward()
    print(net.residual_layers[1].conv2.weight.grad[0, 0])
