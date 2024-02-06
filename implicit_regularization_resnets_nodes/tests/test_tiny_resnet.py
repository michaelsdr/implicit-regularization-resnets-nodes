import torch
import torch.nn as nn


from implicit_regularization_resnets_nodes import iTinyResnet

torch.manual_seed(1)


def test_dimension_layers():
    net = iTinyResnet(
        10,
        16,
        use_relu=True,
    )
    x = torch.randn(10, 3, 32, 32)
    in_, out_ = net(x, early_exit=True)
    loss = (out_**2).sum()
    loss.backward()
