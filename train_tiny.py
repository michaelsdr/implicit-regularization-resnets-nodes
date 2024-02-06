"""Train CIFAR10 with PyTorch."""
import argparse
import os
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from implicit_regularization_resnets_nodes import iTinyResnet

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.03, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument("--adaptive", action="store_true")
parser.add_argument("--smooth_init", action="store_true")
parser.add_argument("--load_first_layer", action="store_true")
parser.add_argument("--no_bn", action="store_true")
parser.add_argument("--device", "-d", default=0, type=int)
parser.add_argument("--depth", default=8, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--planes", default=16, type=int)
parser.add_argument("--n_epochs", default=60, type=int)
parser.add_argument("--non_lin", default="relu", type=str)

args = parser.parse_args()
seed = args.seed
load_first_layer = args.load_first_layer
smooth_init = args.smooth_init
use_bn = False if args.no_bn else True
non_lin = args.non_lin
formatted_lr = "{:.0e}".format(args.lr)
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 10 if torch.cuda.is_available() else 0
if args.device > 0:
    device = "cuda:%d" % args.device
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root=".data/CIFAR10", train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=num_workers
)

testset = torchvision.datasets.CIFAR10(
    root=".data/CIFAR10", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=num_workers
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


bn_dict = {True: "with_bn", False: "no_bn"}
# Model
print("==> Building model..")
net = iTinyResnet(args.depth, in_planes=args.planes, use_bn=use_bn, non_lin=non_lin)

if smooth_init:
    print("we initialize the inner weights smoothly")
    state_dict = net.state_dict()
    modules = ["conv1", "conv2", "bn1", "bn2"] if use_bn else ["conv1", "conv2"]
    for i in range(1, args.depth):
        for n in modules:
            for m in ["weight", "bias"]:
                state_dict["residual_layers.%d.%s.%s" % (i, n, m)] = state_dict[
                    "residual_layers.0.%s.%s" % (n, m)
                ]
    net.load_state_dict(state_dict)

if load_first_layer:
    checkpoint_first_layers = "./checkpoint_with_0_depth/ckpt_lr_1e-01_depth_0_seed_0_use_bn_True_smooth_init_True_adaptive_True_non_lin_relu.pth"
    checkpoint_first_layers = torch.load(checkpoint_first_layers)["net"]
    state_dict = net.state_dict()
    for mod in [
        "conv1.weight",
        "bn1.weight",
        "bn1.bias",
        "linear.weight",
        "linear.bias",
    ]:
        state_dict[mod] = checkpoint_first_layers[mod]
    for mod in [
        "pre_residual.0.conv1.weight",
        "pre_residual.0.conv1.bias",
        "pre_residual.0.conv2.weight",
        "pre_residual.0.conv2.bias",
        "pre_residual.0.bn1.weight",
        "pre_residual.0.bn1.bias",
        "pre_residual.0.bn2.weight",
        "pre_residual.0.bn2.bias",
    ]:
        state_dict[mod] = checkpoint_first_layers[mod]
    net.load_state_dict(state_dict)

net = net.to(device)
cudnn.benchmark = True

checkpoint_name = (
    "./checkpoint/ckpt_lr_%s_depth_%s_seed_%s_use_bn_%s_smooth_init_%s_adaptive_%s_non_lin_%s.pth"
    % (formatted_lr, args.depth, seed, use_bn, smooth_init, args.adaptive, non_lin)
)

log_dict = {"test_loss": [], "train_loss": [], "test_acc": [], "train_acc": []}

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load(checkpoint_name)
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    log_dict = checkpoint["log_dict"]


criterion = nn.CrossEntropyLoss()

residual_parameters = []
other_parameters = []
for name, param in net.named_parameters():
    if "residual_layers" in name:
        residual_parameters.append(param)
    else:
        other_parameters.append(param)

lr = 1e-2 if args.adaptive else args.lr
optimizer = optim.SGD(other_parameters, lr=lr, momentum=0.9, weight_decay=5e-4)

residual_lr = args.lr
optimizer_residual = optim.SGD(
    residual_parameters, lr=residual_lr, momentum=0.9, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
scheduler_residual = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_residual, T_max=args.n_epochs
)


# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        optimizer_residual.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer_residual.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total)
    train_loss = train_loss / (batch_idx + 1)
    train_acc = 100.0 * correct / total
    log_dict["train_loss"].append(train_loss)
    log_dict["train_acc"].append(train_acc)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(test_loss / (batch_idx + 1), 100.0 * correct / total)

    # Save checkpoint.
    acc = 100.0 * correct / total
    test_loss = test_loss / (batch_idx + 1)
    test_acc = 100.0 * correct / total
    log_dict["test_loss"].append(test_loss)
    log_dict["test_acc"].append(test_acc)
    print("Saving..")
    state = {
        "net": net.state_dict(),
        "acc": acc,
        "epoch": epoch,
        "log_dict": log_dict,
    }
    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    torch.save(state, checkpoint_name)
    best_acc = acc


for epoch in range(start_epoch, start_epoch + args.n_epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
    scheduler_residual.step()
    for param_groups in optimizer.param_groups:
        print("lr= ", param_groups["lr"])
    for param_groups in optimizer_residual.param_groups:
        print("residual lr= ", param_groups["lr"])
