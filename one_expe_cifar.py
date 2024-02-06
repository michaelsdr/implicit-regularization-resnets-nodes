import argparse
import os

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--depth", default=128, type=int)
parser.add_argument("--smooth_init", default="true", type=str)
parser.add_argument("--use_bn", default="true", type=str)
parser.add_argument("--adaptive", default="false", type=str)
parser.add_argument("--non_lin", default="relu", type=str)
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")

args = parser.parse_args()

smooth_init = args.smooth_init.lower() == "true"
use_bn = args.use_bn.lower() == "true"
adaptive = args.adaptive.lower() == "true"
non_lin = args.non_lin
lr = args.lr
depth = args.depth
seed = args.seed
n_epochs = 180
argument = (
    "python train_tiny.py --lr %f --depth %d --n_epochs %d --seed %d --non_lin %s"
    % (lr, depth, n_epochs, seed, non_lin)
)

if smooth_init:
    argument += " --smooth_init"
if use_bn is False:
    argument += " --no_bn"
if adaptive:
    argument += " --adaptive"

print(argument)

if __name__ == "__main__":
    os.system(argument)
